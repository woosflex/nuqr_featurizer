//! GPU-accelerated HOG computation using WGPU
//!
//! This module implements HOG feature extraction using WebGPU compute shaders.
//! It uses 3 shader passes: gradient computation, histogram accumulation, and normalization.

use bytemuck::{Pod, Zeroable};
use std::sync::OnceLock;
use std::time::Instant;
use wgpu;

use crate::core::FeaturizerError;
use crate::core::Result;
use crate::gpu::backend::{gpu_profiling_enabled, log_gpu_profile, GpuBackend};

const ORIENTATIONS: usize = 8;
const CELL_SIZE: usize = 8;
static HOG_PIPELINES: OnceLock<HogPipelines> = OnceLock::new();

struct HogPipelines {
    grad_bind_group_layout: wgpu::BindGroupLayout,
    grad_pipeline: wgpu::ComputePipeline,
    hist_bind_group_layout: wgpu::BindGroupLayout,
    hist_pipeline: wgpu::ComputePipeline,
    norm_bind_group_layout: wgpu::BindGroupLayout,
    norm_pipeline: wgpu::ComputePipeline,
}

/// Parameters for gradient computation shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GradientParams {
    width: u32,
    height: u32,
}

/// Parameters for histogram accumulation shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HistogramParams {
    width: u32,
    height: u32,
    n_cells_row: u32,
    n_cells_col: u32,
}

/// Parameters for normalization shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NormalizeParams {
    n_cells: u32,
    _padding: [u32; 3], // Pad to 16 bytes for uniform buffer alignment
}

/// Compute HOG features on GPU using WGPU
///
/// # Arguments
/// * `image_data` - Grayscale image (f32 values)
/// * `mask_data` - Binary mask flattened as u32 (1 = nucleus, 0 = background)
/// * `height` - Image height in pixels
/// * `width` - Image width in pixels
/// * `backend` - GPU backend (device + queue)
///
/// # Returns
/// * HOG descriptor as `Vec<f32>` with length = n_cells * 8
pub fn compute_hog_wgpu(
    image_data: &[f32],
    mask_data: &[u32],
    height: usize,
    width: usize,
    backend: &GpuBackend,
) -> Result<Vec<f32>> {
    let n_cells_row = height / CELL_SIZE;
    let n_cells_col = width / CELL_SIZE;

    if n_cells_row == 0 || n_cells_col == 0 {
        return Err(FeaturizerError::InvalidInput(
            "Image too small for HOG (need at least 8x8 pixels)".to_string(),
        ));
    }

    let n_cells = n_cells_row * n_cells_col;
    let n_pixels = height * width;
    if image_data.len() != n_pixels || mask_data.len() != n_pixels {
        return Err(FeaturizerError::InvalidInput(
            "HOG GPU input length mismatch".to_string(),
        ));
    }
    let profile = gpu_profiling_enabled();
    let total_start = Instant::now();

    let pipelines = get_or_create_hog_pipelines(&backend.device);

    // Create buffers
    let setup_start = Instant::now();
    let image_buffer = create_buffer_from_data(&backend.device, image_data, "Image Buffer");
    let mask_buffer = create_buffer_from_data(&backend.device, mask_data, "Mask Buffer");

    let gradient_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gradient Buffer"),
        size: (n_pixels * std::mem::size_of::<[f32; 2]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let histogram_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Buffer"),
        size: (n_cells * ORIENTATIONS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if profile {
        log_gpu_profile("hog", "setup_buffers", setup_start.elapsed());
    }

    // === Pass 1: Gradient Computation ===
    let grad_params = GradientParams {
        width: width as u32,
        height: height as u32,
    };

    let grad_params_buffer =
        create_buffer_from_data(&backend.device, &[grad_params], "Gradient Params");

    let grad_bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group"),
            layout: &pipelines.grad_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_buffer.as_entire_binding(),
                },
            ],
        });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("HOG Command Encoder"),
        });
    let dispatch_start = Instant::now();

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gradient Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.grad_pipeline);
        pass.set_bind_group(0, &grad_bind_group, &[]);

        // Dispatch: ceil(width/16) x ceil(height/16)
        let workgroups_x = (width as u32 + 15) / 16;
        let workgroups_y = (height as u32 + 15) / 16;
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // === Pass 2: Histogram Accumulation ===
    let hist_params = HistogramParams {
        width: width as u32,
        height: height as u32,
        n_cells_row: n_cells_row as u32,
        n_cells_col: n_cells_col as u32,
    };
    let hist_params_buffer =
        create_buffer_from_data(&backend.device, &[hist_params], "Histogram Params");

    let hist_bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Histogram Bind Group"),
            layout: &pipelines.hist_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hist_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gradient_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: histogram_buffer.as_entire_binding(),
                },
            ],
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Histogram Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.hist_pipeline);
        pass.set_bind_group(0, &hist_bind_group, &[]);
        pass.dispatch_workgroups(n_cells_col as u32, n_cells_row as u32, 1);
    }

    // === Pass 3: Normalization ===
    let norm_params = NormalizeParams {
        n_cells: n_cells as u32,
        _padding: [0; 3],
    };
    let norm_params_buffer =
        create_buffer_from_data(&backend.device, &[norm_params], "Normalize Params");

    let norm_bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Normalize Bind Group"),
            layout: &pipelines.norm_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: norm_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram_buffer.as_entire_binding(),
                },
            ],
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Normalize Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.norm_pipeline);
        pass.set_bind_group(0, &norm_bind_group, &[]);
        let norm_workgroups_x = ((n_cells as u32) + 255) / 256;
        pass.dispatch_workgroups(norm_workgroups_x, 1, 1);
    }

    backend.queue.submit(Some(encoder.finish()));
    if profile {
        log_gpu_profile("hog", "dispatch_compute", dispatch_start.elapsed());
    }

    // Read back results
    let map_start = Instant::now();
    let staging_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (n_cells * ORIENTATIONS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

    encoder.copy_buffer_to_buffer(
        &histogram_buffer,
        0,
        &staging_buffer,
        0,
        (n_cells * ORIENTATIONS * std::mem::size_of::<f32>()) as u64,
    );

    backend.queue.submit(Some(encoder.finish()));

    // Map and read
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    backend.device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver.receive())
        .ok_or_else(|| {
            FeaturizerError::CudaError("Failed to receive buffer map result".to_string())
        })?
        .map_err(|e| FeaturizerError::CudaError(format!("Buffer map error: {:?}", e)))?;
    if profile {
        log_gpu_profile("hog", "copy_and_map_readback", map_start.elapsed());
    }

    let post_start = Instant::now();
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    if profile {
        log_gpu_profile("hog", "cpu_postprocess", post_start.elapsed());
        log_gpu_profile("hog", "total", total_start.elapsed());
    }

    drop(data);
    staging_buffer.unmap();

    Ok(result)
}

fn get_or_create_hog_pipelines(device: &wgpu::Device) -> &'static HogPipelines {
    HOG_PIPELINES.get_or_init(|| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HOG Shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hog.wgsl").into()),
        });

        let grad_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gradient Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let grad_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gradient Pipeline Layout"),
            bind_group_layouts: &[&grad_bind_group_layout],
            push_constant_ranges: &[],
        });
        let grad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gradient Pipeline"),
            layout: Some(&grad_pipeline_layout),
            module: &shader,
            entry_point: "gradient_compute",
        });

        let hist_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Histogram Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let hist_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Histogram Pipeline Layout"),
            bind_group_layouts: &[&hist_bind_group_layout],
            push_constant_ranges: &[],
        });
        let hist_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Histogram Pipeline"),
            layout: Some(&hist_pipeline_layout),
            module: &shader,
            entry_point: "histogram_accumulate",
        });

        let norm_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Normalize Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let norm_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Normalize Pipeline Layout"),
            bind_group_layouts: &[&norm_bind_group_layout],
            push_constant_ranges: &[],
        });
        let norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Normalize Pipeline"),
            layout: Some(&norm_pipeline_layout),
            module: &shader,
            entry_point: "normalize_l2hys",
        });

        HogPipelines {
            grad_bind_group_layout,
            grad_pipeline,
            hist_bind_group_layout,
            hist_pipeline,
            norm_bind_group_layout,
            norm_pipeline,
        }
    })
}

/// Helper: Create a WGPU buffer from Rust data
fn create_buffer_from_data<T: Pod>(device: &wgpu::Device, data: &[T], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
    })
}
