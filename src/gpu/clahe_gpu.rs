//! GPU-accelerated CLAHE using WGPU.

use bytemuck::{Pod, Zeroable};
use ndarray::{Array2, ArrayView2};
use std::time::Instant;
use wgpu::util::DeviceExt;

use crate::core::{FeaturizerError, Result};
use crate::gpu::backend::{gpu_profiling_enabled, log_gpu_profile, GpuBackend};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ClaheParams {
    width: u32,
    height: u32,
    kernel_size: u32,
    n_tiles_x: u32,
    n_tiles_y: u32,
    clip_limit: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Compute CLAHE on GPU for a u8 grayscale image.
pub fn compute_clahe_wgpu(
    image: &ArrayView2<u8>,
    clip_limit: f32,
    kernel_size: usize,
    backend: &GpuBackend,
) -> Result<Array2<u8>> {
    let (h, w) = image.dim();
    if h == 0 || w == 0 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "Non-zero image dimensions".to_string(),
            got: format!("({}, {})", h, w),
        });
    }
    if kernel_size == 0 {
        return Err(FeaturizerError::InvalidInput(
            "kernel_size must be >= 1".to_string(),
        ));
    }
    let profile = gpu_profiling_enabled();
    let total_start = Instant::now();

    let n_pixels = h * w;
    let n_tiles_y = h.div_ceil(kernel_size);
    let n_tiles_x = w.div_ceil(kernel_size);

    let pack_start = Instant::now();
    let image_flat: Vec<u32> = image.iter().map(|&v| v as u32).collect();

    let params = ClaheParams {
        width: w as u32,
        height: h as u32,
        kernel_size: kernel_size as u32,
        n_tiles_x: n_tiles_x as u32,
        n_tiles_y: n_tiles_y as u32,
        clip_limit,
        _pad0: 0,
        _pad1: 0,
    };
    if profile {
        log_gpu_profile("clahe", "pack_inputs", pack_start.elapsed());
    }

    let setup_start = Instant::now();
    let shader = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CLAHE Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/clahe.wgsl").into()),
        });

    let params_buffer = backend
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CLAHE Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let image_buffer = backend
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CLAHE Image Buffer"),
            contents: bytemuck::cast_slice(&image_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CLAHE Output Buffer"),
        size: (n_pixels * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout =
        backend
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CLAHE Bind Group Layout"),
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

    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CLAHE Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

    let pipeline_layout = backend
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CLAHE Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CLAHE Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clahe_compute",
        });
    if profile {
        log_gpu_profile("clahe", "setup_buffers", setup_start.elapsed());
    }

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CLAHE Encoder"),
        });
    let dispatch_start = Instant::now();

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("CLAHE Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (w as u32 + 7) / 8;
        let wg_y = (h as u32 + 7) / 8;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    backend.queue.submit(Some(encoder.finish()));

    let staging_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CLAHE Staging Buffer"),
        size: (n_pixels * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut copy_encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CLAHE Copy Encoder"),
        });
    copy_encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        (n_pixels * std::mem::size_of::<u32>()) as u64,
    );
    backend.queue.submit(Some(copy_encoder.finish()));
    if profile {
        log_gpu_profile("clahe", "dispatch_copy_submit", dispatch_start.elapsed());
    }

    let map_start = Instant::now();
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    backend.device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver.receive())
        .ok_or_else(|| {
            FeaturizerError::CudaError("Failed to receive CLAHE map result".to_string())
        })?
        .map_err(|e| FeaturizerError::CudaError(format!("CLAHE buffer map error: {:?}", e)))?;
    if profile {
        log_gpu_profile("clahe", "map_readback", map_start.elapsed());
    }

    let post_start = Instant::now();
    let mapped = slice.get_mapped_range();
    let out_u32: &[u32] = bytemuck::cast_slice(&mapped);
    let out_u8: Vec<u8> = out_u32.iter().map(|&v| v.min(255) as u8).collect();
    if profile {
        log_gpu_profile("clahe", "cpu_postprocess", post_start.elapsed());
        log_gpu_profile("clahe", "total", total_start.elapsed());
    }
    drop(mapped);
    staging_buffer.unmap();

    Array2::from_shape_vec((h, w), out_u8)
        .map_err(|e| FeaturizerError::NumericalError(format!("Invalid CLAHE output shape: {e}")))
}
