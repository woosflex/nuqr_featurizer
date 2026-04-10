//! GPU-accelerated GLCM computation using WGPU.
//!
//! Computes 4-angle symmetric co-occurrence matrices for 8-bit grayscale images.

use bytemuck::{Pod, Zeroable};
use ndarray::{Array4, ArrayView2};
use std::sync::OnceLock;
use std::time::Instant;
use wgpu::util::DeviceExt;

use crate::core::{FeaturizerError, Result};
use crate::gpu::backend::{gpu_profiling_enabled, log_gpu_profile, GpuBackend};

const LEVELS: usize = 256;
const NUM_ANGLES: usize = 4;
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
static GLCM_PIPELINE: OnceLock<GlcmPipeline> = OnceLock::new();

struct GlcmPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlcmParams {
    width: u32,
    height: u32,
    symmetric: u32,
    _pad: u32,
}

/// Compute a 4D GLCM tensor `[levels, levels, 1, 4]` on GPU.
pub fn compute_glcm_wgpu(
    image: &ArrayView2<u8>,
    mask: &ArrayView2<bool>,
    symmetric: bool,
    normed: bool,
    backend: &GpuBackend,
) -> Result<Array4<f32>> {
    let (height, width) = image.dim();
    if height == 0 || width == 0 {
        return Err(FeaturizerError::InvalidInput(
            "GLCM image must have non-zero dimensions".to_string(),
        ));
    }
    if mask.dim() != (height, width) {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("({}, {})", height, width),
            got: format!("({}, {})", mask.dim().0, mask.dim().1),
        });
    }
    let profile = gpu_profiling_enabled();
    let total_start = Instant::now();

    let pack_start = Instant::now();
    let mut image_flat = Vec::<u32>::with_capacity(height * width);
    let mut mask_flat = Vec::<u32>::with_capacity(height * width);
    for (&pixel, &is_masked) in image.iter().zip(mask.iter()) {
        image_flat.push(pixel as u32);
        mask_flat.push(if is_masked { 1_u32 } else { 0_u32 });
    }
    if profile {
        log_gpu_profile("glcm", "pack_inputs", pack_start.elapsed());
    }
    let glcm_len = LEVELS * LEVELS * NUM_ANGLES;
    let glcm_pipeline = get_or_create_glcm_pipeline(&backend.device);

    let setup_start = Instant::now();
    let image_buffer = backend
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GLCM Image Buffer"),
            contents: bytemuck::cast_slice(&image_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mask_buffer = backend
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GLCM Mask Buffer"),
            contents: bytemuck::cast_slice(&mask_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let params = GlcmParams {
        width: width as u32,
        height: height as u32,
        symmetric: if symmetric { 1 } else { 0 },
        _pad: 0,
    };
    let params_buffer = backend
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GLCM Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let glcm_counts_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GLCM Counts Buffer"),
        size: (glcm_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GLCM Bind Group"),
            layout: &glcm_pipeline.bind_group_layout,
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
                    resource: mask_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: glcm_counts_buffer.as_entire_binding(),
                },
            ],
        });
    if profile {
        log_gpu_profile("glcm", "setup_buffers", setup_start.elapsed());
    }

    let staging_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GLCM Staging Buffer"),
        size: (glcm_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GLCM Encoder"),
        });
    let dispatch_start = Instant::now();
    encoder.clear_buffer(&glcm_counts_buffer, 0, None);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GLCM Accumulate Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&glcm_pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let wg_y = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(
        &glcm_counts_buffer,
        0,
        &staging_buffer,
        0,
        (glcm_len * std::mem::size_of::<u32>()) as u64,
    );
    backend.queue.submit(Some(encoder.finish()));
    if profile {
        log_gpu_profile("glcm", "dispatch_copy_submit", dispatch_start.elapsed());
    }

    let map_start = Instant::now();
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    backend.device.poll(wgpu::Maintain::Wait);

    pollster::block_on(receiver.receive())
        .ok_or_else(|| FeaturizerError::CudaError("Failed to receive map result".to_string()))?
        .map_err(|e| FeaturizerError::CudaError(format!("GLCM map error: {:?}", e)))?;
    if profile {
        log_gpu_profile("glcm", "map_readback", map_start.elapsed());
    }

    let post_start = Instant::now();
    let mapped = slice.get_mapped_range();
    let counts: &[u32] = bytemuck::cast_slice(&mapped);

    let mut glcm = Array4::<f32>::zeros((LEVELS, LEVELS, 1, NUM_ANGLES));
    for angle in 0..NUM_ANGLES {
        let base = angle * LEVELS * LEVELS;
        let plane = &counts[base..base + LEVELS * LEVELS];
        let plane_sum: u64 = plane
            .iter()
            .map(|&v| v as u64)
            .sum::<u64>()
            .saturating_sub(plane[0] as u64);
        let inv_sum = if normed && plane_sum > 0 {
            1.0_f32 / plane_sum as f32
        } else {
            1.0_f32
        };

        for i in 0..LEVELS {
            for j in 0..LEVELS {
                let idx = i * LEVELS + j;
                let raw = if idx == 0 { 0.0 } else { plane[idx] as f32 };
                glcm[[i, j, 0, angle]] = if normed { raw * inv_sum } else { raw };
            }
        }
    }
    if profile {
        log_gpu_profile("glcm", "cpu_postprocess", post_start.elapsed());
        log_gpu_profile("glcm", "total", total_start.elapsed());
    }

    drop(mapped);
    staging_buffer.unmap();

    Ok(glcm)
}

fn get_or_create_glcm_pipeline(device: &wgpu::Device) -> &'static GlcmPipeline {
    GLCM_PIPELINE.get_or_init(|| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GLCM Shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/glcm.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GLCM Bind Group Layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GLCM Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GLCM Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "glcm_accumulate",
        });

        GlcmPipeline {
            bind_group_layout,
            pipeline,
        }
    })
}
