//! GPU acceleration module using WGPU (WebGPU)
//!
//! This module provides cross-platform GPU compute via WGPU, supporting:
//! - NVIDIA GPUs (via Vulkan backend)
//! - AMD GPUs (via Vulkan or DX12)
//! - Intel GPUs (via Vulkan or DX12)
//! - Apple Silicon (via Metal)
//!
//! All shaders are written in WGSL (WebGPU Shading Language).

pub mod backend;
pub mod clahe_gpu;
pub mod edt_gpu;
pub mod glcm_gpu;
pub mod hog_gpu;

pub use backend::{gpu_profiling_enabled, is_gpu_available, log_gpu_profile, GpuBackend};
pub use clahe_gpu::compute_clahe_wgpu;
pub use edt_gpu::compute_edt_wgpu;
pub use glcm_gpu::compute_glcm_wgpu;
pub use hog_gpu::compute_hog_wgpu;
