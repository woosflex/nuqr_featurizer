//! WGPU backend initialization and device management
//!
//! This module handles:
//! - Device and queue creation
//! - Buffer allocation and management
//! - Shader module compilation
//! - Compute pipeline setup

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use wgpu;

use crate::core::FeaturizerError;
use crate::core::Result;

/// Global GPU backend instance (initialized lazily)
static GPU_BACKEND: OnceLock<Arc<Mutex<Option<GpuBackend>>>> = OnceLock::new();
static GPU_PROFILING_ENABLED: OnceLock<bool> = OnceLock::new();

/// GPU backend wrapping WGPU device and queue
pub struct GpuBackend {
    /// WGPU logical device used to create buffers/pipelines and submit work.
    pub device: wgpu::Device,
    /// WGPU queue used for command submission and buffer transfers.
    pub queue: wgpu::Queue,
}

impl GpuBackend {
    /// Initialize WGPU backend with default adapter
    ///
    /// # Returns
    /// * `Ok(GpuBackend)` if a compatible GPU is found
    /// * `Err` if no GPU is available or initialization fails
    pub async fn new() -> Result<Self> {
        // Create WGPU instance with default backends (Vulkan, Metal, DX12, GLES)
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                FeaturizerError::CudaError("No compatible GPU adapter found".to_string())
            })?;

        // Log adapter info
        let info = adapter.get_info();
        tracing::info!(
            "WGPU adapter: {} ({:?}, backend: {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("NuQR Featurizer GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| FeaturizerError::CudaError(format!("Failed to create device: {}", e)))?;

        Ok(Self { device, queue })
    }

    /// Get or initialize the global GPU backend
    pub fn get_or_init() -> Result<Arc<Mutex<Option<GpuBackend>>>> {
        let backend = GPU_BACKEND.get_or_init(|| {
            // Try to initialize GPU backend
            let backend = pollster::block_on(GpuBackend::new());
            match backend {
                Ok(b) => {
                    tracing::info!("GPU backend initialized successfully");
                    Arc::new(Mutex::new(Some(b)))
                }
                Err(e) => {
                    tracing::warn!("GPU backend initialization failed: {}", e);
                    Arc::new(Mutex::new(None))
                }
            }
        });
        Ok(Arc::clone(backend))
    }
}

/// Check if GPU is available and initialized
pub fn is_gpu_available() -> bool {
    if let Ok(backend) = GpuBackend::get_or_init() {
        backend.lock().unwrap().is_some()
    } else {
        false
    }
}

/// Returns true when GPU profiling logs are enabled via `NUQR_GPU_PROFILE`.
///
/// Enabled values are: `1`, `true`, `yes`, `on` (case-insensitive).
pub fn gpu_profiling_enabled() -> bool {
    *GPU_PROFILING_ENABLED.get_or_init(|| {
        std::env::var("NUQR_GPU_PROFILE")
            .map(|v| {
                let s = v.trim().to_ascii_lowercase();
                matches!(s.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

/// Emit a per-stage GPU profiling log entry when profiling is enabled.
pub fn log_gpu_profile(kernel: &str, stage: &str, elapsed: Duration) {
    if gpu_profiling_enabled() {
        tracing::info!(
            target: "nuqr_gpu_profile",
            kernel = kernel,
            stage = stage,
            elapsed_ms = elapsed.as_secs_f64() * 1000.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        // This test checks if GPU can be initialized
        // It will pass even if no GPU is available (just logs a warning)
        let available = is_gpu_available();
        println!("GPU available: {}", available);
    }

    #[test]
    fn test_backend_initialization() {
        // Test that we can get the backend without panicking
        let result = GpuBackend::get_or_init();
        assert!(result.is_ok());
    }
}
