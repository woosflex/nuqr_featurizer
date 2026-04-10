//! GPU EDT entrypoint placeholder.
//!
//! The previous brute-force O(N^2) shader implementation is intentionally
//! disabled because it can trigger driver timeout detection/recovery on
//! real GPUs. Callers should use the CPU EDT path until a jump-flooding (JFA)
//! or equivalent bounded-time GPU algorithm is implemented.

use ndarray::{Array2, ArrayView2};

use crate::core::{FeaturizerError, Result};
use crate::gpu::backend::GpuBackend;

/// Compute EDT on GPU for a boolean mask.
///
/// This path is currently disabled for safety.
pub fn compute_edt_wgpu(_mask: &ArrayView2<bool>, _backend: &GpuBackend) -> Result<Array2<f64>> {
    Err(FeaturizerError::CudaError(
        "GPU EDT disabled: brute-force O(N^2) shader risks GPU TDR; use CPU EDT path".to_string(),
    ))
}
