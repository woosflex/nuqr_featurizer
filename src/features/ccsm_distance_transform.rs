//! CPU Euclidean Distance Transform (EDT) utilities for CCSM.
//!
//! Port target: `scipy.ndimage.distance_transform_edt` usage in the Python
//! reference (`calculate_ccsm_features`).

use image::{GrayImage, Luma};
use imageproc::distance_transform::euclidean_squared_distance_transform;
use ndarray::{Array2, ArrayView2};

use crate::core::Result;
use crate::gpu::is_gpu_available;

const EDT_GPU_DISABLED_REASON: &str =
    "GPU EDT is disabled: brute-force O(N^2) shader risks driver timeout; using CPU EDT";

/// Compute Euclidean distance transform of a binary mask.
///
/// Behavior matches `scipy.ndimage.distance_transform_edt(mask)` where
/// non-zero values in `mask` are considered foreground and distance is measured
/// to the nearest zero-valued pixel.
///
/// For a nucleus mask (`true` = nucleus), the result is:
/// - `0` outside nucleus
/// - positive distances inside nucleus to nearest boundary/background
pub fn euclidean_distance_transform(mask: &ArrayView2<bool>) -> Result<Array2<f64>> {
    euclidean_distance_transform_with_gpu(mask, false)
}

/// Compute Euclidean distance transform with optional WGPU acceleration.
pub fn euclidean_distance_transform_with_gpu(
    mask: &ArrayView2<bool>,
    use_gpu: bool,
) -> Result<Array2<f64>> {
    if use_gpu && is_gpu_available() {
        eprintln!("{EDT_GPU_DISABLED_REASON}");
    }
    euclidean_distance_transform_cpu(mask)
}

/// Compute EDT for a batch of masks, reusing a single GPU backend when enabled.
pub fn euclidean_distance_transform_batch_with_gpu(
    masks: &[Array2<bool>],
    use_gpu: bool,
) -> Result<Vec<Array2<f64>>> {
    if masks.is_empty() {
        return Ok(Vec::new());
    }

    if use_gpu && is_gpu_available() {
        eprintln!("{EDT_GPU_DISABLED_REASON}");
    }

    masks
        .iter()
        .map(|mask| euclidean_distance_transform_cpu(&mask.view()))
        .collect()
}

fn euclidean_distance_transform_cpu(mask: &ArrayView2<bool>) -> Result<Array2<f64>> {
    let (h, w) = mask.dim();
    let mut inverted = GrayImage::new(w as u32, h as u32);

    // imageproc EDT computes distance to nearest foreground pixel (non-zero).
    // To match scipy distance_transform_edt(mask) where distance is measured
    // from nucleus pixels to background, we mark original background as
    // foreground in the inverted image.
    for r in 0..h {
        for c in 0..w {
            let value = if mask[[r, c]] { 0_u8 } else { 255_u8 };
            inverted.put_pixel(c as u32, r as u32, Luma([value]));
        }
    }

    let dist2 = euclidean_squared_distance_transform(&inverted);
    let mut out = Array2::<f64>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            out[[r, c]] = dist2.get_pixel(c as u32, r as u32).0[0].sqrt();
        }
    }

    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_mask_all_zeros() {
        let mask = Array2::<bool>::from_elem((6, 6), false);
        let edt = euclidean_distance_transform(&mask.view()).unwrap();
        assert!(edt.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_single_pixel_distance_is_one() {
        let mut mask = Array2::<bool>::from_elem((7, 7), false);
        mask[[3, 3]] = true;
        let edt = euclidean_distance_transform(&mask.view()).unwrap();

        assert!((edt[[3, 3]] - 1.0).abs() < 1e-12);
        assert_eq!(edt[[0, 0]], 0.0);
    }

    #[test]
    fn test_square_center_has_larger_distance() {
        let mut mask = Array2::<bool>::from_elem((9, 9), false);
        for r in 2..7 {
            for c in 2..7 {
                mask[[r, c]] = true;
            }
        }

        let edt = euclidean_distance_transform(&mask.view()).unwrap();
        assert_eq!(edt[[1, 1]], 0.0);
        assert!(edt[[4, 4]] > edt[[2, 2]]);
        assert!(edt[[4, 4]] >= 2.0);
    }

    #[test]
    fn test_gpu_flag_small_mask_matches_cpu() {
        let mut mask = Array2::<bool>::from_elem((12, 12), false);
        for r in 3..9 {
            for c in 3..9 {
                mask[[r, c]] = true;
            }
        }

        let cpu = euclidean_distance_transform_with_gpu(&mask.view(), false).unwrap();
        let gpu_flag = euclidean_distance_transform_with_gpu(&mask.view(), true).unwrap();
        assert_eq!(cpu, gpu_flag);
    }

    #[test]
    fn test_batch_gpu_flag_small_masks_match_cpu() {
        let mut mask1 = Array2::<bool>::from_elem((12, 12), false);
        let mut mask2 = Array2::<bool>::from_elem((12, 12), false);
        for r in 3..9 {
            for c in 3..9 {
                mask1[[r, c]] = true;
            }
        }
        for r in 2..10 {
            for c in 4..8 {
                mask2[[r, c]] = true;
            }
        }
        let masks = vec![mask1, mask2];

        let cpu = euclidean_distance_transform_batch_with_gpu(&masks, false).unwrap();
        let gpu_flag = euclidean_distance_transform_batch_with_gpu(&masks, true).unwrap();
        assert_eq!(cpu, gpu_flag);
    }
}
