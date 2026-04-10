//! Gray-Level Co-occurrence Matrix (GLCM) texture features.
//!
//! Port of scikit-image `skimage.feature.graycomatrix` and `skimage.feature.graycoprops`.
//!
//! Computes 6 properties from GLCM:
//! - contrast, dissimilarity, homogeneity
//! - energy, correlation, ASM (Angular Second Moment)
//!
//! Uses f32 precision per precision_guidelines.md (pixel-level operations).

use ndarray::{s, Array2, Array4, ArrayView2, ArrayView4};
use std::collections::HashMap;
use std::f32::consts::PI;

use crate::core::{FeaturizerError, Result};
use crate::gpu::{compute_glcm_wgpu, is_gpu_available, GpuBackend};

/// GLCM angle in radians
const ANGLES: [f32; 4] = [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0];
const GPU_MIN_SIZE: usize = 40;

/// Calculate GLCM features from a grayscale patch with mask.
///
/// Matches Python reference (Final_Code_Features_13.10.py, lines 262-268):
/// ```python
/// glcm = feature.graycomatrix(img_for_glcm, distances=[1], angles=[0, π/4, π/2, 3π/4],
///                               symmetric=True, normed=True, levels=256)
/// ```
///
/// Returns 6 features (mean across 4 angles):
/// - glcm_contrast, glcm_dissimilarity, glcm_homogeneity
/// - glcm_energy, glcm_correlation, glcm_ASM
pub fn calculate_glcm_features(
    grayscale_patch: &ArrayView2<f32>,
    mask: &ArrayView2<bool>,
) -> Result<HashMap<String, f64>> {
    calculate_glcm_features_with_gpu(grayscale_patch, mask, false)
}

/// Calculate GLCM features with optional WGPU acceleration.
pub fn calculate_glcm_features_with_gpu(
    grayscale_patch: &ArrayView2<f32>,
    mask: &ArrayView2<bool>,
    use_gpu: bool,
) -> Result<HashMap<String, f64>> {
    if grayscale_patch.shape() != mask.shape() {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("{:?}", mask.shape()),
            got: format!("{:?}", grayscale_patch.shape()),
        });
    }

    // Build uint8 image from grayscale patch.
    let mut img_u8 = Array2::<u8>::zeros(grayscale_patch.dim());
    for ((r, c), &val) in grayscale_patch.indexed_iter() {
        img_u8[[r, c]] = val.clamp(0.0, 255.0) as u8;
    }

    // Edge case: empty or single non-zero intensity inside mask.
    // Mirrors Python reference logic that ignores zeros for this check.
    let mut seen = [false; 256];
    let mut unique_nonzero = 0usize;
    for ((r, c), &v) in img_u8.indexed_iter() {
        if !mask[[r, c]] || v == 0 {
            continue;
        }
        let idx = v as usize;
        if !seen[idx] {
            seen[idx] = true;
            unique_nonzero += 1;
            if unique_nonzero >= 2 {
                break;
            }
        }
    }
    if unique_nonzero < 2 {
        return Ok(empty_features());
    }

    let (h, w) = grayscale_patch.dim();

    // Compute GLCM for 4 angles, distance=1.
    let glcm = if use_gpu && h >= GPU_MIN_SIZE && w >= GPU_MIN_SIZE && is_gpu_available() {
        match compute_glcm_gpu(&img_u8.view(), mask) {
            Ok(gpu_glcm) => gpu_glcm,
            Err(e) => {
                eprintln!("GPU GLCM failed ({}), falling back to CPU", e);
                graycomatrix(&img_u8.view(), mask, 1, &ANGLES, true, true)?
            }
        }
    } else {
        graycomatrix(&img_u8.view(), mask, 1, &ANGLES, true, true)?
    };

    // Compute properties and average across angles.
    let (contrast, dissimilarity, homogeneity, asm, energy, correlation) =
        compute_all_properties(&glcm.view());
    let mut props = HashMap::new();
    props.insert("glcm_contrast".to_string(), contrast);
    props.insert("glcm_dissimilarity".to_string(), dissimilarity);
    props.insert("glcm_homogeneity".to_string(), homogeneity);
    props.insert("glcm_ASM".to_string(), asm);
    props.insert("glcm_energy".to_string(), energy);
    props.insert("glcm_correlation".to_string(), correlation);

    Ok(props)
}

fn compute_glcm_gpu(image: &ArrayView2<u8>, mask: &ArrayView2<bool>) -> Result<Array4<f32>> {
    let backend_handle = GpuBackend::get_or_init()?;
    let backend_guard = backend_handle
        .lock()
        .map_err(|_| FeaturizerError::CudaError("GPU backend lock poisoned".to_string()))?;
    let backend = backend_guard
        .as_ref()
        .ok_or_else(|| FeaturizerError::CudaError("GPU backend unavailable".to_string()))?;

    compute_glcm_wgpu(image, mask, true, true, backend)
}

/// Compute Gray-Level Co-occurrence Matrix.
///
/// Port of `skimage.feature.graycomatrix` (texture.py:15-165).
///
/// # Arguments
/// * `image` - u8 image (0-255)
/// * `distance` - Pixel pair distance offset (typically 1)
/// * `angles` - Pixel pair angles in radians
/// * `symmetric` - If true, (i,j) and (j,i) both accumulate
/// * `normed` - If true, normalize each angle's matrix to sum=1
///
/// # Returns
/// 4D array: [levels × levels × 1 × num_angles]
fn graycomatrix(
    image: &ArrayView2<u8>,
    mask: &ArrayView2<bool>,
    distance: usize,
    angles: &[f32],
    symmetric: bool,
    normed: bool,
) -> Result<Array4<f32>> {
    const LEVELS: usize = 256;
    let num_angles = angles.len();
    let (height, width) = image.dim();

    let mut p = Array4::<u32>::zeros((LEVELS, LEVELS, 1, num_angles));

    // Accumulate co-occurrences for each angle
    for (angle_idx, &angle) in angles.iter().enumerate() {
        let (dr, dc) = angle_to_offset(angle, distance);

        for r in 0..height {
            for c in 0..width {
                let new_r = r as i32 + dr;
                let new_c = c as i32 + dc;

                if new_r < 0 || new_r >= height as i32 || new_c < 0 || new_c >= width as i32 {
                    continue;
                }
                if !mask[[r, c]] || !mask[[new_r as usize, new_c as usize]] {
                    continue;
                }

                let i = image[[r, c]] as usize;
                let j = image[[new_r as usize, new_c as usize]] as usize;

                p[[i, j, 0, angle_idx]] += 1;

                if symmetric && i != j {
                    p[[j, i, 0, angle_idx]] += 1;
                }
            }
        }
    }

    // Convert to f32 and normalize if requested
    let mut p_f32 = p.mapv(|x| x as f32);

    for angle_idx in 0..num_angles {
        // Suppress background-origin [0,0] co-occurrence bias.
        p_f32[[0, 0, 0, angle_idx]] = 0.0;

        if normed {
            let sum: f32 = p_f32.slice(s![.., .., 0, angle_idx]).sum();
            if sum > 0.0 {
                p_f32
                    .slice_mut(s![.., .., 0, angle_idx])
                    .mapv_inplace(|x| x / sum);
            }
        }
    }

    Ok(p_f32)
}

/// Convert angle (radians) + distance to (row_offset, col_offset).
fn angle_to_offset(angle: f32, distance: usize) -> (i32, i32) {
    let d = distance as f32;
    let dr = (-d * angle.sin()).round() as i32;
    let dc = (d * angle.cos()).round() as i32;
    (dr, dc)
}

/// Compute all GLCM properties in a cache-friendly pass pattern.
///
/// Returns:
/// (contrast, dissimilarity, homogeneity, asm, energy, correlation)
fn compute_all_properties(p: &ArrayView4<f32>) -> (f64, f64, f64, f64, f64, f64) {
    let (levels, _, _, num_angles) = p.dim();
    let mut contrast_sum = 0.0_f64;
    let mut dissimilarity_sum = 0.0_f64;
    let mut homogeneity_sum = 0.0_f64;
    let mut asm_sum = 0.0_f64;
    let mut correlation_sum = 0.0_f64;

    for a in 0..num_angles {
        let mut contrast = 0.0_f64;
        let mut dissimilarity = 0.0_f64;
        let mut homogeneity = 0.0_f64;
        let mut asm = 0.0_f64;
        let mut mu_i = 0.0_f64;
        let mut mu_j = 0.0_f64;

        for i in 0..levels {
            let i_f = i as f64;
            for j in 0..levels {
                let j_f = j as f64;
                let v = p[[i, j, 0, a]] as f64;
                let diff = i_f - j_f;
                let diff2 = diff * diff;

                contrast += v * diff2;
                dissimilarity += v * diff.abs();
                homogeneity += v / (1.0 + diff2);
                asm += v * v;
                mu_i += i_f * v;
                mu_j += j_f * v;
            }
        }

        let mut var_i = 0.0_f64;
        let mut var_j = 0.0_f64;
        let mut corr_num = 0.0_f64;

        for i in 0..levels {
            let i_f = i as f64;
            for j in 0..levels {
                let j_f = j as f64;
                let v = p[[i, j, 0, a]] as f64;
                let di = i_f - mu_i;
                let dj = j_f - mu_j;
                var_i += v * di * di;
                var_j += v * dj * dj;
                corr_num += v * di * dj;
            }
        }

        let correlation = if var_i * var_j <= 1e-16 {
            // Constant image - correlation undefined
            0.0
        } else {
            corr_num / (var_i * var_j).sqrt()
        };

        contrast_sum += contrast;
        dissimilarity_sum += dissimilarity;
        homogeneity_sum += homogeneity;
        asm_sum += asm;
        correlation_sum += correlation;
    }

    let inv_angles = 1.0 / num_angles as f64;
    let asm_mean = asm_sum * inv_angles;
    let energy = asm_mean.sqrt();

    (
        contrast_sum * inv_angles,
        dissimilarity_sum * inv_angles,
        homogeneity_sum * inv_angles,
        asm_mean,
        energy,
        correlation_sum * inv_angles,
    )
}

fn empty_features() -> HashMap<String, f64> {
    [
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        "glcm_ASM",
    ]
    .iter()
    .map(|&k| (k.to_string(), 0.0))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_empty_mask() {
        let patch = array![[100.0, 150.0], [200.0, 50.0]];
        let mask = array![[false, false], [false, false]];

        let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();
        assert_eq!(features.get("glcm_contrast"), Some(&0.0));
        assert_eq!(features.len(), 6);
    }

    #[test]
    fn test_uniform_texture() {
        let patch = array![
            [128.0, 128.0, 128.0],
            [128.0, 128.0, 128.0],
            [128.0, 128.0, 128.0]
        ];
        let mask = array![[true, true, true], [true, true, true], [true, true, true]];

        let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

        // Uniform texture: returns zeros (only one unique value)
        let contrast = features.get("glcm_contrast").unwrap();
        assert_eq!(*contrast, 0.0, "Uniform texture should have zero contrast");
    }

    #[test]
    fn test_basic_glcm() {
        // Simple 4x4 with clear gradient
        let patch = array![
            [0.0, 50.0, 100.0, 150.0],
            [50.0, 100.0, 150.0, 200.0],
            [100.0, 150.0, 200.0, 255.0],
            [150.0, 200.0, 255.0, 255.0]
        ];
        let mask = array![
            [true, true, true, true],
            [true, true, true, true],
            [true, true, true, true],
            [true, true, true, true]
        ];

        let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

        // Should have non-zero contrast (many distinct values)
        let contrast = features.get("glcm_contrast").unwrap();
        assert!(*contrast > 0.0, "Gradient should have non-zero contrast");

        // All features should be finite
        for (key, &val) in features.iter() {
            assert!(val.is_finite(), "Feature {} should be finite", key);
        }
    }

    #[test]
    fn test_glcm_gpu_flag_small_image_uses_cpu_path() {
        let patch = array![
            [0.0, 10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0, 40.0],
            [20.0, 30.0, 40.0, 50.0],
            [30.0, 40.0, 50.0, 60.0]
        ];
        let mask = array![
            [true, true, true, true],
            [true, true, true, true],
            [true, true, true, true],
            [true, true, true, true]
        ];

        let cpu = calculate_glcm_features_with_gpu(&patch.view(), &mask.view(), false).unwrap();
        let gpu_flag = calculate_glcm_features_with_gpu(&patch.view(), &mask.view(), true).unwrap();
        assert_eq!(cpu, gpu_flag);
    }
}
