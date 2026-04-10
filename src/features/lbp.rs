//! Local Binary Pattern (LBP) texture features.
//!
//! Port of `skimage.feature.local_binary_pattern` (Cython implementation in _texture.pyx).
//!
//! # Algorithm
//!
//! For each pixel (r, c):
//! 1. Sample P neighbors at radius R using bilinear interpolation
//! 2. Threshold neighbors against center pixel (1 if >= center, else 0)
//! 3. For "uniform" method: compute pattern code (0-9 for P=8)
//!    - Uniform = at most 2 transitions (0→1 or 1→0)
//!    - Non-uniform patterns get code P+1 (=9 for P=8)
//! 4. Extract features from LBP histogram: mean, std, entropy
//!
//! # References
//! - scikit-image: `src/skimage/feature/_texture.pyx` lines 87-270

use crate::core::error::{FeaturizerError, Result};
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Calculate LBP (Local Binary Pattern) features from grayscale patch.
///
/// Port of Python reference (Final_Code_Features_13.10.py:270-282):
/// ```python
/// lbp_image = feature.local_binary_pattern(img_for_lbp, P=8, R=1, method='uniform')
/// lbp_values_in_nucleus = lbp_image[nucleus_mask_cropped > 0]
/// n_bins = int(lbp_image.max() + 1)
/// hist_lbp, _ = np.histogram(lbp_values_in_nucleus, bins=n_bins, ...)
/// features = {'lbp_mean': ..., 'lbp_std': ..., 'lbp_entropy': ...}
/// ```
///
/// # Arguments
/// * `grayscale_patch` - f32 image (0.0-255.0)
/// * `mask` - bool array, true = nucleus pixels
///
/// # Returns
/// HashMap with 3 features:
/// - lbp_mean: mean of LBP codes in masked region
/// - lbp_std: standard deviation of LBP codes
/// - lbp_entropy: Shannon entropy of LBP histogram
pub fn calculate_lbp_features(
    grayscale_patch: &ArrayView2<f32>,
    mask: &ArrayView2<bool>,
) -> Result<HashMap<String, f64>> {
    if grayscale_patch.shape() != mask.shape() {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("{:?}", mask.shape()),
            got: format!("{:?}", grayscale_patch.shape()),
        });
    }

    // Use full grayscale patch for interpolation and apply mask only when
    // selecting output codes.
    let mut img_for_lbp = Array2::<f32>::zeros(grayscale_patch.dim());
    for ((r, c), &val) in grayscale_patch.indexed_iter() {
        img_for_lbp[[r, c]] = val.clamp(0.0, 255.0);
    }

    // Edge case: empty mask
    if !mask.iter().any(|&v| v) {
        return Ok(empty_features());
    }

    // Compute LBP image with P=8, R=1, method='uniform'
    const P: usize = 8;
    const R: f32 = 1.0;
    let lbp_image = local_binary_pattern_uniform(&img_for_lbp.view(), P, R);

    // Extract values in masked region
    let mut lbp_values = Vec::new();
    for ((r, c), &lbp_val) in lbp_image.indexed_iter() {
        if mask[[r, c]] {
            lbp_values.push(lbp_val);
        }
    }

    if lbp_values.is_empty() {
        return Ok(empty_features());
    }

    // Match Python: n_bins = int(lbp_image.max() + 1)
    let max_code = lbp_image.iter().copied().max().unwrap_or(0);
    let n_bins = max_code + 1;
    let mut histogram = vec![0usize; n_bins];

    for &val in &lbp_values {
        if val < n_bins {
            histogram[val] += 1;
        }
    }

    // Normalize to density
    let total = lbp_values.len() as f64;
    let hist_density: Vec<f64> = histogram
        .iter()
        .map(|&count| count as f64 / total)
        .collect();

    // Compute entropy: -Σ p*log2(p) for p > 0
    let entropy: f64 = hist_density
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum();

    // Compute mean and std
    let mean: f64 = lbp_values.iter().map(|&v| v as f64).sum::<f64>() / total;
    let variance: f64 = lbp_values
        .iter()
        .map(|&v| {
            let diff = v as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / total;
    let std = variance.sqrt();

    let mut features = HashMap::new();
    features.insert("lbp_mean".to_string(), mean);
    features.insert("lbp_std".to_string(), std);
    features.insert("lbp_entropy".to_string(), entropy);

    Ok(features)
}

/// Compute LBP image with uniform pattern encoding.
///
/// Port of `_local_binary_pattern` with method='U' (texture.pyx:87-270).
///
/// # Arguments
/// * `image` - Grayscale image (f32)
/// * `p` - Number of circularly symmetric neighbors (typically 8)
/// * `r` - Radius of circle (typically 1.0)
///
/// # Returns
/// Array of LBP codes (usize values 0 to P+1 for method='uniform')
fn local_binary_pattern_uniform(image: &ArrayView2<f32>, p: usize, r: f32) -> Array2<usize> {
    let (rows, cols) = image.dim();
    let mut output = Array2::<usize>::zeros((rows, cols));

    // Pre-compute neighbor positions (polar coordinates)
    let mut rp = Vec::with_capacity(p);
    let mut cp = Vec::with_capacity(p);
    for i in 0..p {
        let angle = 2.0 * PI * (i as f32) / (p as f32);
        let rr = -r * angle.sin();
        let cc = r * angle.cos();
        // Match scikit-image's rounding to 5 decimals.
        rp.push((rr * 100_000.0).round() / 100_000.0);
        cp.push((cc * 100_000.0).round() / 100_000.0);
    }

    // Process each pixel
    for r in 0..rows {
        for c in 0..cols {
            let center = image[[r, c]];

            // Sample P neighbors with bilinear interpolation
            let mut signed_texture = Vec::with_capacity(p);
            for i in 0..p {
                let nr = r as f32 + rp[i];
                let nc = c as f32 + cp[i];
                let neighbor = bilinear_interpolate(image, nr, nc);
                signed_texture.push(if neighbor >= center { 1u8 } else { 0u8 });
            }

            // Compute LBP code for uniform patterns
            let lbp_code = compute_uniform_lbp(&signed_texture, p);
            output[[r, c]] = lbp_code;
        }
    }

    output
}

/// Compute uniform LBP code from binary pattern.
///
/// Port of texture.pyx:177-251 (method='U').
///
/// # Algorithm
/// Uniform patterns have ≤ 2 transitions (0→1 or 1→0).
/// For P=8:
/// - Uniform patterns: 0 to 8 (encoded by number of 1s)
/// - Non-uniform patterns: 9 (catch-all)
fn compute_uniform_lbp(signed_texture: &[u8], p: usize) -> usize {
    // Match scikit-image method='U':
    // changes = sum(signed_texture[i] != signed_texture[i + 1] for i in 0..P-2)
    // (no wrap-around transition term)
    let mut changes = 0;
    for i in 0..p - 1 {
        if signed_texture[i] != signed_texture[i + 1] {
            changes += 1;
        }
    }

    if changes <= 2 {
        // Uniform pattern: code is number of ones.
        let n_ones: usize = signed_texture.iter().filter(|&&b| b == 1).count();
        n_ones
    } else {
        // Non-uniform pattern: catch-all bin.
        p + 1
    }
}

/// Bilinear interpolation at sub-pixel location.
///
/// Port of `bilinear_interpolation` from scikit-image (interpolation.pyx).
///
/// # Arguments
/// * `image` - Input image
/// * `r` - Row coordinate (can be fractional)
/// * `c` - Column coordinate (can be fractional)
///
/// # Returns
/// Interpolated value using clamped boundary coordinates.
fn bilinear_interpolate(image: &ArrayView2<f32>, r: f32, c: f32) -> f32 {
    let r0 = r.floor() as isize;
    let c0 = c.floor() as isize;
    let r1 = r0 + 1;
    let c1 = c0 + 1;

    let dr = r - (r0 as f32);
    let dc = c - (c0 as f32);

    let v00 = sample_clamped(image, r0, c0);
    let v01 = sample_clamped(image, r0, c1);
    let v10 = sample_clamped(image, r1, c0);
    let v11 = sample_clamped(image, r1, c1);

    let v0 = v00 * (1.0 - dc) + v01 * dc;
    let v1 = v10 * (1.0 - dc) + v11 * dc;
    v0 * (1.0 - dr) + v1 * dr
}

fn sample_clamped(image: &ArrayView2<f32>, r: isize, c: isize) -> f32 {
    let (rows, cols) = image.dim();
    let rr = r.clamp(0, rows as isize - 1);
    let cc = c.clamp(0, cols as isize - 1);
    image[[rr as usize, cc as usize]]
}

/// Return empty feature dict (all zeros).
fn empty_features() -> HashMap<String, f64> {
    let mut features = HashMap::new();
    features.insert("lbp_mean".to_string(), 0.0);
    features.insert("lbp_std".to_string(), 0.0);
    features.insert("lbp_entropy".to_string(), 0.0);
    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lbp_empty_mask() {
        let image = array![[100.0, 150.0], [200.0, 250.0]];
        let mask = array![[false, false], [false, false]];

        let features = calculate_lbp_features(&image.view(), &mask.view()).unwrap();

        assert_eq!(features["lbp_mean"], 0.0);
        assert_eq!(features["lbp_std"], 0.0);
        assert_eq!(features["lbp_entropy"], 0.0);
    }

    #[test]
    fn test_lbp_uniform_pattern() {
        // Create a smooth gradient (should produce low entropy, mostly uniform patterns)
        let image = array![
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [105.0, 106.0, 107.0, 108.0, 109.0],
            [110.0, 111.0, 112.0, 113.0, 114.0],
            [115.0, 116.0, 117.0, 118.0, 119.0],
            [120.0, 121.0, 122.0, 123.0, 124.0],
        ];
        let mask = Array2::from_elem(image.dim(), true);

        let features = calculate_lbp_features(&image.view(), &mask.view()).unwrap();

        // Smooth gradient should have low entropy (repetitive patterns)
        assert!(
            features["lbp_entropy"] < 5.0,
            "Entropy should be low for smooth gradient"
        );
        assert!(features["lbp_mean"].is_finite());
        assert!(features["lbp_std"].is_finite());
    }

    #[test]
    fn test_bilinear_interpolate_exact() {
        let image = array![[1.0, 2.0], [3.0, 4.0]];

        // Exact pixel values
        assert_eq!(bilinear_interpolate(&image.view(), 0.0, 0.0), 1.0);
        assert_eq!(bilinear_interpolate(&image.view(), 0.0, 1.0), 2.0);

        // Midpoint
        let mid = bilinear_interpolate(&image.view(), 0.5, 0.5);
        assert!(
            (mid - 2.5).abs() < 1e-5,
            "Midpoint should be 2.5, got {}",
            mid
        );
    }

    #[test]
    fn test_uniform_pattern_detection() {
        // Pattern: 00000000 → uniform (all 0s)
        let pattern = vec![0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(compute_uniform_lbp(&pattern, 8), 0);

        // Pattern: 11111111 → uniform (all 1s)
        let pattern = vec![1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(compute_uniform_lbp(&pattern, 8), 8);

        // Pattern: 00001111 → uniform (2 transitions)
        let pattern = vec![0, 0, 0, 0, 1, 1, 1, 1];
        assert_eq!(compute_uniform_lbp(&pattern, 8), 4);

        // Pattern: 01010101 → non-uniform (8 transitions)
        let pattern = vec![0, 1, 0, 1, 0, 1, 0, 1];
        assert_eq!(compute_uniform_lbp(&pattern, 8), 9); // catch-all for method='uniform'
    }
}
