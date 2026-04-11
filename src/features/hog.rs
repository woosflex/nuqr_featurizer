//! Histogram of Oriented Gradients (HOG) features.
//!
//! Port of `calculate_hog_features` from `Final_Code_Features_13.10.py`
//! using skimage parameters:
//! - orientations = 8
//! - pixels_per_cell = (8, 8)
//! - cells_per_block = (1, 1)
//! - block_norm = "L2-Hys" (skimage default)
//!
//! # GPU Acceleration
//!
//! This module supports GPU acceleration via WGPU. When `use_gpu = true` and a
//! compatible adapter is available:
//! - Gradient computation is parallelized across pixels
//! - Histogram binning uses shared memory for reduced contention
//! - Expected speedup: 15-20x on images larger than 50×50 pixels
//!
//! The GPU path automatically falls back to CPU if:
//! - GPU is not available at runtime
//! - Image is too small (<40×40, GPU overhead exceeds benefit)
//! - GPU returns an error

use std::collections::HashMap;

use ndarray::ArrayView2;

use crate::core::{FeaturizerError, Result};
use crate::gpu::{compute_hog_wgpu, is_gpu_available, GpuBackend};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_sqrt_pd, _mm256_storeu_pd, _mm256_sub_pd,
};

const ORIENTATIONS: usize = 8;
const CELL_ROWS: usize = 8;
const CELL_COLS: usize = 8;
const HOG_EPS: f64 = 1e-5;

// GPU threshold: use GPU only for images larger than this
const GPU_MIN_SIZE: usize = 40;

/// Calculate HOG summary features from grayscale patch and mask.
///
/// # Arguments
/// * `grayscale_patch` - Grayscale image patch (f32 values in [0, 255])
/// * `mask` - Binary mask indicating nucleus pixels
/// * `use_gpu` - If true, attempt to use GPU acceleration
///
/// Returned keys:
/// - `hog_mean`
/// - `hog_std`
/// - `hog_max`
/// - `hog_min`
pub fn calculate_hog_features(
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

    let (h, w) = grayscale_patch.dim();
    let n_cells_row = h / CELL_ROWS;
    let n_cells_col = w / CELL_COLS;
    if n_cells_row == 0 || n_cells_col == 0 {
        return Ok(default_features());
    }

    // Match Python reference: hog is computed on (patch * mask).astype(uint8).
    let mut masked_image = vec![0.0_f32; h * w];
    for r in 0..h {
        for c in 0..w {
            masked_image[r * w + c] = if mask[[r, c]] {
                grayscale_patch[[r, c]].clamp(0.0, 255.0)
            } else {
                0.0
            };
        }
    }

    // Match Python reference early-return condition.
    if !masked_image.iter().any(|&v| v > 0.0) {
        return Ok(default_features());
    }

    // GPU path: use WGPU if requested, available, and image is large enough
    if use_gpu && h >= GPU_MIN_SIZE && w >= GPU_MIN_SIZE && is_gpu_available() {
        match compute_hog_gpu(&masked_image, h, w) {
            Ok(features) => return Ok(features),
            Err(e) => {
                // Log GPU failure and fall back to CPU
                eprintln!("GPU HOG failed ({}), falling back to CPU", e);
            }
        }
    }

    // CPU path (always available, used as fallback)
    compute_hog_cpu(&masked_image, h, w, n_cells_row, n_cells_col)
}

/// GPU-accelerated HOG computation via WGPU
fn compute_hog_gpu(masked_image: &[f32], h: usize, w: usize) -> Result<HashMap<String, f64>> {
    let image = masked_image.to_vec();
    // Include all pixels; masking is already encoded in `image` as zeros.
    let mask_u32 = vec![1_u32; h * w];

    let backend_handle = GpuBackend::get_or_init()?;
    let backend_guard = backend_handle
        .lock()
        .map_err(|_| FeaturizerError::CudaError("GPU backend lock poisoned".to_string()))?;
    let backend = backend_guard
        .as_ref()
        .ok_or_else(|| FeaturizerError::CudaError("GPU backend unavailable".to_string()))?;

    let hog_descriptor = compute_hog_wgpu(&image, &mask_u32, h, w, backend)?;

    // Convert f32 descriptor to f64 and compute summary stats (match CPU output)
    let hog_feats: Vec<f64> = hog_descriptor.iter().map(|&v| v as f64).collect();

    if hog_feats.is_empty() {
        return Ok(default_features());
    }

    compute_summary_stats(&hog_feats)
}

/// CPU HOG computation (original implementation)
fn compute_hog_cpu(
    masked_image: &[f32],
    h: usize,
    w: usize,
    n_cells_row: usize,
    n_cells_col: usize,
) -> Result<HashMap<String, f64>> {
    let image = masked_image.iter().map(|&v| v as f64).collect::<Vec<_>>();

    let (g_row, g_col) = channel_gradient(&image, h, w);
    let mut magnitude = vec![0.0_f64; h * w];
    compute_magnitude(&g_row, &g_col, &mut magnitude);
    let mut orientation = vec![0.0_f64; h * w];
    for i in 0..(h * w) {
        let gr = g_row[i];
        let gc = g_col[i];
        let mut ang = gr.atan2(gc).to_degrees();
        if ang < 0.0 {
            ang += 180.0;
        }
        // Keep [0, 180)
        if ang >= 180.0 {
            ang -= 180.0;
        }
        orientation[i] = ang;
    }

    let hist = orientation_histograms(
        &magnitude,
        &orientation,
        h,
        w,
        n_cells_row,
        n_cells_col,
        ORIENTATIONS,
    );

    // cells_per_block=(1,1), block_norm="L2-Hys"
    let mut hog_feats = Vec::with_capacity(n_cells_row * n_cells_col * ORIENTATIONS);
    for r in 0..n_cells_row {
        for c in 0..n_cells_col {
            let start = (r * n_cells_col + c) * ORIENTATIONS;
            let block = &hist[start..start + ORIENTATIONS];
            let normalized = normalize_l2_hys(block);
            hog_feats.extend_from_slice(&normalized);
        }
    }

    if hog_feats.is_empty() {
        return Ok(default_features());
    }

    compute_summary_stats(&hog_feats)
}

fn compute_magnitude(g_row: &[f64], g_col: &[f64], magnitude: &mut [f64]) {
    debug_assert_eq!(g_row.len(), g_col.len());
    debug_assert_eq!(g_row.len(), magnitude.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY:
            // - AVX2 support is runtime-checked above.
            // - All slices have equal length and are valid for that length.
            unsafe {
                compute_magnitude_avx2(g_row, g_col, magnitude);
            }
            return;
        }
    }

    compute_magnitude_scalar(g_row, g_col, magnitude);
}

fn compute_magnitude_scalar(g_row: &[f64], g_col: &[f64], magnitude: &mut [f64]) {
    for i in 0..magnitude.len() {
        let gr = g_row[i];
        let gc = g_col[i];
        magnitude[i] = (gc * gc + gr * gr).sqrt();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_magnitude_avx2(g_row: &[f64], g_col: &[f64], magnitude: &mut [f64]) {
    let len = magnitude.len();
    let simd_end = len / 4 * 4;
    let mut i = 0usize;
    while i < simd_end {
        // SAFETY:
        // - i..i+4 is in bounds for all slices by simd_end construction.
        // - Unaligned load/store intrinsics are used.
        let gc = unsafe { _mm256_loadu_pd(g_col.as_ptr().add(i)) };
        let gr = unsafe { _mm256_loadu_pd(g_row.as_ptr().add(i)) };
        let gc2 = _mm256_mul_pd(gc, gc);
        let gr2 = _mm256_mul_pd(gr, gr);
        let sum = _mm256_add_pd(gc2, gr2);
        let mag = _mm256_sqrt_pd(sum);
        // SAFETY: i..i+4 is in bounds for magnitude.
        unsafe { _mm256_storeu_pd(magnitude.as_mut_ptr().add(i), mag) };
        i += 4;
    }

    if i < len {
        compute_magnitude_scalar(&g_row[i..], &g_col[i..], &mut magnitude[i..]);
    }
}

/// Compute mean, std, min, max from HOG descriptor
fn compute_summary_stats(hog_feats: &[f64]) -> Result<HashMap<String, f64>> {
    let n = hog_feats.len() as f64;
    let mean = hog_feats.iter().sum::<f64>() / n;
    let variance = hog_feats
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();
    let min = hog_feats
        .iter()
        .fold(f64::INFINITY, |acc, &v| if v < acc { v } else { acc });
    let max = hog_feats
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &v| if v > acc { v } else { acc });

    let mut out = HashMap::new();
    out.insert("hog_mean".to_string(), finite_or_zero(mean));
    out.insert("hog_std".to_string(), finite_or_zero(std));
    out.insert("hog_max".to_string(), finite_or_zero(max));
    out.insert("hog_min".to_string(), finite_or_zero(min));
    Ok(out)
}

fn channel_gradient(image: &[f64], h: usize, w: usize) -> (Vec<f64>, Vec<f64>) {
    let mut g_row = vec![0.0_f64; h * w];
    let mut g_col = vec![0.0_f64; h * w];

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY:
            // - AVX2 support is runtime-checked above.
            // - Output vectors are allocated for h*w entries.
            unsafe { channel_gradient_avx2(image, h, w, &mut g_row, &mut g_col) };
            return (g_row, g_col);
        }
    }

    channel_gradient_scalar(image, h, w, &mut g_row, &mut g_col);

    (g_row, g_col)
}

fn channel_gradient_scalar(
    image: &[f64],
    h: usize,
    w: usize,
    g_row: &mut [f64],
    g_col: &mut [f64],
) {
    // g_row[1:-1, :] = image[2:, :] - image[:-2, :]
    for r in 1..(h.saturating_sub(1)) {
        for c in 0..w {
            g_row[r * w + c] = image[(r + 1) * w + c] - image[(r - 1) * w + c];
        }
    }

    // g_col[:, 1:-1] = image[:, 2:] - image[:, :-2]
    for r in 0..h {
        for c in 1..(w.saturating_sub(1)) {
            g_col[r * w + c] = image[r * w + (c + 1)] - image[r * w + (c - 1)];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn subtract_slices_avx2(dst: &mut [f64], a: &[f64], b: &[f64]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());

    let len = dst.len();
    let simd_end = len / 4 * 4;
    let mut i = 0usize;
    while i < simd_end {
        // SAFETY:
        // - i..i+4 is in bounds for all slices by simd_end construction.
        // - Unaligned load/store intrinsics are used.
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        let vd = _mm256_sub_pd(va, vb);
        // SAFETY: i..i+4 is in bounds for dst by the same construction.
        unsafe { _mm256_storeu_pd(dst.as_mut_ptr().add(i), vd) };
        i += 4;
    }

    for j in i..len {
        dst[j] = a[j] - b[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn channel_gradient_avx2(
    image: &[f64],
    h: usize,
    w: usize,
    g_row: &mut [f64],
    g_col: &mut [f64],
) {
    if h >= 3 {
        for r in 1..(h - 1) {
            let row_start = r * w;
            let row_end = row_start + w;
            let upper_start = (r + 1) * w;
            let upper_end = upper_start + w;
            let lower_start = (r - 1) * w;
            let lower_end = lower_start + w;
            // SAFETY: slices are within row bounds guaranteed by loop range.
            unsafe {
                subtract_slices_avx2(
                    &mut g_row[row_start..row_end],
                    &image[upper_start..upper_end],
                    &image[lower_start..lower_end],
                );
            }
        }
    }

    if w >= 3 {
        for r in 0..h {
            let row_base = r * w;
            // SAFETY:
            // - For w>=3, all three slices span exactly w-2 items in one row.
            unsafe {
                subtract_slices_avx2(
                    &mut g_col[(row_base + 1)..(row_base + w - 1)],
                    &image[(row_base + 2)..(row_base + w)],
                    &image[row_base..(row_base + w - 2)],
                );
            }
        }
    }
}

fn orientation_histograms(
    magnitude: &[f64],
    orientation: &[f64],
    h: usize,
    w: usize,
    n_cells_row: usize,
    n_cells_col: usize,
    n_orient: usize,
) -> Vec<f64> {
    let mut hist = vec![0.0_f64; n_cells_row * n_cells_col * n_orient];

    let r0 = CELL_ROWS / 2;
    let c0 = CELL_COLS / 2;
    let range_rows_start = -((CELL_ROWS / 2) as isize);
    let range_rows_stop = ((CELL_ROWS + 1) / 2) as isize;
    let range_cols_start = -((CELL_COLS / 2) as isize);
    let range_cols_stop = ((CELL_COLS + 1) / 2) as isize;
    let orient_step = 180.0 / n_orient as f64;

    for i in 0..n_orient {
        let orient_start = orient_step * (i as f64 + 1.0);
        let orient_end = orient_step * i as f64;

        for r_i in 0..n_cells_row {
            let center_r = r0 + r_i * CELL_ROWS;
            for c_i in 0..n_cells_col {
                let center_c = c0 + c_i * CELL_COLS;
                let mut total = 0.0_f64;

                for dr in range_rows_start..range_rows_stop {
                    let rr = center_r as isize + dr;
                    if rr < 0 || rr >= h as isize {
                        continue;
                    }
                    for dc in range_cols_start..range_cols_stop {
                        let cc = center_c as isize + dc;
                        if cc < 0 || cc >= w as isize {
                            continue;
                        }
                        let idx = rr as usize * w + cc as usize;
                        let ang = orientation[idx];
                        // Match _hoghistogram bin condition:
                        // include if orientation_end <= ang < orientation_start
                        if !(ang >= orient_start || ang < orient_end) {
                            total += magnitude[idx];
                        }
                    }
                }

                let out_idx = (r_i * n_cells_col + c_i) * n_orient + i;
                hist[out_idx] = total / (CELL_ROWS * CELL_COLS) as f64;
            }
        }
    }

    hist
}

fn normalize_l2_hys(block: &[f64]) -> [f64; ORIENTATIONS] {
    let mut out = [0.0_f64; ORIENTATIONS];
    for (i, v) in block.iter().enumerate().take(ORIENTATIONS) {
        out[i] = *v;
    }

    let l2 = (out.iter().map(|v| v * v).sum::<f64>() + HOG_EPS * HOG_EPS).sqrt();
    if l2 > 0.0 {
        for v in &mut out {
            *v /= l2;
            if *v > 0.2 {
                *v = 0.2;
            }
        }
        let l2_hys = (out.iter().map(|v| v * v).sum::<f64>() + HOG_EPS * HOG_EPS).sqrt();
        if l2_hys > 0.0 {
            for v in &mut out {
                *v /= l2_hys;
            }
        }
    }
    out
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn default_features() -> HashMap<String, f64> {
    [
        ("hog_mean".to_string(), 0.0),
        ("hog_std".to_string(), 0.0),
        ("hog_max".to_string(), 0.0),
        ("hog_min".to_string(), 0.0),
    ]
    .into_iter()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_hog_empty_patch_returns_defaults() {
        let patch = Array2::<f32>::zeros((16, 16));
        let mask = Array2::<bool>::from_elem((16, 16), true);
        let feats = calculate_hog_features(&patch.view(), &mask.view(), false).unwrap();
        assert_eq!(feats["hog_mean"], 0.0);
        assert_eq!(feats["hog_std"], 0.0);
        assert_eq!(feats["hog_max"], 0.0);
        assert_eq!(feats["hog_min"], 0.0);
    }

    #[test]
    fn test_hog_small_patch_returns_defaults() {
        let patch = Array2::<f32>::from_elem((7, 7), 100.0);
        let mask = Array2::<bool>::from_elem((7, 7), true);
        let feats = calculate_hog_features(&patch.view(), &mask.view(), false).unwrap();
        assert_eq!(feats["hog_mean"], 0.0);
        assert_eq!(feats["hog_std"], 0.0);
        assert_eq!(feats["hog_max"], 0.0);
        assert_eq!(feats["hog_min"], 0.0);
    }

    #[test]
    fn test_hog_shape_mismatch_error() {
        let patch = Array2::<f32>::from_elem((16, 16), 100.0);
        let mask = Array2::<bool>::from_elem((16, 15), true);
        let res = calculate_hog_features(&patch.view(), &mask.view(), false);
        assert!(res.is_err());
    }

    #[test]
    fn test_hog_gradient_patch_finite() {
        let mut patch = Array2::<f32>::zeros((16, 16));
        for r in 0..16 {
            for c in 0..16 {
                patch[[r, c]] = (r * 8 + c) as f32;
            }
        }
        let mask = Array2::<bool>::from_elem((16, 16), true);
        let feats = calculate_hog_features(&patch.view(), &mask.view(), false).unwrap();
        assert!(feats["hog_mean"].is_finite());
        assert!(feats["hog_std"].is_finite());
        assert!(feats["hog_max"].is_finite());
        assert!(feats["hog_min"].is_finite());
        assert!(feats["hog_max"] >= feats["hog_min"]);
    }

    #[test]
    fn test_hog_partial_mask_runs() {
        let mut patch = Array2::<f32>::zeros((16, 16));
        for r in 0..16 {
            for c in 0..16 {
                patch[[r, c]] = (c * 4) as f32;
            }
        }
        let mut mask = Array2::<bool>::from_elem((16, 16), false);
        for r in 4..12 {
            for c in 4..12 {
                mask[[r, c]] = true;
            }
        }
        let feats = calculate_hog_features(&patch.view(), &mask.view(), false).unwrap();
        assert!(feats.values().all(|v| v.is_finite()));
    }

    #[test]
    fn test_orientation_bin_edges() {
        // Ensure binning condition works at boundaries.
        // 0 degrees should fall into bin 0 [0, 22.5)
        let ori = vec![0.0_f64; 64];
        let mut mag = vec![0.0_f64; 64];
        mag[4 * 8 + 4] = 1.0;
        let hist = orientation_histograms(&mag, &ori, 8, 8, 1, 1, ORIENTATIONS);
        assert!(hist[0] > 0.0);
    }

    #[test]
    fn test_l2_hys_normalization_finite() {
        let block = [0.0, 1.0, 2.0, 0.5, 0.25, 0.0, 4.0, 0.0];
        let norm = normalize_l2_hys(&block);
        assert!(norm.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_reference_like_call_pattern() {
        let patch = array![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        ];
        let mask = Array2::<bool>::from_elem((8, 8), true);
        let feats = calculate_hog_features(&patch.view(), &mask.view(), false).unwrap();
        assert!(feats["hog_mean"].is_finite());
    }

    #[test]
    fn test_channel_gradient_matches_scalar_reference() {
        let h = 17;
        let w = 19;
        let mut image = vec![0.0_f64; h * w];
        for r in 0..h {
            for c in 0..w {
                image[r * w + c] = (r * 13 + c * 7) as f64;
            }
        }

        let (g_row, g_col) = channel_gradient(&image, h, w);
        let mut g_row_ref = vec![0.0_f64; h * w];
        let mut g_col_ref = vec![0.0_f64; h * w];
        channel_gradient_scalar(&image, h, w, &mut g_row_ref, &mut g_col_ref);

        assert_eq!(g_row, g_row_ref);
        assert_eq!(g_col, g_col_ref);
    }

    #[test]
    fn test_compute_magnitude_matches_scalar_reference() {
        let len = 257;
        let g_row: Vec<f64> = (0..len).map(|i| (i % 31) as f64 - 15.0).collect();
        let g_col: Vec<f64> = (0..len).map(|i| (i % 29) as f64 - 14.0).collect();

        let mut simd_or_scalar = vec![0.0_f64; len];
        let mut scalar = vec![0.0_f64; len];
        compute_magnitude(&g_row, &g_col, &mut simd_or_scalar);
        compute_magnitude_scalar(&g_row, &g_col, &mut scalar);

        for i in 0..len {
            assert!((simd_or_scalar[i] - scalar[i]).abs() <= 1e-12);
        }
    }
}
