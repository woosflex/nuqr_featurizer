//! Contrast Limited Adaptive Histogram Equalization (CLAHE) for CCSM.
//!
//! This is a 2D grayscale CLAHE implementation aligned with the skimage usage
//! in the Python reference:
//! `equalize_adapthist(img_uint8, clip_limit=0.03, kernel_size=16)`.

use ndarray::{Array2, ArrayView2};

use crate::core::{FeaturizerError, Result};
use crate::gpu::{compute_clahe_wgpu, is_gpu_available, GpuBackend};

const N_BINS: usize = 256;
const GPU_MIN_SIZE: usize = 40;
const GPU_MAX_PIXELS: usize = 128 * 128;

/// Apply CLAHE to a u8 grayscale image.
///
/// `clip_limit` is normalized (0..1], consistent with skimage.
/// `kernel_size` controls contextual tile size.
pub fn clahe_u8(image: &ArrayView2<u8>, clip_limit: f32, kernel_size: usize) -> Result<Array2<u8>> {
    clahe_u8_with_gpu(image, clip_limit, kernel_size, false)
}

/// Apply CLAHE with optional WGPU acceleration.
pub fn clahe_u8_with_gpu(
    image: &ArrayView2<u8>,
    clip_limit: f32,
    kernel_size: usize,
    use_gpu: bool,
) -> Result<Array2<u8>> {
    let (h, w) = image.dim();
    let n_pixels = h * w;

    if use_gpu
        && h >= GPU_MIN_SIZE
        && w >= GPU_MIN_SIZE
        && n_pixels <= GPU_MAX_PIXELS
        && is_gpu_available()
    {
        match compute_clahe_gpu(image, clip_limit, kernel_size) {
            Ok(out) => return Ok(out),
            Err(e) => {
                eprintln!("GPU CLAHE failed ({}), falling back to CPU", e);
            }
        }
    }

    clahe_u8_cpu(image, clip_limit, kernel_size)
}

/// Apply CLAHE to a batch of images, reusing a single GPU backend when enabled.
pub fn clahe_u8_batch_with_gpu(
    images: &[Array2<u8>],
    clip_limit: f32,
    kernel_size: usize,
    use_gpu: bool,
) -> Result<Vec<Array2<u8>>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }

    if !use_gpu || !is_gpu_available() {
        return images
            .iter()
            .map(|img| clahe_u8_cpu(&img.view(), clip_limit, kernel_size))
            .collect();
    }

    let backend_handle = GpuBackend::get_or_init()?;
    let backend_guard = backend_handle
        .lock()
        .map_err(|_| FeaturizerError::CudaError("GPU backend lock poisoned".to_string()))?;
    let Some(backend) = backend_guard.as_ref() else {
        return images
            .iter()
            .map(|img| clahe_u8_cpu(&img.view(), clip_limit, kernel_size))
            .collect();
    };

    let mut outputs = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        let (h, w) = image.dim();
        let n_pixels = h * w;

        if h >= GPU_MIN_SIZE && w >= GPU_MIN_SIZE && n_pixels <= GPU_MAX_PIXELS {
            match compute_clahe_wgpu(&image.view(), clip_limit, kernel_size, backend) {
                Ok(out) => {
                    outputs.push(out);
                    continue;
                }
                Err(e) => {
                    eprintln!(
                        "GPU CLAHE failed for batch item {} ({}), falling back to CPU",
                        idx, e
                    );
                }
            }
        }

        outputs.push(clahe_u8_cpu(&image.view(), clip_limit, kernel_size)?);
    }

    Ok(outputs)
}

fn clahe_u8_cpu(image: &ArrayView2<u8>, clip_limit: f32, kernel_size: usize) -> Result<Array2<u8>> {
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

    let tile_h = kernel_size;
    let tile_w = kernel_size;
    let n_tiles_y = h.div_ceil(tile_h);
    let n_tiles_x = w.div_ceil(tile_w);

    let mut luts = vec![[0_u8; N_BINS]; n_tiles_y * n_tiles_x];

    for ty in 0..n_tiles_y {
        let y0 = ty * tile_h;
        let y1 = ((ty + 1) * tile_h).min(h);
        for tx in 0..n_tiles_x {
            let x0 = tx * tile_w;
            let x1 = ((tx + 1) * tile_w).min(w);

            let tile_pixels = (y1 - y0) * (x1 - x0);
            let clip = if clip_limit > 0.0 {
                ((clip_limit * tile_pixels as f32).max(1.0)) as u32
            } else {
                tile_pixels as u32
            };

            let mut hist = [0_u32; N_BINS];
            for y in y0..y1 {
                for x in x0..x1 {
                    hist[image[[y, x]] as usize] += 1;
                }
            }

            clip_histogram(&mut hist, clip);
            luts[ty * n_tiles_x + tx] = histogram_to_lut(&hist, tile_pixels as u32);
        }
    }

    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        let (y0, y1, wy) = interpolation_indices(y, n_tiles_y, tile_h);
        for x in 0..w {
            let (x0, x1, wx) = interpolation_indices(x, n_tiles_x, tile_w);
            let p = image[[y, x]] as usize;

            let v00 = luts[y0 * n_tiles_x + x0][p] as f64;
            let v01 = luts[y0 * n_tiles_x + x1][p] as f64;
            let v10 = luts[y1 * n_tiles_x + x0][p] as f64;
            let v11 = luts[y1 * n_tiles_x + x1][p] as f64;

            let top = (1.0 - wx) * v00 + wx * v01;
            let bottom = (1.0 - wx) * v10 + wx * v11;
            let value = ((1.0 - wy) * top + wy * bottom).round().clamp(0.0, 255.0);
            out[[y, x]] = value as u8;
        }
    }

    Ok(out)
}

/// CCSM-specific CLAHE wrapper (same defaults as reference code).
pub fn clahe_ccsm(image: &ArrayView2<u8>) -> Result<Array2<u8>> {
    clahe_u8(image, 0.03, 16)
}

/// CCSM-specific CLAHE wrapper with optional GPU acceleration.
pub fn clahe_ccsm_with_gpu(image: &ArrayView2<u8>, use_gpu: bool) -> Result<Array2<u8>> {
    clahe_u8_with_gpu(image, 0.03, 16, use_gpu)
}

fn compute_clahe_gpu(
    image: &ArrayView2<u8>,
    clip_limit: f32,
    kernel_size: usize,
) -> Result<Array2<u8>> {
    let backend_handle = GpuBackend::get_or_init()?;
    let backend_guard = backend_handle
        .lock()
        .map_err(|_| FeaturizerError::CudaError("GPU backend lock poisoned".to_string()))?;
    let backend = backend_guard
        .as_ref()
        .ok_or_else(|| FeaturizerError::CudaError("GPU backend unavailable".to_string()))?;
    compute_clahe_wgpu(image, clip_limit, kernel_size, backend)
}

fn interpolation_indices(coord: usize, n_tiles: usize, tile_size: usize) -> (usize, usize, f64) {
    let g = coord as f64 / tile_size as f64 - 0.5;
    let mut i0 = g.floor() as isize;
    let mut i1 = i0 + 1;
    let mut w = g - i0 as f64;

    if i0 < 0 {
        i0 = 0;
        i1 = 0;
        w = 0.0;
    }
    if i1 >= n_tiles as isize {
        i1 = n_tiles as isize - 1;
        i0 = i1;
        w = 0.0;
    }

    (i0 as usize, i1 as usize, w.clamp(0.0, 1.0))
}

fn clip_histogram(hist: &mut [u32; N_BINS], clip_limit: u32) {
    if clip_limit == 0 {
        return;
    }

    // First clipping pass and excess count.
    let mut n_excess: i64 = 0;
    for h in hist.iter_mut() {
        if *h > clip_limit {
            n_excess += (*h - clip_limit) as i64;
            *h = clip_limit;
        }
    }

    if n_excess <= 0 {
        return;
    }

    // Match skimage redistribution strategy.
    let bin_incr = (n_excess as u64 / N_BINS as u64) as u32;
    let upper = clip_limit.saturating_sub(bin_incr);

    if bin_incr > 0 {
        for h in hist.iter_mut() {
            if *h < upper {
                *h += bin_incr;
                n_excess -= bin_incr as i64;
            } else if *h >= upper && *h < clip_limit {
                n_excess += (*h as i64) - clip_limit as i64;
                *h = clip_limit;
            }
        }
    }

    while n_excess > 0 {
        let prev = n_excess;
        let n_under = hist.iter().filter(|&&v| v < clip_limit).count();
        if n_under == 0 {
            break;
        }
        let step_size = (n_under / n_excess as usize).max(1);
        let mut idx = 0usize;
        while idx < N_BINS && n_excess > 0 {
            if hist[idx] < clip_limit {
                hist[idx] += 1;
                n_excess -= 1;
            }
            idx += step_size;
        }
        if prev == n_excess {
            break;
        }
    }
}

fn histogram_to_lut(hist: &[u32; N_BINS], n_pixels: u32) -> [u8; N_BINS] {
    let mut lut = [0_u8; N_BINS];
    if n_pixels == 0 {
        return lut;
    }

    let mut cumsum = 0_u32;
    for i in 0..N_BINS {
        cumsum = cumsum.saturating_add(hist[i]);
        let mapped = ((cumsum as f64 * 255.0) / n_pixels as f64)
            .floor()
            .clamp(0.0, 255.0);
        lut[i] = mapped as u8;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clahe_shape_preserved() {
        let mut image = Array2::<u8>::zeros((32, 40));
        for y in 0..32 {
            for x in 0..40 {
                image[[y, x]] = ((x + y) % 256) as u8;
            }
        }
        let out = clahe_u8(&image.view(), 0.03, 8).unwrap();
        assert_eq!(out.dim(), image.dim());
    }

    #[test]
    fn test_clahe_uniform_stays_uniform() {
        let image = Array2::<u8>::from_elem((24, 24), 80);
        let out = clahe_u8(&image.view(), 0.03, 8).unwrap();
        let first = out[[0, 0]];
        assert!(out.iter().all(|&v| v == first));
    }

    #[test]
    fn test_clahe_enhances_low_contrast_region() {
        let mut image = Array2::<u8>::from_elem((32, 32), 120);
        for y in 8..24 {
            for x in 8..24 {
                image[[y, x]] = 124;
            }
        }
        let out = clahe_u8(&image.view(), 0.03, 8).unwrap();
        let in_region = out[[16, 16]];
        let out_region = out[[2, 2]];
        assert!(in_region != out_region);
    }

    #[test]
    fn test_clahe_gpu_flag_small_image_matches_cpu() {
        let mut image = Array2::<u8>::zeros((24, 24));
        for y in 0..24 {
            for x in 0..24 {
                image[[y, x]] = ((x + y) % 256) as u8;
            }
        }
        let cpu = clahe_u8_with_gpu(&image.view(), 0.03, 8, false).unwrap();
        let gpu_flag = clahe_u8_with_gpu(&image.view(), 0.03, 8, true).unwrap();
        assert_eq!(cpu, gpu_flag);
    }

    #[test]
    fn test_clahe_batch_gpu_flag_small_images_match_cpu() {
        let mut img1 = Array2::<u8>::zeros((24, 24));
        let mut img2 = Array2::<u8>::zeros((24, 24));
        for y in 0..24 {
            for x in 0..24 {
                img1[[y, x]] = ((x + y) % 256) as u8;
                img2[[y, x]] = ((2 * x + y) % 256) as u8;
            }
        }
        let images = vec![img1, img2];

        let cpu = clahe_u8_batch_with_gpu(&images, 0.03, 8, false).unwrap();
        let gpu_flag = clahe_u8_batch_with_gpu(&images, 0.03, 8, true).unwrap();
        assert_eq!(cpu, gpu_flag);
    }
}
