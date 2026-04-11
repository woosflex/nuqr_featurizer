//! Contrast Limited Adaptive Histogram Equalization (CLAHE) for CCSM.
//!
//! This is a 2D grayscale CLAHE implementation aligned with the skimage usage
//! in the Python reference:
//! `equalize_adapthist(img_uint8, clip_limit=0.03, kernel_size=16)`.

use ndarray::{Array2, ArrayView2};

use crate::core::{FeaturizerError, Result};
use crate::gpu::{compute_clahe_wgpu, is_gpu_available, GpuBackend};

const N_BINS: usize = 256;
const NR_OF_GRAY: usize = 1 << 14; // skimage CLAHE internal gray levels
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

    if clahe_gpu_enabled(use_gpu)
        && h >= GPU_MIN_SIZE
        && w >= GPU_MIN_SIZE
        && n_pixels <= GPU_MAX_PIXELS
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

    if !clahe_gpu_enabled(use_gpu) {
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

fn clahe_gpu_enabled(use_gpu: bool) -> bool {
    if !use_gpu || !is_gpu_available() {
        return false;
    }
    std::env::var("NUQR_ENABLE_CLAHE_GPU")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false)
}

fn clahe_u8_cpu(image: &ArrayView2<u8>, clip_limit: f32, kernel_size: usize) -> Result<Array2<u8>> {
    clahe_u8_cpu_exact_skimage(image, clip_limit, kernel_size)
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

    // Match skimage.exposure._adapthist.clip_histogram redistribution sequence.
    let bin_incr = (n_excess / N_BINS as i64) as u32;
    let upper = clip_limit.saturating_sub(bin_incr);

    if bin_incr > 0 {
        for h in hist.iter_mut() {
            if *h < upper {
                *h += bin_incr;
                n_excess -= bin_incr as i64;
            }
        }
    }

    for h in hist.iter_mut() {
        if *h >= upper && *h < clip_limit {
            n_excess += (*h as i64) - clip_limit as i64;
            *h = clip_limit;
        }
    }

    while n_excess > 0 {
        let prev = n_excess;
        for index in 0..N_BINS {
            if n_excess <= 0 {
                break;
            }

            let n_under = hist.iter().filter(|&&v| v < clip_limit).count();
            if n_under == 0 {
                break;
            }
            let step_size = (n_under / n_excess as usize).max(1);

            let mut added = 0_i64;
            let mut idx = index;
            while idx < N_BINS {
                if hist[idx] < clip_limit {
                    hist[idx] += 1;
                    added += 1;
                }
                idx += step_size;
            }
            n_excess -= added;
        }
        if prev == n_excess {
            break;
        }
    }
}

fn preprocess_to_14bit(image: &ArrayView2<u8>) -> Array2<u16> {
    let (h, w) = image.dim();
    let mut img_u16 = Array2::<u16>::zeros((h, w));
    let mut min_v = u16::MAX;
    let mut max_v = u16::MIN;

    for y in 0..h {
        for x in 0..w {
            // skimage img_as_uint for uint8 scales by 257.
            let v = image[[y, x]] as u16 * 257;
            img_u16[[y, x]] = v;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }

    let mut out = Array2::<u16>::zeros((h, w));
    if max_v <= min_v {
        return out;
    }

    let max_gray = (NR_OF_GRAY - 1) as f64;
    let scale = max_gray / (max_v as f64 - min_v as f64);
    for y in 0..h {
        for x in 0..w {
            let v = img_u16[[y, x]] as f64;
            let mapped = ((v - min_v as f64) * scale).round().clamp(0.0, max_gray);
            out[[y, x]] = mapped as u16;
        }
    }
    out
}

fn reflect_index(mut idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let n = len as isize;
    while idx < 0 || idx >= n {
        if idx < 0 {
            idx = -idx;
        }
        if idx >= n {
            idx = 2 * n - idx - 2;
        }
    }
    idx as usize
}

fn reflect_pad_u16(
    image: &ArrayView2<u16>,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) -> Array2<u16> {
    let (h, w) = image.dim();
    let hp = h + pad_top + pad_bottom;
    let wp = w + pad_left + pad_right;
    let mut out = Array2::<u16>::zeros((hp, wp));

    for py in 0..hp {
        let src_y = reflect_index(py as isize - pad_top as isize, h);
        for px in 0..wp {
            let src_x = reflect_index(px as isize - pad_left as isize, w);
            out[[py, px]] = image[[src_y, src_x]];
        }
    }
    out
}

fn map_histogram_u16(hist: &[u32; N_BINS], max_val: u16, n_pixels: u32) -> [u16; N_BINS] {
    let mut out = [0_u16; N_BINS];
    if n_pixels == 0 {
        return out;
    }

    let mut cumsum = 0_u32;
    let scale = max_val as f64 / n_pixels as f64;
    for i in 0..N_BINS {
        cumsum = cumsum.saturating_add(hist[i]);
        let mapped = (cumsum as f64 * scale).clamp(0.0, max_val as f64);
        out[i] = mapped as u16;
    }
    out
}

fn rescale_u16_to_u8_full_range(image: &ArrayView2<u16>) -> Array2<u8> {
    let (h, w) = image.dim();
    let mut min_v = u16::MAX;
    let mut max_v = u16::MIN;
    for &v in image.iter() {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }

    let mut out = Array2::<u8>::zeros((h, w));
    if max_v <= min_v {
        return out;
    }

    let scale = 255.0 / (max_v as f64 - min_v as f64);
    for y in 0..h {
        for x in 0..w {
            let v = image[[y, x]] as f64;
            let mapped = ((v - min_v as f64) * scale).clamp(0.0, 255.0);
            out[[y, x]] = mapped as u8;
        }
    }
    out
}

fn clahe_u8_cpu_exact_skimage(
    image: &ArrayView2<u8>,
    clip_limit: f32,
    kernel_size: usize,
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

    let image14 = preprocess_to_14bit(image);
    let k = kernel_size;
    let pad_top = k / 2;
    let pad_left = k / 2;
    let pad_bottom = ((k - (h % k)) % k) + k.div_ceil(2);
    let pad_right = ((k - (w % k)) % k) + k.div_ceil(2);
    let padded = reflect_pad_u16(&image14.view(), pad_top, pad_bottom, pad_left, pad_right);
    let (hp, wp) = padded.dim();

    let bin_size = 1 + NR_OF_GRAY / N_BINS;
    let mut binned = Array2::<u8>::zeros((hp, wp));
    for y in 0..hp {
        for x in 0..wp {
            binned[[y, x]] = (padded[[y, x]] as usize / bin_size) as u8;
        }
    }

    let ns_hist_y = hp / k - 1;
    let ns_hist_x = wp / k - 1;
    let kernel_elements = (k * k) as u32;
    let clim = if clip_limit > 0.0 {
        (clip_limit * kernel_elements as f32).max(1.0) as u32
    } else {
        kernel_elements
    };

    let mut maps = vec![[0_u16; N_BINS]; ns_hist_y * ns_hist_x];
    let hist_start = k / 2;
    for ty in 0..ns_hist_y {
        let y0 = hist_start + ty * k;
        for tx in 0..ns_hist_x {
            let x0 = hist_start + tx * k;
            let mut hist = [0_u32; N_BINS];
            for dy in 0..k {
                for dx in 0..k {
                    let b = binned[[y0 + dy, x0 + dx]] as usize;
                    hist[b] += 1;
                }
            }
            clip_histogram(&mut hist, clim);
            maps[ty * ns_hist_x + tx] = map_histogram_u16(&hist, (NR_OF_GRAY - 1) as u16, kernel_elements);
        }
    }

    // Edge-padding of contextual maps by one in each dimension.
    let mut map_pad = vec![[0_u16; N_BINS]; (ns_hist_y + 2) * (ns_hist_x + 2)];
    for py in 0..(ns_hist_y + 2) {
        let sy = py.saturating_sub(1).min(ns_hist_y.saturating_sub(1));
        for px in 0..(ns_hist_x + 2) {
            let sx = px.saturating_sub(1).min(ns_hist_x.saturating_sub(1));
            map_pad[py * (ns_hist_x + 2) + px] = maps[sy * ns_hist_x + sx];
        }
    }

    let ns_proc_y = hp / k;
    let ns_proc_x = wp / k;
    let mut result14 = Array2::<u16>::zeros((hp, wp));

    for by in 0..ns_proc_y {
        for bx in 0..ns_proc_x {
            let m00 = &map_pad[by * (ns_hist_x + 2) + bx];
            let m01 = &map_pad[by * (ns_hist_x + 2) + (bx + 1)];
            let m10 = &map_pad[(by + 1) * (ns_hist_x + 2) + bx];
            let m11 = &map_pad[(by + 1) * (ns_hist_x + 2) + (bx + 1)];

            for iy in 0..k {
                let wy = iy as f64 / k as f64;
                for ix in 0..k {
                    let wx = ix as f64 / k as f64;
                    let y = by * k + iy;
                    let x = bx * k + ix;
                    let b = binned[[y, x]] as usize;

                    let v00 = m00[b] as f64;
                    let v01 = m01[b] as f64;
                    let v10 = m10[b] as f64;
                    let v11 = m11[b] as f64;

                    let top = (1.0 - wx) * v00 + wx * v01;
                    let bottom = (1.0 - wx) * v10 + wx * v11;
                    let value = ((1.0 - wy) * top + wy * bottom)
                        .clamp(0.0, (NR_OF_GRAY - 1) as f64);
                    result14[[y, x]] = value as u16;
                }
            }
        }
    }

    let unpadded = result14
        .slice(ndarray::s![pad_top..(hp - pad_bottom), pad_left..(wp - pad_right)])
        .to_owned();

    Ok(rescale_u16_to_u8_full_range(&unpadded.view()))
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
