//! CCSM morphology primitives:
//! - remove_small_objects
//! - binary_opening

use image::{GrayImage, Luma};
use imageproc::region_labelling::{connected_components, Connectivity};
use ndarray::{Array2, ArrayView2};

use crate::core::{FeaturizerError, Result};

/// Remove connected components smaller than `min_size` (4-connectivity).
pub fn remove_small_objects(mask: &ArrayView2<bool>, min_size: usize) -> Result<Array2<bool>> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "Non-zero mask dimensions".to_string(),
            got: format!("({}, {})", h, w),
        });
    }
    if min_size <= 1 {
        return Ok(mask.to_owned());
    }

    let gray = bool_to_gray(mask);
    let labels = connected_components(&gray, Connectivity::Four, Luma([0u8]));

    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for (_x, _y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0];
        if label != 0 {
            *counts.entry(label).or_insert(0) += 1;
        }
    }

    let mut out = Array2::<bool>::from_elem((h, w), false);
    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0];
        if label != 0 && counts.get(&label).copied().unwrap_or(0) >= min_size {
            out[[y as usize, x as usize]] = true;
        }
    }

    Ok(out)
}

/// Binary opening using a cross-like (L1 radius 1) footprint.
pub fn binary_opening(mask: &ArrayView2<bool>) -> Result<Array2<bool>> {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "Non-zero mask dimensions".to_string(),
            got: format!("({}, {})", h, w),
        });
    }

    // Match skimage.binary_opening with default cross footprint and mode='ignore':
    // - erosion treats outside image as True
    // - dilation treats outside image as False
    const OFFSETS: [(isize, isize); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

    let mut eroded = Array2::<bool>::from_elem((h, w), false);
    for y in 0..h {
        for x in 0..w {
            let mut keep = true;
            for (dy, dx) in OFFSETS {
                let ny = y as isize + dy;
                let nx = x as isize + dx;
                if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
                    // mode='ignore' for erosion: outside treated as True.
                    continue;
                }
                if !mask[[ny as usize, nx as usize]] {
                    keep = false;
                    break;
                }
            }
            eroded[[y, x]] = keep;
        }
    }

    let mut out = Array2::<bool>::from_elem((h, w), false);
    for y in 0..h {
        for x in 0..w {
            let mut on = false;
            for (dy, dx) in OFFSETS {
                let ny = y as isize + dy;
                let nx = x as isize + dx;
                if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
                    // mode='ignore' for dilation: outside treated as False.
                    continue;
                }
                if eroded[[ny as usize, nx as usize]] {
                    on = true;
                    break;
                }
            }
            out[[y, x]] = on;
        }
    }
    Ok(out)
}

/// Apply CCSM morphology sequence.
pub fn apply_ccsm_morphops(mask: &ArrayView2<bool>, min_size: usize) -> Result<Array2<bool>> {
    let filtered = remove_small_objects(mask, min_size)?;
    binary_opening(&filtered.view())
}

fn bool_to_gray(mask: &ArrayView2<bool>) -> GrayImage {
    let (h, w) = mask.dim();
    let mut img = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = if mask[[y, x]] { 255 } else { 0 };
            img.put_pixel(x as u32, y as u32, Luma([v]));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_small_objects() {
        let mut mask = Array2::<bool>::from_elem((10, 10), false);
        mask[[1, 1]] = true; // tiny object
        for y in 5..9 {
            for x in 5..9 {
                mask[[y, x]] = true; // large object
            }
        }

        let out = remove_small_objects(&mask.view(), 4).unwrap();
        assert!(!out[[1, 1]]);
        assert!(out[[6, 6]]);
    }

    #[test]
    fn test_binary_opening_removes_noise_pixel() {
        let mut mask = Array2::<bool>::from_elem((9, 9), false);
        for y in 3..7 {
            for x in 3..7 {
                mask[[y, x]] = true;
            }
        }
        mask[[1, 1]] = true; // isolated pixel

        let out = binary_opening(&mask.view()).unwrap();
        assert!(!out[[1, 1]]);
        assert!(out[[4, 4]]);
    }

    #[test]
    fn test_apply_ccsm_morphops() {
        let mut mask = Array2::<bool>::from_elem((12, 12), false);
        for y in 2..10 {
            for x in 2..10 {
                mask[[y, x]] = true;
            }
        }
        mask[[0, 0]] = true; // tiny noise

        let out = apply_ccsm_morphops(&mask.view(), 10).unwrap();
        assert!(!out[[0, 0]]);
        assert!(out[[6, 6]]);
    }
}
