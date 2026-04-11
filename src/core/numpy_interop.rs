//! PyO3 integration and NumPy zero-copy conversions

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::core::{FeaturizerError, Result};

// NOTE: PyReadonlyArray conversions are used inline where needed
// The lifetime of the ArrayView is tied to the PyReadonlyArray,
// so we don't provide standalone conversion functions.

/// Validate image dimensions
pub fn validate_image_shape(shape: &[usize]) -> Result<()> {
    match shape {
        [h, w, 3] if *h > 0 && *w > 0 => Ok(()),
        [h, w, c] => Err(FeaturizerError::InvalidDimensions {
            expected: format!("(H, W, 3) with H,W > 0"),
            got: format!("({}, {}, {})", h, w, c),
        }),
        _ => Err(FeaturizerError::InvalidDimensions {
            expected: format!("3D array (H, W, 3)"),
            got: format!("{}D array", shape.len()),
        }),
    }
}

/// Validate mask dimensions
pub fn validate_mask_shape(shape: &[usize]) -> Result<()> {
    match shape {
        [h, w] if *h > 0 && *w > 0 => Ok(()),
        [h, w] => Err(FeaturizerError::InvalidDimensions {
            expected: format!("(H, W) with H,W > 0"),
            got: format!("({}, {})", h, w),
        }),
        _ => Err(FeaturizerError::InvalidDimensions {
            expected: format!("2D array (H, W)"),
            got: format!("{}D array", shape.len()),
        }),
    }
}

/// Convert RGB to grayscale (weighted average: 0.299R + 0.587G + 0.114B)
pub fn rgb_to_grayscale(rgb: &ArrayView3<u8>) -> Array2<u8> {
    let (height, width, _) = rgb.dim();
    let mut gray = Array2::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            let r = rgb[[i, j, 0]] as u32;
            let g = rgb[[i, j, 1]] as u32;
            let b = rgb[[i, j, 2]] as u32;

            // Match OpenCV COLOR_RGB2GRAY fixed-point path exactly.
            // Equivalent to round((0.299 * R + 0.587 * G + 0.114 * B)).
            let gray_val = ((r * 19_596 + g * 38_470 + b * 7_470 + 32_768) >> 16) as u8;
            gray[[i, j]] = gray_val;
        }
    }

    gray
}

/// Extract a cropped patch from an image with padding
pub fn extract_patch(
    image: &ArrayView3<u8>,
    mask: &ArrayView2<bool>,
    padding: usize,
) -> Result<(Array3<u8>, Array2<u8>, Array2<bool>)> {
    let (img_height, img_width, _) = image.dim();
    let (mask_height, mask_width) = mask.dim();

    if img_height != mask_height || img_width != mask_width {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("image and mask same size"),
            got: format!(
                "image ({}, {}), mask ({}, {})",
                img_height, img_width, mask_height, mask_width
            ),
        });
    }

    // Find bounding box
    let mut min_row = mask_height;
    let mut min_col = mask_width;
    let mut max_row = 0;
    let mut max_col = 0;

    for (idx, &val) in mask.indexed_iter() {
        if val {
            min_row = min_row.min(idx.0);
            min_col = min_col.min(idx.1);
            max_row = max_row.max(idx.0);
            max_col = max_col.max(idx.1);
        }
    }

    if min_row >= mask_height {
        return Err(FeaturizerError::EmptyMask);
    }

    // Add padding
    let pad_min_row = min_row.saturating_sub(padding);
    let pad_min_col = min_col.saturating_sub(padding);
    let pad_max_row = (max_row + padding).min(img_height - 1);
    let pad_max_col = (max_col + padding).min(img_width - 1);

    // Extract patches
    let cropped_image = image
        .slice(ndarray::s![
            pad_min_row..=pad_max_row,
            pad_min_col..=pad_max_col,
            ..
        ])
        .to_owned();

    let cropped_mask = mask
        .slice(ndarray::s![
            pad_min_row..=pad_max_row,
            pad_min_col..=pad_max_col
        ])
        .to_owned();

    // Convert to grayscale
    let rgb_view = cropped_image.view();
    let cropped_gray = rgb_to_grayscale(&rgb_view);

    Ok((cropped_image, cropped_gray, cropped_mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_grayscale() {
        let mut rgb = Array3::zeros((2, 2, 3));
        rgb[[0, 0, 0]] = 255; // Red
        rgb[[0, 1, 1]] = 255; // Green
        rgb[[1, 0, 2]] = 255; // Blue
        rgb[[1, 1, 0]] = 255; // White
        rgb[[1, 1, 1]] = 255;
        rgb[[1, 1, 2]] = 255;

        let gray = rgb_to_grayscale(&rgb.view());

        // Check grayscale values (weighted average)
        assert!(gray[[0, 0]] > 0); // Red
        assert!(gray[[0, 1]] > 0); // Green
        assert!(gray[[1, 0]] > 0); // Blue
        assert_eq!(gray[[1, 1]], 255); // White
    }

    #[test]
    fn test_extract_patch() {
        let image = Array3::zeros((10, 10, 3));
        let mut mask = Array2::from_elem((10, 10), false);
        mask[[5, 5]] = true;
        mask[[5, 6]] = true;

        let result = extract_patch(&image.view(), &mask.view(), 2);
        assert!(result.is_ok());

        let (cropped_img, cropped_gray, cropped_mask) = result.unwrap();
        assert!(cropped_img.dim().0 > 0);
        assert!(cropped_gray.dim().0 > 0);
        assert!(cropped_mask.iter().any(|&x| x));
    }
}
