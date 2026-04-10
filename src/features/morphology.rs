//! Morphological feature calculations
//!
//! Implements area, perimeter, centroid, and related shape descriptors.
//! Reference: scikit-image/skimage/measure/_regionprops.py

use crate::core::{FeaturizerError, Result};
use ndarray::{Array2, ArrayView2};

/// Calculate basic morphological features from a binary mask
///
/// Returns a vector of (feature_name, value) pairs for:
/// - area: Number of pixels in the mask
/// - perimeter: Perimeter estimate using 4-connectivity
/// - centroid_row, centroid_col: Center of mass
///
/// # Arguments
/// * `mask` - Binary mask (true = foreground, false = background)
///
/// # Returns
/// Vector of (feature_name, feature_value) tuples
pub fn calculate_basic_morphology(mask: &ArrayView2<bool>) -> Result<Vec<(String, f64)>> {
    let (height, width) = mask.dim();

    if height == 0 || width == 0 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "Non-zero dimensions".to_string(),
            got: format!("({}, {})", height, width),
        });
    }

    // Check if mask is empty
    let area_pixels = calculate_area(mask);

    if area_pixels == 0 {
        return Err(FeaturizerError::EmptyMask);
    }

    let area = area_pixels as f64;

    // Calculate centroid
    let (centroid_row, centroid_col) = calculate_centroid(mask);

    // Calculate perimeter using 4-connectivity
    let perimeter = calculate_perimeter(mask);

    Ok(vec![
        ("area".to_string(), area),
        ("perimeter".to_string(), perimeter),
        ("centroid_row".to_string(), centroid_row),
        ("centroid_col".to_string(), centroid_col),
    ])
}

/// Calculate area as the number of foreground pixels.
pub fn calculate_area(mask: &ArrayView2<bool>) -> usize {
    mask.iter().filter(|&&v| v).count()
}

/// Calculate centroid (center of mass) of binary mask
///
/// Formula: centroid = (Σ row_i, Σ col_j) / N
/// where summation is over all foreground pixels
pub fn calculate_centroid(mask: &ArrayView2<bool>) -> (f64, f64) {
    let mut sum_row = 0.0;
    let mut sum_col = 0.0;
    let mut count = 0.0;

    for ((row, col), &value) in mask.indexed_iter() {
        if value {
            sum_row += row as f64;
            sum_col += col as f64;
            count += 1.0;
        }
    }

    if count > 0.0 {
        (sum_row / count, sum_col / count)
    } else {
        (0.0, 0.0)
    }
}

/// Calculate perimeter using scikit-image's 4-neighborhood LUT estimator.
///
/// Algorithm:
/// 1. Binary erosion with 4-connected structuring element
/// 2. Border = original - eroded
/// 3. Convolve border with kernel [[10, 2, 10], [2, 1, 2], [10, 2, 10]]
/// 4. Sum weighted histogram bins (matching skimage._regionprops_utils.perimeter)
///
/// Reference: scikit-image _regionprops_utils.py lines 417-481
pub fn calculate_perimeter(mask: &ArrayView2<bool>) -> f64 {
    let (height, width) = mask.dim();

    if height == 0 || width == 0 {
        return 0.0;
    }

    let eroded = binary_erosion_4(mask);
    let mut border = Array2::from_elem((height, width), false);

    for row in 0..height {
        for col in 0..width {
            border[[row, col]] = mask[[row, col]] && !eroded[[row, col]];
        }
    }

    let mut histogram = [0_u64; 50];
    for row in 0..height {
        for col in 0..width {
            let code = perimeter_code(&border.view(), row, col);
            histogram[code] += 1;
        }
    }

    histogram
        .iter()
        .enumerate()
        .map(|(code, count)| *count as f64 * perimeter_weight(code))
        .sum()
}

fn binary_erosion_4(mask: &ArrayView2<bool>) -> Array2<bool> {
    let (height, width) = mask.dim();
    let mut eroded = Array2::from_elem((height, width), false);

    // border_value=0 in skimage: border pixels cannot survive erosion.
    if height < 3 || width < 3 {
        return eroded;
    }

    for row in 1..(height - 1) {
        for col in 1..(width - 1) {
            eroded[[row, col]] = mask[[row, col]]
                && mask[[row - 1, col]]
                && mask[[row + 1, col]]
                && mask[[row, col - 1]]
                && mask[[row, col + 1]];
        }
    }

    eroded
}

fn perimeter_code(border: &ArrayView2<bool>, row: usize, col: usize) -> usize {
    let (height, width) = border.dim();
    let offsets = [
        (-1, -1, 10_usize),
        (-1, 0, 2),
        (-1, 1, 10),
        (0, -1, 2),
        (0, 0, 1),
        (0, 1, 2),
        (1, -1, 10),
        (1, 0, 2),
        (1, 1, 10),
    ];

    let mut code = 0;
    for (dr, dc, weight) in offsets {
        let rr = row as isize + dr;
        let cc = col as isize + dc;
        if rr >= 0
            && rr < height as isize
            && cc >= 0
            && cc < width as isize
            && border[[rr as usize, cc as usize]]
        {
            code += weight;
        }
    }

    code
}

fn perimeter_weight(code: usize) -> f64 {
    match code {
        5 | 7 | 15 | 17 | 25 | 27 => 1.0,
        21 | 33 => std::f64::consts::SQRT_2,
        13 | 23 => (1.0 + std::f64::consts::SQRT_2) / 2.0,
        _ => 0.0,
    }
}

/// Calculate all morphological features (extended set)
///
/// This will be expanded in Phase 3 to include:
/// - equivalent_diameter, major_axis_length, minor_axis_length
/// - eccentricity, solidity, extent, convex_area
/// - orientation, euler_number
/// - circularity, aspect_ratio
/// - Hu moments (7 values)
pub fn calculate_morphological_features(mask: &ArrayView2<bool>) -> Result<Vec<(String, f64)>> {
    // Phase 2: Just return basic features
    calculate_basic_morphology(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_centroid_single_pixel() {
        let mut mask = Array2::from_elem((5, 5), false);
        mask[[2, 3]] = true;

        let (row, col) = calculate_centroid(&mask.view());
        assert_eq!(row, 2.0);
        assert_eq!(col, 3.0);
    }

    #[test]
    fn test_centroid_square() {
        let mut mask = Array2::from_elem((10, 10), false);
        // 3x3 square at (4-6, 4-6)
        for i in 4..7 {
            for j in 4..7 {
                mask[[i, j]] = true;
            }
        }

        let (row, col) = calculate_centroid(&mask.view());
        assert!((row - 5.0).abs() < 1e-10);
        assert!((col - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_square() {
        let mut mask = Array2::from_elem((10, 10), false);
        // 3x3 square = 9 pixels
        for i in 0..3 {
            for j in 0..3 {
                mask[[i, j]] = true;
            }
        }

        let features = calculate_basic_morphology(&mask.view()).unwrap();
        let area = features.iter().find(|(k, _)| k == "area").map(|(_, v)| v);
        assert_eq!(area, Some(&9.0));
    }

    #[test]
    fn test_perimeter_square() {
        let mut mask = Array2::from_elem((10, 10), false);
        // 3x3 square perimeter under the skimage LUT estimator.
        for i in 1..4 {
            for j in 1..4 {
                mask[[i, j]] = true;
            }
        }

        let features = calculate_basic_morphology(&mask.view()).unwrap();
        let perim = features
            .iter()
            .find(|(k, _)| k == "perimeter")
            .map(|(_, v)| v);
        assert_eq!(perim, Some(&8.0));
    }

    #[test]
    fn test_perimeter_single_pixel() {
        let mut mask = Array2::from_elem((5, 5), false);
        mask[[2, 2]] = true;

        let perim = calculate_perimeter(&mask.view());
        // Matches skimage's LUT-based perimeter estimator for a single isolated pixel.
        assert_eq!(perim, 0.0);
    }

    #[test]
    fn test_perimeter_line_three_pixels() {
        let mut mask = Array2::from_elem((7, 7), false);
        mask[[3, 2]] = true;
        mask[[3, 3]] = true;
        mask[[3, 4]] = true;

        let perim = calculate_perimeter(&mask.view());
        assert!(perim > 0.0);
        assert!(perim < 8.0);
    }

    #[test]
    fn test_empty_mask_error() {
        let mask = Array2::from_elem((10, 10), false);
        let result = calculate_basic_morphology(&mask.view());
        assert!(result.is_err());
        assert!(matches!(result, Err(FeaturizerError::EmptyMask)));
    }

    #[test]
    fn test_basic_morphology_returns_4_features() {
        let mut mask = Array2::from_elem((10, 10), false);
        mask[[5, 5]] = true;

        let features = calculate_basic_morphology(&mask.view()).unwrap();
        assert_eq!(features.len(), 4);

        let names: Vec<&str> = features.iter().map(|(k, _)| k.as_str()).collect();
        assert!(names.contains(&"area"));
        assert!(names.contains(&"perimeter"));
        assert!(names.contains(&"centroid_row"));
        assert!(names.contains(&"centroid_col"));
    }
}
