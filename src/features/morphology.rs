//! Morphological feature calculations
//!
//! Implements area, perimeter, centroid, and related shape descriptors.
//! Reference: scikit-image/skimage/measure/_regionprops.py

use crate::core::{FeaturizerError, Result};
use ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

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
    let basic = calculate_basic_morphology(mask)?;
    let area = basic
        .iter()
        .find(|(k, _)| k == "area")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let perimeter = basic
        .iter()
        .find(|(k, _)| k == "perimeter")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let centroid_row = basic
        .iter()
        .find(|(k, _)| k == "centroid_row")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let centroid_col = basic
        .iter()
        .find(|(k, _)| k == "centroid_col")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);

    let equivalent_diameter = 2.0 * (area / std::f64::consts::PI).sqrt();

    let (l1, l2, a, b, c) = inertia_tensor_eigvals(mask, centroid_row, centroid_col, area);
    let major_axis_length = 4.0 * l1.sqrt();
    let minor_axis_length = 4.0 * l2.sqrt();
    let eccentricity = if l1 > 1e-12 {
        (1.0 - l2 / l1).max(0.0).sqrt()
    } else {
        0.0
    };
    let orientation = if (a - c).abs() < 1e-12 {
        if b < 0.0 {
            std::f64::consts::PI / 4.0
        } else {
            -std::f64::consts::PI / 4.0
        }
    } else {
        0.5 * (-2.0 * b).atan2(c - a)
    };

    let (bbox_h, bbox_w) = bbox_size(mask);
    let extent = if bbox_h > 0 && bbox_w > 0 {
        area / (bbox_h * bbox_w) as f64
    } else {
        0.0
    };

    let convex_area = convex_area_from_mask(mask);
    let solidity = if convex_area > 0.0 {
        area / convex_area
    } else {
        0.0
    };

    let euler_number = calculate_euler_number(mask) as f64;
    let circularity = if perimeter > 0.0 {
        (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter)
    } else {
        0.0
    };
    let aspect_ratio = if minor_axis_length > 0.0 {
        major_axis_length / minor_axis_length
    } else {
        0.0
    };

    let mut out = Vec::with_capacity(15);
    out.push(("area".to_string(), area));
    out.push(("perimeter".to_string(), perimeter));
    out.push((
        "equivalent_diameter".to_string(),
        finite_or_zero(equivalent_diameter),
    ));
    out.push((
        "major_axis_length".to_string(),
        finite_or_zero(major_axis_length),
    ));
    out.push((
        "minor_axis_length".to_string(),
        finite_or_zero(minor_axis_length),
    ));
    out.push(("eccentricity".to_string(), finite_or_zero(eccentricity)));
    out.push(("solidity".to_string(), finite_or_zero(solidity)));
    out.push(("extent".to_string(), finite_or_zero(extent)));
    out.push(("circularity".to_string(), finite_or_zero(circularity)));
    out.push(("aspect_ratio".to_string(), finite_or_zero(aspect_ratio)));
    out.push(("orientation".to_string(), finite_or_zero(orientation)));
    out.push(("euler_number".to_string(), finite_or_zero(euler_number)));
    out.push(("convex_area".to_string(), finite_or_zero(convex_area)));
    out.push(("centroid_row".to_string(), centroid_row));
    out.push(("centroid_col".to_string(), centroid_col));

    Ok(out)
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn inertia_tensor_eigvals(
    mask: &ArrayView2<bool>,
    centroid_row: f64,
    centroid_col: f64,
    area: f64,
) -> (f64, f64, f64, f64, f64) {
    if area <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut mu20 = 0.0_f64;
    let mut mu02 = 0.0_f64;
    let mut mu11 = 0.0_f64;
    for ((row, col), &value) in mask.indexed_iter() {
        if !value {
            continue;
        }
        let dr = row as f64 - centroid_row;
        let dc = col as f64 - centroid_col;
        mu20 += dr * dr;
        mu02 += dc * dc;
        mu11 += dr * dc;
    }

    // Match skimage inertia tensor conventions:
    // [[mu02/mu00, -mu11/mu00], [-mu11/mu00, mu20/mu00]]
    let a = mu02 / area;
    let b = -mu11 / area;
    let c = mu20 / area;
    let trace = a + c;
    let det = a * c - b * b;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let mut l1 = ((trace + disc) * 0.5).max(0.0);
    let mut l2 = ((trace - disc) * 0.5).max(0.0);
    if l1 < l2 {
        std::mem::swap(&mut l1, &mut l2);
    }
    (l1, l2, a, b, c)
}

fn bbox_size(mask: &ArrayView2<bool>) -> (usize, usize) {
    let (h, w) = mask.dim();
    let mut min_row = h;
    let mut max_row = 0usize;
    let mut min_col = w;
    let mut max_col = 0usize;
    let mut found = false;

    for ((row, col), &value) in mask.indexed_iter() {
        if !value {
            continue;
        }
        found = true;
        min_row = min_row.min(row);
        max_row = max_row.max(row);
        min_col = min_col.min(col);
        max_col = max_col.max(col);
    }

    if !found {
        return (0, 0);
    }
    (max_row - min_row + 1, max_col - min_col + 1)
}

fn convex_area_from_mask(mask: &ArrayView2<bool>) -> f64 {
    let (h, w) = mask.dim();
    let mut points = Vec::<(f64, f64)>::new();
    points.reserve(calculate_area(mask) * 4);

    // Match skimage convex_hull_image offset behavior by adding
    // edge-midpoint offsets around each foreground pixel.
    for ((row, col), &value) in mask.indexed_iter() {
        if value {
            let y = row as f64;
            let x = col as f64;
            points.push((x, y - 0.5));
            points.push((x, y + 0.5));
            points.push((x - 0.5, y));
            points.push((x + 0.5, y));
        }
    }
    if points.is_empty() {
        return 0.0;
    }

    let hull = convex_hull_f64(points);
    if hull.len() < 3 {
        return calculate_area(mask) as f64;
    }

    let mut count = 0usize;
    for row in 0..h {
        for col in 0..w {
            if point_in_polygon_or_on_edge(col as f64, row as f64, &hull) {
                count += 1;
            }
        }
    }
    count as f64
}

fn convex_hull_f64(mut pts: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    if pts.len() <= 1 {
        return pts;
    }

    let mut lower: Vec<(f64, f64)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<(f64, f64)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

#[inline]
fn cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn point_in_polygon_or_on_edge(x: f64, y: f64, poly: &[(f64, f64)]) -> bool {
    if poly.len() < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];

        if point_on_segment(x, y, xi, yi, xj, yj) {
            return true;
        }

        let intersects = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[inline]
fn point_on_segment(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> bool {
    let cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    if cross.abs() > 1e-9 {
        return false;
    }
    let dot = (px - ax) * (px - bx) + (py - ay) * (py - by);
    dot <= 1e-9
}

fn count_components(mask: &Array2<bool>, target: bool, dirs: &[(isize, isize)]) -> usize {
    let (h, w) = mask.dim();
    let mut visited = Array2::<bool>::from_elem((h, w), false);
    let mut count = 0usize;

    for row in 0..h {
        for col in 0..w {
            if visited[[row, col]] || mask[[row, col]] != target {
                continue;
            }
            count += 1;
            let mut queue = VecDeque::new();
            queue.push_back((row, col));
            visited[[row, col]] = true;

            while let Some((r, c)) = queue.pop_front() {
                for &(dr, dc) in dirs {
                    let rr = r as isize + dr;
                    let cc = c as isize + dc;
                    if rr < 0 || cc < 0 || rr >= h as isize || cc >= w as isize {
                        continue;
                    }
                    let rru = rr as usize;
                    let ccu = cc as usize;
                    if !visited[[rru, ccu]] && mask[[rru, ccu]] == target {
                        visited[[rru, ccu]] = true;
                        queue.push_back((rru, ccu));
                    }
                }
            }
        }
    }

    count
}

fn calculate_euler_number(mask: &ArrayView2<bool>) -> i32 {
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return 0;
    }

    let fg = mask.to_owned();
    let dirs_8 = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let dirs_4 = [(-1, 0), (0, -1), (0, 1), (1, 0)];
    let object_components = count_components(&fg, true, &dirs_8) as i32;

    // Hole counting with 4-connectivity on background and an outer padded frame.
    let mut bg = Array2::<bool>::from_elem((h + 2, w + 2), true);
    for row in 0..h {
        for col in 0..w {
            bg[[row + 1, col + 1]] = !fg[[row, col]];
        }
    }
    let background_components = count_components(&bg, true, &dirs_4) as i32;
    let holes = (background_components - 1).max(0);
    object_components - holes
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

    #[test]
    fn test_full_morphology_includes_extended_keys() {
        let mut mask = Array2::from_elem((16, 16), false);
        for r in 5..10 {
            for c in 6..11 {
                mask[[r, c]] = true;
            }
        }

        let features = calculate_morphological_features(&mask.view()).unwrap();
        let names: Vec<&str> = features.iter().map(|(k, _)| k.as_str()).collect();
        let required = [
            "area",
            "perimeter",
            "equivalent_diameter",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "solidity",
            "extent",
            "circularity",
            "aspect_ratio",
            "orientation",
            "euler_number",
            "convex_area",
            "centroid_row",
            "centroid_col",
        ];

        assert_eq!(features.len(), required.len());
        for &k in &required {
            assert!(names.contains(&k), "missing key: {k}");
        }
    }
}
