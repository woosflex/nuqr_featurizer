//! Spatial features.
//!
//! Port of `calculate_nearest_neighbor_distance` from
//! `Final_Code_Features_13.10.py` (uses `scipy.spatial.distance.cdist` + `min`).

use ndarray::ArrayView2;

use crate::core::{FeaturizerError, Result};

/// Compute distance to nearest neighboring centroid.
///
/// Equivalent behavior to Python:
/// - If no neighbors: returns 0
/// - If nucleus centroid contains NaN: returns 0
///
/// `all_centroids_except_current` must have shape `(N, 2)` where columns are
/// `(row, col)` or `(y, x)` coordinates.
pub fn calculate_nearest_neighbor_distance(
    nucleus_centroid: (f64, f64),
    all_centroids_except_current: &ArrayView2<f64>,
) -> Result<f64> {
    if all_centroids_except_current.ncols() != 2 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "(*, 2)".to_string(),
            got: format!("{:?}", all_centroids_except_current.dim()),
        });
    }

    if all_centroids_except_current.nrows() == 0
        || !nucleus_centroid.0.is_finite()
        || !nucleus_centroid.1.is_finite()
    {
        return Ok(0.0);
    }

    let mut min_dist = f64::INFINITY;
    for row in 0..all_centroids_except_current.nrows() {
        let cy = all_centroids_except_current[[row, 0]];
        let cx = all_centroids_except_current[[row, 1]];
        if !cy.is_finite() || !cx.is_finite() {
            continue;
        }

        let dy = nucleus_centroid.0 - cy;
        let dx = nucleus_centroid.1 - cx;
        let dist = (dy * dy + dx * dx).sqrt();
        if dist < min_dist {
            min_dist = dist;
        }
    }

    if min_dist.is_finite() {
        Ok(min_dist)
    } else {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_empty_neighbors_returns_zero() {
        let neighbors = Array2::<f64>::zeros((0, 2));
        let d = calculate_nearest_neighbor_distance((10.0, 10.0), &neighbors.view()).unwrap();
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_nan_centroid_returns_zero() {
        let neighbors = array![[1.0, 1.0], [5.0, 5.0]];
        let d = calculate_nearest_neighbor_distance((f64::NAN, 2.0), &neighbors.view()).unwrap();
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_correct_min_distance() {
        let neighbors = array![[10.0, 10.0], [13.0, 14.0], [2.0, 3.0]];
        let d = calculate_nearest_neighbor_distance((10.0, 13.0), &neighbors.view()).unwrap();
        // Distances: 3.0, sqrt(10), sqrt(170) -> nearest = 3.0
        assert!((d - 3.0).abs() < 1e-12, "expected 3.0, got {d}");
    }

    #[test]
    fn test_invalid_shape_returns_error() {
        let neighbors = array![[1.0, 2.0, 3.0]];
        let r = calculate_nearest_neighbor_distance((0.0, 0.0), &neighbors.view());
        assert!(r.is_err());
    }
}
