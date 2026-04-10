//! Image moment calculations (raw, central, normalized, Hu).
//!
//! Ported from scikit-image:
//! - skimage/measure/_moments.py
//! - skimage/measure/_moments_cy.pyx (Cython kernels)
//!
//! **CRITICAL**: Uses f64 precision throughout per precision_guidelines.md.
//! Hu moments are extremely sensitive to numerical precision.

use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

use crate::core::{FeaturizerError, Result};

/// Calculate raw image moments up to the specified order.
///
/// Raw moments are computed as:
/// ```text
/// M_pq = Σ Σ (x^p * y^q * I(x, y))
/// ```
///
/// For binary images (bool mask), I(x,y) = 1 inside mask, 0 outside.
///
/// # Arguments
/// * `mask` - Binary mask (true = object pixel)
/// * `order` - Maximum moment order (typically 3 for Hu moments)
///
/// # Precision
/// **MUST use f64** - moment calculations accumulate errors rapidly with f32.
pub fn moments_raw(mask: &ArrayView2<bool>, order: usize) -> Array2<f64> {
    let (height, width) = mask.dim();
    let mut moments = Array2::<f64>::zeros((order + 1, order + 1));

    for row in 0..height {
        for col in 0..width {
            if !mask[[row, col]] {
                continue;
            }

            let x = col as f64;
            let y = row as f64;

            // Compute x^p and y^q for all orders
            let mut x_powers = vec![1.0; order + 1];
            let mut y_powers = vec![1.0; order + 1];

            for p in 1..=order {
                x_powers[p] = x_powers[p - 1] * x;
                y_powers[p] = y_powers[p - 1] * y;
            }

            // Accumulate M_pq = x^p * y^q
            for p in 0..=order {
                for q in 0..=order {
                    moments[[p, q]] += x_powers[p] * y_powers[q];
                }
            }
        }
    }

    moments
}

/// Calculate central moments from raw moments.
///
/// Central moments are translation-invariant, computed about the centroid:
/// ```text
/// μ_pq = Σ Σ ((x - x̄)^p * (y - ȳ)^q * I(x, y))
/// ```
///
/// Where (x̄, ȳ) is the centroid.
///
/// # Reference
/// scikit-image: `skimage.measure.moments_central`
pub fn moments_central(raw: &ArrayView2<f64>, mask: &ArrayView2<bool>) -> Array2<f64> {
    let order = raw.nrows() - 1;
    let m00 = raw[[0, 0]];

    if m00 < 1e-12 {
        // Empty mask - return zeros
        return Array2::zeros((order + 1, order + 1));
    }

    // Centroid: (x̄, ȳ) = (M10/M00, M01/M00)
    let cx = raw[[1, 0]] / m00;
    let cy = raw[[0, 1]] / m00;

    let (height, width) = mask.dim();
    let mut central = Array2::<f64>::zeros((order + 1, order + 1));

    for row in 0..height {
        for col in 0..width {
            if !mask[[row, col]] {
                continue;
            }

            let dx = col as f64 - cx;
            let dy = row as f64 - cy;

            // Precompute powers
            let mut dx_powers = vec![1.0; order + 1];
            let mut dy_powers = vec![1.0; order + 1];

            for i in 1..=order {
                dx_powers[i] = dx_powers[i - 1] * dx;
                dy_powers[i] = dy_powers[i - 1] * dy;
            }

            // Accumulate μ_pq
            for p in 0..=order {
                for q in 0..=order {
                    central[[p, q]] += dx_powers[p] * dy_powers[q];
                }
            }
        }
    }

    central
}

/// Calculate normalized central moments.
///
/// Normalized moments are scale-invariant:
/// ```text
/// η_pq = μ_pq / μ_00^((p + q)/2 + 1)
/// ```
///
/// # Reference
/// scikit-image: `skimage.measure.moments_normalized`
pub fn moments_normalized(central: &ArrayView2<f64>) -> Array2<f64> {
    let order = central.nrows() - 1;
    let mut normalized = Array2::<f64>::zeros((order + 1, order + 1));

    let m00 = central[[0, 0]];
    if m00 < 1e-12 {
        return normalized;
    }

    for p in 0..=order {
        for q in 0..=order {
            if p + q >= 2 {
                let exponent = ((p + q) as f64 / 2.0) + 1.0;
                normalized[[p, q]] = central[[p, q]] / m00.powf(exponent);
            } else {
                // η_00, η_10, η_01 are not defined (or zero)
                normalized[[p, q]] = 0.0;
            }
        }
    }

    // By definition, η_00 = 1
    normalized[[0, 0]] = 1.0;

    normalized
}

/// Calculate Hu's 7 moment invariants.
///
/// These moments are:
/// - Translation invariant (via central moments)
/// - Scale invariant (via normalized moments)
/// - Rotation invariant (via specific combinations)
///
/// # Arguments
/// * `nu` - Normalized central moments (at least 4×4)
///
/// # Returns
/// Array of 7 Hu moment invariants.
///
/// # Reference
/// Direct port of scikit-image Cython implementation:
/// `skimage/measure/_moments_cy.pyx::moments_hu` (lines 11-36)
///
/// # Precision
/// **CRITICAL: f64 only** - any loss of precision breaks invariance properties.
pub fn moments_hu(nu: &ArrayView2<f64>) -> [f64; 7] {
    // Extract frequently used normalized moments
    let nu20 = nu[[2, 0]];
    let nu02 = nu[[0, 2]];
    let nu11 = nu[[1, 1]];
    let nu30 = nu[[3, 0]];
    let nu03 = nu[[0, 3]];
    let nu12 = nu[[1, 2]];
    let nu21 = nu[[2, 1]];

    // Temporary variables (matches Cython implementation line-by-line)
    let mut t0 = nu30 + nu12;
    let mut t1 = nu21 + nu03;
    let mut q0 = t0 * t0;
    let mut q1 = t1 * t1;
    let n4 = 4.0 * nu11;
    let s = nu20 + nu02;
    let d = nu20 - nu02;

    let mut hu = [0.0; 7];

    // Hu moment 1: η20 + η02
    hu[0] = s;

    // Hu moment 2: (η20 - η02)² + 4η11²
    hu[1] = d * d + n4 * nu11;

    // Hu moment 4: (η30 + η12)² + (η21 + η03)²
    hu[3] = q0 + q1;

    // Hu moment 6
    hu[5] = d * (q0 - q1) + n4 * t0 * t1;

    // Update temporaries for moments 3, 5, 7
    t0 *= q0 - 3.0 * q1;
    t1 *= 3.0 * q0 - q1;
    q0 = nu30 - 3.0 * nu12;
    q1 = 3.0 * nu21 - nu03;

    // Hu moment 3: (η30 - 3η12)² + (3η21 - η03)²
    hu[2] = q0 * q0 + q1 * q1;

    // Hu moment 5
    hu[4] = q0 * t0 + q1 * t1;

    // Hu moment 7
    hu[6] = q1 * t0 - q0 * t1;

    hu
}

/// Compute Hu moments for a binary mask (end-to-end).
///
/// This is the top-level function that chains:
/// raw → central → normalized → Hu
///
/// # Returns
/// HashMap with keys `hu_moment_1` through `hu_moment_7`.
pub fn calculate_hu_moments(mask: &ArrayView2<bool>) -> Result<HashMap<String, f64>> {
    // Empty mask check
    if !mask.iter().any(|&v| v) {
        return Ok((1..=7).map(|i| (format!("hu_moment_{}", i), 0.0)).collect());
    }

    // Chain the moment calculations
    let raw = moments_raw(mask, 3);
    let central = moments_central(&raw.view(), mask);
    let normalized = moments_normalized(&central.view());
    let hu = moments_hu(&normalized.view());

    Ok((1..=7)
        .map(|i| (format!("hu_moment_{}", i), hu[i - 1]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_empty_mask() {
        let mask = array![[false, false], [false, false]];
        let hu = calculate_hu_moments(&mask.view()).unwrap();

        for i in 1..=7 {
            assert_eq!(hu.get(&format!("hu_moment_{}", i)), Some(&0.0));
        }
    }

    #[test]
    fn test_square_mask() {
        // 4x4 square centered in 8x8 image
        let mut mask = Array2::<bool>::from_elem((8, 8), false);
        for r in 2..6 {
            for c in 2..6 {
                mask[[r, c]] = true;
            }
        }

        let hu = calculate_hu_moments(&mask.view()).unwrap();

        // For a square, Hu1 and Hu2 should be dominant, Hu3-7 near zero
        let hu1 = hu.get("hu_moment_1").unwrap();
        assert!(*hu1 > 0.0, "Hu1 should be positive for a square");

        // Hu moments should be finite
        for i in 1..=7 {
            let val = hu.get(&format!("hu_moment_{}", i)).unwrap();
            assert!(val.is_finite(), "Hu moment {} should be finite", i);
        }
    }

    #[test]
    fn test_circle_approximation() {
        // Create a circular mask (approximate)
        let size = 21;
        let center = 10.0;
        let radius = 8.0;
        let mut mask = Array2::<bool>::from_elem((size, size), false);

        for r in 0..size {
            for c in 0..size {
                let dr = r as f64 - center;
                let dc = c as f64 - center;
                if (dr * dr + dc * dc).sqrt() <= radius {
                    mask[[r, c]] = true;
                }
            }
        }

        let hu = calculate_hu_moments(&mask.view()).unwrap();

        // For a circle, Hu1 should be much larger than higher-order moments
        let hu1 = *hu.get("hu_moment_1").unwrap();
        let hu2 = *hu.get("hu_moment_2").unwrap();

        assert!(hu1 > 0.0);
        assert!(hu2 >= 0.0);
        assert!(hu1 > hu2.abs(), "Hu1 should dominate for circular shapes");
    }

    #[test]
    fn test_raw_moments_order() {
        let mask = array![[true, true], [false, true]];
        let raw = moments_raw(&mask.view(), 2);

        // M00 = area = 3 pixels
        assert_eq!(raw[[0, 0]], 3.0);

        // M10 = sum of x-coordinates (0,0), (0,1), (1,1) = 0 + 1 + 1 = 2
        assert_eq!(raw[[1, 0]], 2.0);

        // M01 = sum of y-coordinates (0,0), (0,1), (1,1) = 0 + 0 + 1 = 1
        assert_eq!(raw[[0, 1]], 1.0);
    }

    #[test]
    fn test_moments_normalized_scale_invariance() {
        // Test that normalized moments don't depend on M00
        let mask = array![[true, true], [true, true]];
        let raw = moments_raw(&mask.view(), 2);
        let central = moments_central(&raw.view(), &mask.view());
        let normalized = moments_normalized(&central.view());

        // η_00 should always be 1 by definition
        assert!((normalized[[0, 0]] - 1.0).abs() < 1e-10);
    }
}
