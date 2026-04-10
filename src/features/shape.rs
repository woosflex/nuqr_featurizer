//! Advanced shape features.
//!
//! Port of `calculate_advanced_shape_features` from
//! `Final_Code_Features_13.10.py` (convexity, fractal dimension, roughness,
//! Fourier descriptors, and bending energy).

use std::collections::HashMap;

use image::{GrayImage, Luma};
use imageproc::contours::{find_contours, BorderType, Contour};
use imageproc::drawing::draw_polygon_mut;
use imageproc::geometry::{contour_area, convex_hull};
use imageproc::point::Point;
use ndarray::ArrayView2;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::core::Result;

/// Compute advanced shape features for a nucleus mask.
///
/// Output keys:
/// - `convexity`
/// - `fractal_dimension`
/// - `roughness`
/// - `bending_energy`
/// - `fourier_descriptor_1..5`
pub fn calculate_advanced_shape_features(mask: &ArrayView2<bool>) -> Result<HashMap<String, f64>> {
    if !mask.iter().any(|&v| v) {
        return Ok(default_features());
    }

    let mut features = default_features();
    let area = mask.iter().filter(|&&v| v).count() as f64;
    if area <= 0.0 {
        return Ok(features);
    }

    let perimeter = crate::features::morphology::calculate_perimeter(mask);
    let contour = largest_external_contour(mask);

    // Convexity = convex_area / area
    let convexity = if let Some(points) = contour.as_ref() {
        let hull = convex_hull(points.clone());
        let convex_area = convex_hull_pixel_area(&hull, mask.dim());
        if convex_area.is_finite() && convex_area > 0.0 {
            convex_area / area
        } else {
            0.0
        }
    } else {
        0.0
    };
    features.insert("convexity".to_string(), finite_or_zero(convexity));

    // Fractal dimension = log(perimeter) / log(area)
    let fractal_dimension = if area > 1.0 && perimeter > 0.0 {
        let log_area = area.ln();
        if log_area.abs() > 1e-12 {
            perimeter.ln() / log_area
        } else {
            0.0
        }
    } else {
        0.0
    };
    features.insert(
        "fractal_dimension".to_string(),
        finite_or_zero(fractal_dimension),
    );

    // Roughness = perimeter^2 / area
    let roughness = if area > 0.0 {
        (perimeter * perimeter) / area
    } else {
        0.0
    };
    features.insert("roughness".to_string(), finite_or_zero(roughness));

    // Fourier descriptors
    let fds = calculate_fourier_descriptors(mask)?;
    for (i, &v) in fds.iter().enumerate() {
        features.insert(format!("fourier_descriptor_{}", i + 1), finite_or_zero(v));
    }

    // Bending energy
    let bending = calculate_bending_energy(mask)?;
    features.insert("bending_energy".to_string(), finite_or_zero(bending));

    // Final NaN/Inf guard
    for value in features.values_mut() {
        if !value.is_finite() {
            *value = 0.0;
        }
    }

    Ok(features)
}

/// Compute first 5 Fourier descriptors from the largest external contour.
///
/// Equivalent to Python logic:
/// 1. contour -> complex sequence (x + i y)
/// 2. FFT
/// 3. magnitude normalization by DC component
/// 4. keep first 5 non-DC coefficients
pub fn calculate_fourier_descriptors(mask: &ArrayView2<bool>) -> Result<[f64; 5]> {
    let Some(points) = largest_external_contour(mask) else {
        return Ok([0.0; 5]);
    };
    Ok(fourier_descriptors_from_points(&points))
}

/// Compute curvature-based bending energy from the largest external contour.
///
/// Uses the same formula as Python `_calculate_curvature`:
/// `kappa = |d2x*dy - dx*d2y| / (dx^2 + dy^2)^(3/2)`.
pub fn calculate_bending_energy(mask: &ArrayView2<bool>) -> Result<f64> {
    let Some(points) = largest_external_contour(mask) else {
        return Ok(0.0);
    };
    Ok(bending_energy_from_points(&points))
}

fn default_features() -> HashMap<String, f64> {
    [
        "convexity",
        "fractal_dimension",
        "roughness",
        "bending_energy",
        "fourier_descriptor_1",
        "fourier_descriptor_2",
        "fourier_descriptor_3",
        "fourier_descriptor_4",
        "fourier_descriptor_5",
    ]
    .iter()
    .map(|&k| (k.to_string(), 0.0))
    .collect()
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn convex_hull_pixel_area(hull: &[Point<i32>], dims: (usize, usize)) -> f64 {
    if hull.len() < 3 {
        return 0.0;
    }

    let (h, w) = dims;
    let mut image = GrayImage::new(w as u32, h as u32);
    draw_polygon_mut(&mut image, hull, Luma([255_u8]));
    image.pixels().filter(|p| p.0[0] > 0).count() as f64
}

fn mask_to_padded_gray_image(mask: &ArrayView2<bool>) -> GrayImage {
    let (h, w) = mask.dim();
    let mut image = GrayImage::new((w + 2) as u32, (h + 2) as u32);
    for row in 0..h {
        for col in 0..w {
            let v = if mask[[row, col]] { 255_u8 } else { 0_u8 };
            image.put_pixel((col + 1) as u32, (row + 1) as u32, Luma([v]));
        }
    }
    image
}

pub(crate) fn largest_external_contour(mask: &ArrayView2<bool>) -> Option<Vec<Point<i32>>> {
    let (h, w) = mask.dim();
    let image = mask_to_padded_gray_image(mask);
    let contours = find_contours::<i32>(&image);
    if contours.is_empty() {
        return None;
    }

    let mut best: Option<&Contour<i32>> = None;
    let mut best_area = -1.0_f64;
    let mut best_len = 0_usize;

    // Prefer outer contours first.
    for contour in contours
        .iter()
        .filter(|c| c.border_type == BorderType::Outer)
        .chain(
            contours
                .iter()
                .filter(|c| c.border_type != BorderType::Outer),
        )
    {
        let area = contour_area(&contour.points).abs();
        let len = contour.points.len();
        if area > best_area || ((area - best_area).abs() < 1e-12 && len > best_len) {
            best_area = area;
            best_len = len;
            best = Some(contour);
        }
    }

    best.and_then(|c| {
        let shifted = c
            .points
            .iter()
            .map(|p| Point::new(p.x - 1, p.y - 1))
            .filter(|p| p.x >= 0 && p.x < w as i32 && p.y >= 0 && p.y < h as i32)
            .collect::<Vec<_>>();
        if shifted.is_empty() {
            None
        } else {
            Some(shifted)
        }
    })
}

fn dedup_consecutive_complex(points: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    if points.is_empty() {
        return points;
    }

    let mut out = Vec::with_capacity(points.len());
    out.push(points[0]);
    for p in points.into_iter().skip(1) {
        let last = *out.last().expect("out is non-empty");
        if (p - last).norm() > 1e-10 {
            out.push(p);
        }
    }
    out
}

fn fourier_descriptors_from_points(points: &[Point<i32>]) -> [f64; 5] {
    if points.len() <= 10 {
        return [0.0; 5];
    }

    let mut complex = points
        .iter()
        .map(|p| Complex::new(p.x as f64, p.y as f64))
        .collect::<Vec<_>>();
    complex = dedup_consecutive_complex(complex);
    if complex.len() <= 10 {
        return [0.0; 5];
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(complex.len());
    fft.process(&mut complex);

    let mut magnitudes = complex.iter().map(|c| c.norm()).collect::<Vec<_>>();
    let dc = magnitudes[0];
    if !dc.is_finite() || dc <= 1e-10 {
        return [0.0; 5];
    }

    for m in &mut magnitudes {
        *m /= dc;
    }

    let mut out = [0.0; 5];
    for (i, dst) in out.iter_mut().enumerate() {
        let idx = i + 1; // Skip DC
        if idx < magnitudes.len() {
            let v = magnitudes[idx];
            if v.is_finite() && v < 1e6 {
                *dst = v;
            }
        }
    }
    out
}

fn gradient(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let mut g = vec![0.0; n];
            g[0] = values[1] - values[0];
            for i in 1..(n - 1) {
                g[i] = (values[i + 1] - values[i - 1]) * 0.5;
            }
            g[n - 1] = values[n - 1] - values[n - 2];
            g
        }
    }
}

fn bending_energy_from_points(points: &[Point<i32>]) -> f64 {
    if points.len() <= 10 {
        return 0.0;
    }

    let x = points.iter().map(|p| p.x as f64).collect::<Vec<_>>();
    let y = points.iter().map(|p| p.y as f64).collect::<Vec<_>>();

    let dx = gradient(&x);
    let dy = gradient(&y);
    let d2x = gradient(&dx);
    let d2y = gradient(&dy);

    let mut sum = 0.0;
    for i in 0..points.len() {
        let denom = (dx[i] * dx[i] + dy[i] * dy[i]).powf(1.5).max(1e-10);
        let numer = (d2x[i] * dy[i] - dx[i] * d2y[i]).abs();
        let curvature = numer / denom;
        if curvature.is_finite() && curvature <= 1e6 {
            sum += curvature * curvature;
        }
    }
    finite_or_zero(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn square_mask(size: usize, top: usize, left: usize, side: usize) -> Array2<bool> {
        let mut mask = Array2::from_elem((size, size), false);
        for r in top..(top + side) {
            for c in left..(left + side) {
                mask[[r, c]] = true;
            }
        }
        mask
    }

    #[test]
    fn test_empty_mask_returns_zeros() {
        let mask = Array2::from_elem((16, 16), false);
        let features = calculate_advanced_shape_features(&mask.view()).unwrap();
        for (k, v) in features {
            assert!(v.abs() < 1e-12, "{k} expected 0, got {v}");
        }
    }

    #[test]
    fn test_basic_shape_features_are_finite() {
        let mask = square_mask(32, 8, 8, 12);
        let features = calculate_advanced_shape_features(&mask.view()).unwrap();

        assert!(features["convexity"].is_finite());
        assert!(features["fractal_dimension"].is_finite());
        assert!(features["roughness"].is_finite());
        assert!(features["bending_energy"].is_finite());
        for i in 1..=5 {
            let key = format!("fourier_descriptor_{i}");
            assert!(features[&key].is_finite(), "{key} should be finite");
        }
        assert!(features["roughness"] > 0.0);
    }

    #[test]
    fn test_fourier_descriptors_present() {
        let mask = square_mask(48, 10, 10, 20);
        let fds = calculate_fourier_descriptors(&mask.view()).unwrap();
        assert_eq!(fds.len(), 5);
        assert!(fds.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bending_energy_small_object_zero() {
        let mut mask = Array2::from_elem((8, 8), false);
        mask[[4, 4]] = true;
        let be = calculate_bending_energy(&mask.view()).unwrap();
        assert_eq!(be, 0.0);
    }

    #[test]
    fn test_convexity_reasonable_for_square() {
        let mask = square_mask(32, 10, 10, 8);
        let features = calculate_advanced_shape_features(&mask.view()).unwrap();
        // Convexity should be approximately >= 1 for convex shapes.
        assert!(features["convexity"] >= 0.9);
    }
}
