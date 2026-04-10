//! Nuclear Envelope Irregularity Score (NEIS) features.
//!
//! Port of `calculate_neis_features` from `Final_Code_Features_13.10.py`.

use std::collections::HashMap;
use std::f64::consts::PI;

use ndarray::ArrayView2;
use rustfft::FftPlanner;

use crate::core::Result;
use crate::features::morphology::calculate_centroid;
use crate::features::shape::largest_external_contour;

const DEFAULT_NUM_SAMPLES: usize = 64;

/// Calculate NEIS features using default `num_samples=64`.
///
/// Returned keys:
/// - `neis_irregularity_score`
/// - `neis_spectral_energy`
/// - `neis_spectral_peak_mode`
pub fn calculate_neis_features(mask: &ArrayView2<bool>) -> Result<HashMap<String, f64>> {
    calculate_neis_features_with_samples(mask, DEFAULT_NUM_SAMPLES)
}

/// Calculate NEIS features with configurable radial samples.
pub fn calculate_neis_features_with_samples(
    mask: &ArrayView2<bool>,
    num_samples: usize,
) -> Result<HashMap<String, f64>> {
    if num_samples < 8 || !mask.iter().any(|&v| v) {
        return Ok(default_features());
    }

    let Some(contour) = largest_external_contour(mask) else {
        return Ok(default_features());
    };
    if contour.len() < 3 {
        return Ok(default_features());
    }

    // Python:
    // M = moments(mask, order=1)
    // cx, cy = M[0,1]/M[0,0], M[1,0]/M[0,0]
    // With our centroid helper: (row, col) = (cy, cx)
    let (cy, cx) = calculate_centroid(mask);

    // Convert contour to polar signal around centroid.
    let mut angle_distance = contour
        .iter()
        .map(|p| {
            let dx = p.x as f64 - cx;
            let dy = p.y as f64 - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx);
            (angle, distance)
        })
        .collect::<Vec<_>>();
    if angle_distance.is_empty() {
        return Ok(default_features());
    }

    angle_distance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_angles = angle_distance.iter().map(|(a, _)| *a).collect::<Vec<_>>();
    let sorted_distances = angle_distance.iter().map(|(_, d)| *d).collect::<Vec<_>>();

    // Wrap to enforce continuity over [theta0, theta0 + 2pi).
    let theta0 = sorted_angles[0];
    let mut wrapped_angles = sorted_angles.clone();
    let mut wrapped_distances = sorted_distances.clone();
    wrapped_angles.push(theta0 + 2.0 * PI);
    wrapped_distances.push(sorted_distances[0]);

    let step = 2.0 * PI / num_samples as f64;
    let target_angles = (0..num_samples)
        .map(|i| theta0 + i as f64 * step)
        .collect::<Vec<_>>();
    let radial_signal = interp_linear(&target_angles, &wrapped_angles, &wrapped_distances);

    // FFT of radial signal.
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(num_samples);
    let mut signal = radial_signal
        .into_iter()
        .map(|v| rustfft::num_complex::Complex::new(v, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut signal);

    let half = num_samples / 2;
    let mut power_spectrum = vec![0.0_f64; half];
    for i in 0..half {
        power_spectrum[i] = signal[i].norm_sqr();
    }

    // Python slicing:
    // low = sum(power[2:6]), high = sum(power[6:])
    let low_start = 2.min(power_spectrum.len());
    let low_end = 6.min(power_spectrum.len());
    let low_freq_power = power_spectrum[low_start..low_end].iter().sum::<f64>();

    let high_start = 6.min(power_spectrum.len());
    let high_freq_power = power_spectrum[high_start..].iter().sum::<f64>();

    let irregularity = if low_freq_power > 0.0 {
        high_freq_power / low_freq_power
    } else {
        0.0
    };

    let spectral_energy = if power_spectrum.len() > 1 {
        power_spectrum[1..].iter().sum::<f64>()
    } else {
        0.0
    };

    // Python:
    // high_freq_spectrum = power[4:]
    // peak_mode = argmax(high_freq_spectrum) + 4 else 4
    let hf_start = 4.min(power_spectrum.len());
    let high_freq_spectrum = &power_spectrum[hf_start..];
    let peak_mode = if high_freq_spectrum.is_empty() {
        4.0
    } else {
        let mut max_idx = 0usize;
        let mut max_val = high_freq_spectrum[0];
        for (i, &v) in high_freq_spectrum.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        (max_idx + 4) as f64
    };

    let mut out = HashMap::new();
    out.insert(
        "neis_irregularity_score".to_string(),
        finite_or_zero(irregularity),
    );
    out.insert(
        "neis_spectral_energy".to_string(),
        finite_or_zero(spectral_energy),
    );
    out.insert(
        "neis_spectral_peak_mode".to_string(),
        finite_or_zero(peak_mode),
    );
    Ok(out)
}

fn default_features() -> HashMap<String, f64> {
    [
        ("neis_irregularity_score".to_string(), 0.0),
        ("neis_spectral_energy".to_string(), 0.0),
        ("neis_spectral_peak_mode".to_string(), 0.0),
    ]
    .into_iter()
    .collect()
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn interp_linear(x: &[f64], xp: &[f64], fp: &[f64]) -> Vec<f64> {
    if xp.is_empty() || fp.is_empty() || xp.len() != fp.len() {
        return vec![0.0; x.len()];
    }

    let mut out = Vec::with_capacity(x.len());
    for &xi in x {
        if xi <= xp[0] {
            out.push(fp[0]);
            continue;
        }
        let last = xp.len() - 1;
        if xi >= xp[last] {
            out.push(fp[last]);
            continue;
        }

        let mut hi =
            match xp.binary_search_by(|v| v.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less)) {
                Ok(idx) => idx,
                Err(idx) => idx,
            };
        if hi == 0 {
            hi = 1;
        }
        if hi >= xp.len() {
            hi = xp.len() - 1;
        }
        let lo = hi - 1;

        let x0 = xp[lo];
        let x1 = xp[hi];
        let y0 = fp[lo];
        let y1 = fp[hi];

        let yi = if (x1 - x0).abs() < 1e-12 {
            y0
        } else {
            let t = (xi - x0) / (x1 - x0);
            y0 + t * (y1 - y0)
        };
        out.push(yi);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_empty_mask_returns_zeros() {
        let mask = Array2::<bool>::from_elem((16, 16), false);
        let features = calculate_neis_features(&mask.view()).unwrap();
        assert_eq!(features["neis_irregularity_score"], 0.0);
        assert_eq!(features["neis_spectral_energy"], 0.0);
        assert_eq!(features["neis_spectral_peak_mode"], 0.0);
    }

    #[test]
    fn test_small_num_samples_returns_zeros() {
        let mut mask = Array2::<bool>::from_elem((16, 16), false);
        for r in 4..12 {
            for c in 4..12 {
                mask[[r, c]] = true;
            }
        }
        let features = calculate_neis_features_with_samples(&mask.view(), 4).unwrap();
        assert_eq!(features["neis_irregularity_score"], 0.0);
    }

    #[test]
    fn test_neis_square_mask_finite() {
        let mut mask = Array2::<bool>::from_elem((32, 32), false);
        for r in 8..24 {
            for c in 8..24 {
                mask[[r, c]] = true;
            }
        }
        let features = calculate_neis_features(&mask.view()).unwrap();
        assert!(features["neis_irregularity_score"].is_finite());
        assert!(features["neis_spectral_energy"].is_finite());
        assert!(features["neis_spectral_peak_mode"].is_finite());
        assert!(features["neis_spectral_peak_mode"] >= 4.0);
    }

    #[test]
    fn test_interp_linear_basic() {
        let xp = vec![0.0, 1.0, 2.0];
        let fp = vec![0.0, 10.0, 20.0];
        let x = vec![0.5, 1.5];
        let y = interp_linear(&x, &xp, &fp);
        assert!((y[0] - 5.0).abs() < 1e-12);
        assert!((y[1] - 15.0).abs() < 1e-12);
    }
}
