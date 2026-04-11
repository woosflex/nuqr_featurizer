//! H&E stain color features.
//!
//! Port of `calculate_he_color_features` from `Final_Code_Features_13.10.py`.
//! Uses Ruifrok color deconvolution (`separate_stains`) with a fixed
//! Hematoxylin/Eosin stain matrix, then computes per-channel statistics.

use ndarray::{ArrayView2, ArrayView3};
use std::collections::HashMap;

use crate::core::{FeaturizerError, Result};

const EPS_F32: f32 = 1e-6;
const LOG_ADJUST: f64 = -13.815_510_557_964_274; // ln(1e-6)
const HE_STAIN_MATRIX_FOR_FEATURES: [[f32; 2]; 3] = [
    [0.65, 0.07], // R contributions: H, E
    [0.70, 0.99], // G contributions: H, E
    [0.29, 0.11], // B contributions: H, E
];

/// Calculate H&E color features from an RGB patch and nucleus mask.
///
/// Input RGB values are expected in `[0, 255]` (`u8`).
///
/// Returned feature keys:
/// - `mean_hematoxylin`, `std_hematoxylin`, `skew_hematoxylin`,
///   `kurt_hematoxylin`, `min_hematoxylin`, `max_hematoxylin`
/// - `mean_eosin`, `std_eosin`, `skew_eosin`,
///   `kurt_eosin`, `min_eosin`, `max_eosin`
/// - `he_ratio_H_to_E`
pub fn calculate_he_color_features(
    rgb_patch: &ArrayView3<u8>,
    mask: &ArrayView2<bool>,
) -> Result<HashMap<String, f64>> {
    let (h, w, c) = rgb_patch.dim();
    if c != 3 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "(H, W, 3)".to_string(),
            got: format!("({h}, {w}, {c})"),
        });
    }
    if mask.dim() != (h, w) {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("({h}, {w})"),
            got: format!("{:?}", mask.dim()),
        });
    }

    // Match skimage.color.separate_stains:
    // rgb = max(rgb, 1e-6), stains = (log(rgb)/log(1e-6)) @ conv_matrix, clip >= 0
    let mut h_values = Vec::new();
    let mut e_values = Vec::new();

    for row in 0..h {
        for col in 0..w {
            if !mask[[row, col]] {
                continue;
            }

            // Match skimage separate_stains dtype flow:
            // rgb float32 -> np.log(float32) -> divide by float64 log_adjust -> float64.
            let r = (rgb_patch[[row, col, 0]] as f32 / 255.0).max(EPS_F32);
            let g = (rgb_patch[[row, col, 1]] as f32 / 255.0).max(EPS_F32);
            let b = (rgb_patch[[row, col, 2]] as f32 / 255.0).max(EPS_F32);

            let od_r = (r.ln() as f64) / LOG_ADJUST;
            let od_g = (g.ln() as f64) / LOG_ADJUST;
            let od_b = (b.ln() as f64) / LOG_ADJUST;

            let hema = (od_r * HE_STAIN_MATRIX_FOR_FEATURES[0][0] as f64
                + od_g * HE_STAIN_MATRIX_FOR_FEATURES[1][0] as f64
                + od_b * HE_STAIN_MATRIX_FOR_FEATURES[2][0] as f64)
                .max(0.0);
            let eosin = (od_r * HE_STAIN_MATRIX_FOR_FEATURES[0][1] as f64
                + od_g * HE_STAIN_MATRIX_FOR_FEATURES[1][1] as f64
                + od_b * HE_STAIN_MATRIX_FOR_FEATURES[2][1] as f64)
                .max(0.0);

            h_values.push(hema);
            e_values.push(eosin);
        }
    }

    if h_values.is_empty() {
        return Ok(empty_features());
    }

    let h_stats = channel_stats(&h_values);
    let e_stats = channel_stats(&e_values);

    let mut out = HashMap::new();
    out.insert("mean_hematoxylin".to_string(), h_stats.mean);
    out.insert("std_hematoxylin".to_string(), h_stats.std);
    out.insert("skew_hematoxylin".to_string(), h_stats.skew);
    out.insert("kurt_hematoxylin".to_string(), h_stats.kurtosis);
    out.insert("min_hematoxylin".to_string(), h_stats.min);
    out.insert("max_hematoxylin".to_string(), h_stats.max);

    out.insert("mean_eosin".to_string(), e_stats.mean);
    out.insert("std_eosin".to_string(), e_stats.std);
    out.insert("skew_eosin".to_string(), e_stats.skew);
    out.insert("kurt_eosin".to_string(), e_stats.kurtosis);
    out.insert("min_eosin".to_string(), e_stats.min);
    out.insert("max_eosin".to_string(), e_stats.max);

    let ratio = if e_stats.mean != 0.0 {
        h_stats.mean / e_stats.mean
    } else {
        0.0
    };
    out.insert("he_ratio_H_to_E".to_string(), ratio);

    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct ChannelStats {
    mean: f64,
    std: f64,
    skew: f64,
    kurtosis: f64,
    min: f64,
    max: f64,
}

fn channel_stats(values: &[f64]) -> ChannelStats {
    if values.is_empty() {
        return ChannelStats {
            mean: 0.0,
            std: 0.0,
            skew: 0.0,
            kurtosis: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    let n = values.len() as f64;
    let mean = values.iter().copied().sum::<f64>() / n;
    let min = values
        .iter()
        .fold(f64::INFINITY, |acc, &v| if v < acc { v } else { acc });
    let max = values
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &v| if v > acc { v } else { acc });

    let m2 = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = m2.sqrt();

    let (skew, kurtosis) = if std > 1e-12 {
        let m3 = values
            .iter()
            .map(|&v| {
                let z = (v - mean) / std;
                z * z * z
            })
            .sum::<f64>()
            / n;
        let m4 = values
            .iter()
            .map(|&v| {
                let z = (v - mean) / std;
                z * z * z * z
            })
            .sum::<f64>()
            / n;
        (m3, m4 - 3.0)
    } else {
        (0.0, 0.0)
    };

    ChannelStats {
        mean,
        std,
        skew,
        kurtosis,
        min,
        max,
    }
}

fn empty_features() -> HashMap<String, f64> {
    [
        "mean_hematoxylin",
        "std_hematoxylin",
        "skew_hematoxylin",
        "kurt_hematoxylin",
        "min_hematoxylin",
        "max_hematoxylin",
        "mean_eosin",
        "std_eosin",
        "skew_eosin",
        "kurt_eosin",
        "min_eosin",
        "max_eosin",
        "he_ratio_H_to_E",
    ]
    .iter()
    .map(|&k| (k.to_string(), 0.0))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_shape_mismatch_returns_error() {
        let rgb = array![[[255u8, 200, 150], [10u8, 20, 30]]];
        let mask = array![[true, false], [false, true]];
        let result = calculate_he_color_features(&rgb.view(), &mask.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_mask_returns_zeros() {
        let rgb = array![
            [[255u8, 255, 255], [128u8, 64, 32]],
            [[10u8, 20, 30], [40u8, 50, 60]]
        ];
        let mask = array![[false, false], [false, false]];

        let features = calculate_he_color_features(&rgb.view(), &mask.view()).unwrap();
        assert_eq!(features["mean_hematoxylin"], 0.0);
        assert_eq!(features["mean_eosin"], 0.0);
        assert_eq!(features["he_ratio_H_to_E"], 0.0);
    }

    #[test]
    fn test_all_features_finite() {
        let rgb = array![
            [[180u8, 120, 140], [200u8, 110, 130], [190u8, 115, 125]],
            [[175u8, 125, 145], [185u8, 118, 135], [195u8, 108, 128]],
            [[170u8, 130, 150], [188u8, 116, 132], [198u8, 112, 127]]
        ];
        let mask = array![[true, true, true], [true, true, true], [true, true, true]];

        let features = calculate_he_color_features(&rgb.view(), &mask.view()).unwrap();
        for (k, v) in features {
            assert!(v.is_finite(), "{k} should be finite, got {v}");
        }
    }

    #[test]
    fn test_white_patch_near_zero_stains() {
        let rgb = array![
            [[255u8, 255, 255], [255u8, 255, 255]],
            [[255u8, 255, 255], [255u8, 255, 255]]
        ];
        let mask = array![[true, true], [true, true]];

        let features = calculate_he_color_features(&rgb.view(), &mask.view()).unwrap();
        assert!(features["mean_hematoxylin"].abs() < 1e-6);
        assert!(features["mean_eosin"].abs() < 1e-6);
        assert_eq!(features["he_ratio_H_to_E"], 0.0);
    }
}
