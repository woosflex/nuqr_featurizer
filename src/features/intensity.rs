//! Intensity-based statistical features from grayscale pixel values.
//!
//! Port of `calculate_intensity_features` from Final_Code_Features_13.10.py (lines 252-260).
//!
//! Computes:
//! - Basic statistics: mean, median, std, min, max, range, IQR
//! - Higher moments: skewness, kurtosis (via scipy.stats equivalents)
//! - Entropy: Shannon entropy from histogram (256 bins)
//!
//! Uses f32 precision per precision_guidelines.md (inherent 8-bit quantization).

use ndarray::ArrayView2;
use std::collections::HashMap;

use crate::core::{FeaturizerError, Result};

/// Extract intensity features from a grayscale patch with a binary mask.
///
/// # Arguments
/// * `grayscale_patch` - 2D grayscale image (values 0-255, stored as u8 or f32)
/// * `mask` - Boolean mask indicating nucleus pixels
///
/// # Returns
/// HashMap with keys:
/// - `mean_intensity`, `median_intensity`, `std_intensity`
/// - `min_intensity`, `max_intensity`, `range_intensity`, `iqr_intensity`
/// - `skewness_intensity`, `kurtosis_intensity`, `entropy_intensity`
///
/// # Edge Cases
/// - Empty mask (no pixels): returns all zeros
/// - Constant intensity: skewness=0, kurtosis=0, entropy=0
///
/// # Precision
/// Uses f32 for all calculations (8-bit pixel values don't need f64).
pub fn calculate_intensity_features(
    grayscale_patch: &ArrayView2<f32>,
    mask: &ArrayView2<bool>,
) -> Result<HashMap<String, f64>> {
    if grayscale_patch.shape() != mask.shape() {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("{:?}", mask.shape()),
            got: format!("{:?}", grayscale_patch.shape()),
        });
    }

    // Extract pixels within mask
    let intensities: Vec<f32> = grayscale_patch
        .iter()
        .zip(mask.iter())
        .filter_map(|(val, &m)| if m { Some(*val) } else { None })
        .collect();

    // Edge case: empty mask
    if intensities.is_empty() {
        return Ok(empty_features());
    }

    let n = intensities.len() as f64;
    let mut sorted: Vec<f64> = intensities.iter().map(|&v| v as f64).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Basic statistics
    let sum: f64 = sorted.iter().sum();
    let mean = sum / n;

    let median = percentile_linear(&sorted, 50.0);

    let min_val = sorted[0];
    let max_val = sorted[sorted.len() - 1];
    let range = max_val - min_val;

    // Match NumPy percentile interpolation (`method='linear'`).
    let q1 = percentile_linear(&sorted, 25.0);
    let q3 = percentile_linear(&sorted, 75.0);
    let iqr = q3 - q1;

    // Variance and standard deviation
    let var_sum: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum();
    let variance = var_sum / n;
    let std = variance.sqrt();

    // Higher moments (skewness, kurtosis)
    let (skewness, kurtosis) = if variance > 1e-8 {
        let m3: f64 = sorted.iter().map(|x| ((x - mean) / std).powi(3)).sum();
        let m4: f64 = sorted.iter().map(|x| ((x - mean) / std).powi(4)).sum();
        let skew = m3 / n;
        // Excess kurtosis (scipy.stats default): kurtosis - 3
        let kurt = (m4 / n) - 3.0;
        (skew, kurt)
    } else {
        (0.0, 0.0)
    };

    // Entropy from 256-bin histogram
    let entropy = calculate_entropy(&intensities);

    let mut features = HashMap::new();
    features.insert("mean_intensity".to_string(), mean);
    features.insert("median_intensity".to_string(), median);
    features.insert("std_intensity".to_string(), std);
    features.insert("min_intensity".to_string(), min_val);
    features.insert("max_intensity".to_string(), max_val);
    features.insert("range_intensity".to_string(), range);
    features.insert("iqr_intensity".to_string(), iqr);
    features.insert("skewness_intensity".to_string(), skewness);
    features.insert("kurtosis_intensity".to_string(), kurtosis);
    features.insert("entropy_intensity".to_string(), entropy);

    Ok(features)
}

/// Calculate Shannon entropy from intensity histogram.
///
/// Matches Python reference (line 257-259):
/// ```python
/// hist, _ = np.histogram(intensities, bins=256, range=(0, 255), density=True)
/// hist = hist[hist > 0]
/// entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
/// ```
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let q = q.clamp(0.0, 100.0) / 100.0;
    let idx = q * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = idx - lo as f64;
        (1.0 - w) * sorted[lo] + w * sorted[hi]
    }
}

fn calculate_entropy(intensities: &[f32]) -> f64 {
    const N_BINS: usize = 256;
    let mut histogram = [0u32; N_BINS];
    let bin_width = 255.0_f64 / N_BINS as f64;

    // Match numpy histogram(..., bins=256, range=(0, 255), density=True).
    for &val in intensities {
        let v = val.clamp(0.0, 255.0) as f64;
        let bin = if v >= 255.0 {
            N_BINS - 1
        } else {
            ((v / bin_width).floor() as usize).min(N_BINS - 1)
        };
        histogram[bin] += 1;
    }

    let total = intensities.len() as f64;
    let mut entropy = 0.0_f64;
    for &count in &histogram {
        if count > 0 {
            let density = count as f64 / (total * bin_width);
            entropy -= density * density.log2();
        }
    }

    entropy
}

/// Return zero-valued features for empty masks.
fn empty_features() -> HashMap<String, f64> {
    [
        "mean_intensity",
        "median_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
        "range_intensity",
        "iqr_intensity",
        "skewness_intensity",
        "kurtosis_intensity",
        "entropy_intensity",
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
    fn test_empty_mask() {
        let patch = array![[100.0, 150.0], [200.0, 50.0]];
        let mask = array![[false, false], [false, false]];

        let features = calculate_intensity_features(&patch.view(), &mask.view()).unwrap();
        assert_eq!(features.get("mean_intensity"), Some(&0.0));
        assert_eq!(features.get("entropy_intensity"), Some(&0.0));
        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_uniform_intensity() {
        let patch = array![[128.0, 128.0], [128.0, 128.0]];
        let mask = array![[true, true], [true, true]];

        let features = calculate_intensity_features(&patch.view(), &mask.view()).unwrap();
        assert_eq!(features.get("mean_intensity"), Some(&128.0));
        assert_eq!(features.get("std_intensity"), Some(&0.0));
        assert_eq!(features.get("skewness_intensity"), Some(&0.0));
        assert_eq!(features.get("kurtosis_intensity"), Some(&0.0));
        // Match numpy histogram(..., density=True) entropy behavior used
        // by the Python reference implementation.
        let bin_width: f64 = 255.0 / 256.0;
        let density: f64 = 1.0 / bin_width;
        let expected_entropy = -(density * density.log2());
        let entropy = features.get("entropy_intensity").unwrap();
        assert!((entropy - expected_entropy).abs() < 1e-9);
    }

    #[test]
    fn test_basic_statistics() {
        // Simple 2x2 with known values
        let patch = array![[0.0, 255.0], [100.0, 150.0]];
        let mask = array![[true, true], [true, true]];

        let features = calculate_intensity_features(&patch.view(), &mask.view()).unwrap();

        // Mean = (0 + 255 + 100 + 150) / 4 = 126.25
        let mean = features.get("mean_intensity").unwrap();
        assert!((mean - 126.25).abs() < 1e-3);

        // Min = 0, Max = 255, Range = 255
        assert_eq!(features.get("min_intensity"), Some(&0.0));
        assert_eq!(features.get("max_intensity"), Some(&255.0));
        assert_eq!(features.get("range_intensity"), Some(&255.0));

        // Median = (100 + 150) / 2 = 125
        let median = features.get("median_intensity").unwrap();
        assert!((median - 125.0).abs() < 1e-3);
    }

    #[test]
    fn test_partial_mask() {
        let patch = array![[50.0, 100.0, 150.0], [200.0, 250.0, 0.0]];
        let mask = array![[true, false, true], [false, true, false]];

        // Selected pixels: 50, 150, 250
        let features = calculate_intensity_features(&patch.view(), &mask.view()).unwrap();

        let mean = features.get("mean_intensity").unwrap();
        // Mean = (50 + 150 + 250) / 3 = 150.0
        assert!((mean - 150.0).abs() < 1e-3);

        let median = features.get("median_intensity").unwrap();
        assert_eq!(median, &150.0); // Middle value
    }

    #[test]
    fn test_entropy_calculation() {
        // Two distinct values: half 0, half 255
        let patch = array![[0.0, 0.0], [255.0, 255.0]];
        let mask = array![[true, true], [true, true]];

        let features = calculate_intensity_features(&patch.view(), &mask.view()).unwrap();

        // Match reference formula:
        // hist = np.histogram(..., density=True)
        // entropy = -sum(hist * log2(hist))
        let bin_width: f64 = 255.0 / 256.0;
        let density: f64 = 0.5 / bin_width;
        let expected = -2.0 * density * density.log2();
        let entropy = features.get("entropy_intensity").unwrap();
        assert!((entropy - expected).abs() < 1e-9);
    }

    #[test]
    fn test_shape_mismatch() {
        let patch = array![[100.0, 150.0], [200.0, 50.0]];
        let mask = array![[true, false, true]]; // Different shape

        let result = calculate_intensity_features(&patch.view(), &mask.view());
        assert!(result.is_err());
    }
}
