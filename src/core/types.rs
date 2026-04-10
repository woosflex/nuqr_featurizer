//! Core data types for nucleus feature extraction

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A nucleus mask with its label
#[derive(Debug, Clone)]
pub struct NucleusMask {
    /// Boolean mask indicating nucleus pixels
    pub mask: Array2<bool>,
    /// Unique label for this nucleus
    pub label: u32,
}

impl NucleusMask {
    /// Create a new nucleus mask
    pub fn new(mask: Array2<bool>, label: u32) -> Self {
        Self { mask, label }
    }

    /// Check if the mask is empty
    pub fn is_empty(&self) -> bool {
        !self.mask.iter().any(|&x| x)
    }

    /// Count the number of pixels in the mask
    pub fn pixel_count(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }

    /// Get the bounding box of the nucleus (min_row, min_col, max_row, max_col)
    pub fn bounding_box(&self) -> Option<(usize, usize, usize, usize)> {
        let (height, width) = self.mask.dim();
        let mut min_row = height;
        let mut min_col = width;
        let mut max_row = 0;
        let mut max_col = 0;

        for (idx, &val) in self.mask.indexed_iter() {
            if val {
                min_row = min_row.min(idx.0);
                min_col = min_col.min(idx.1);
                max_row = max_row.max(idx.0);
                max_col = max_col.max(idx.1);
            }
        }

        if min_row < height && min_col < width {
            Some((min_row, min_col, max_row, max_col))
        } else {
            None
        }
    }
}

/// Image patch containing RGB, grayscale, and mask
#[derive(Debug, Clone)]
pub struct ImagePatch {
    /// RGB image patch (height × width × 3)
    pub rgb: Array3<u8>,
    /// Grayscale image patch (height × width)
    pub grayscale: Array2<u8>,
    /// Boolean mask (height × width)
    pub mask: Array2<bool>,
    /// Original nucleus label
    pub label: u32,
}

impl ImagePatch {
    /// Create a new image patch
    pub fn new(rgb: Array3<u8>, grayscale: Array2<u8>, mask: Array2<bool>, label: u32) -> Self {
        Self {
            rgb,
            grayscale,
            mask,
            label,
        }
    }

    /// Get the dimensions (height, width)
    pub fn dims(&self) -> (usize, usize) {
        (self.rgb.shape()[0], self.rgb.shape()[1])
    }

    /// Check if the patch is empty
    pub fn is_empty(&self) -> bool {
        !self.mask.iter().any(|&x| x)
    }
}

/// Feature vector for a single nucleus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Nucleus ID (label)
    pub nucleus_id: u32,
    /// Feature name → value mapping
    pub features: HashMap<String, f64>,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(nucleus_id: u32) -> Self {
        Self {
            nucleus_id,
            features: HashMap::new(),
        }
    }

    /// Add a feature
    pub fn add_feature(&mut self, name: impl Into<String>, value: f64) {
        self.features.insert(name.into(), value);
    }

    /// Add multiple features from a HashMap
    pub fn add_features(&mut self, features: HashMap<String, f64>) {
        self.features.extend(features);
    }

    /// Get a feature value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.features.get(name).copied()
    }

    /// Get all feature names
    pub fn feature_names(&self) -> Vec<String> {
        self.features.keys().cloned().collect()
    }

    /// Validate that all features are finite (not NaN or Inf)
    pub fn validate(&mut self) {
        for (name, value) in &mut self.features {
            if !value.is_finite() {
                tracing::warn!("Feature {} is not finite: {}, setting to 0.0", name, value);
                *value = 0.0;
            }
        }
    }
}

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Use GPU acceleration if available
    pub use_gpu: bool,
    /// Number of parallel threads (0 = auto)
    pub num_threads: usize,
    /// Padding around nucleus for patch extraction
    pub padding: usize,
    /// Compute morphological features
    pub compute_morphology: bool,
    /// Compute intensity features
    pub compute_intensity: bool,
    /// Compute texture features (GLCM, LBP, HOG)
    pub compute_texture: bool,
    /// Compute shape features (Fourier, curvature, etc.)
    pub compute_shape: bool,
    /// Compute H&E color features
    pub compute_color: bool,
    /// Compute CCSM features
    pub compute_ccsm: bool,
    /// Compute NEIS features
    pub compute_neis: bool,
    /// Compute spatial features (nearest neighbor)
    pub compute_spatial: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            use_gpu: false,
            num_threads: 0, // Auto-detect
            padding: 10,
            compute_morphology: true,
            compute_intensity: true,
            compute_texture: true,
            compute_shape: true,
            compute_color: true,
            compute_ccsm: true,
            compute_neis: true,
            compute_spatial: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nucleus_mask_empty() {
        let mask = Array2::from_elem((10, 10), false);
        let nucleus = NucleusMask::new(mask, 1);
        assert!(nucleus.is_empty());
        assert_eq!(nucleus.pixel_count(), 0);
    }

    #[test]
    fn test_nucleus_mask_bounding_box() {
        let mut mask = Array2::from_elem((10, 10), false);
        mask[[2, 3]] = true;
        mask[[2, 4]] = true;
        mask[[3, 3]] = true;

        let nucleus = NucleusMask::new(mask, 1);
        let bbox = nucleus.bounding_box().unwrap();
        assert_eq!(bbox, (2, 3, 3, 4));
    }

    #[test]
    fn test_feature_vector() {
        let mut features = FeatureVector::new(42);
        features.add_feature("area", 100.0);
        features.add_feature("perimeter", 50.0);

        assert_eq!(features.get("area"), Some(100.0));
        assert_eq!(features.feature_names().len(), 2);
    }

    #[test]
    fn test_feature_vector_validation() {
        let mut features = FeatureVector::new(1);
        features.add_feature("valid", 10.0);
        features.add_feature("nan", f64::NAN);
        features.add_feature("inf", f64::INFINITY);

        features.validate();

        assert_eq!(features.get("valid"), Some(10.0));
        assert_eq!(features.get("nan"), Some(0.0));
        assert_eq!(features.get("inf"), Some(0.0));
    }
}
