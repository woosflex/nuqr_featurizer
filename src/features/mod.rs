//! Feature calculation modules

pub mod ccsm;
pub mod ccsm_clahe;
pub mod ccsm_distance_transform;
pub mod ccsm_gmm;
pub mod ccsm_morphops;
pub mod glcm;
pub mod he_color;
pub mod hog;
pub mod intensity;
pub mod lbp;
pub mod moments;
pub mod morphology;
pub mod neis;
pub mod shape;
pub mod spatial;

use std::collections::HashMap;

use ndarray::Array2;
use rayon::prelude::*;

use crate::core::numpy_interop::{validate_image_shape, validate_mask_shape};
use crate::core::{FeaturizerError, Result};
use crate::gpu::is_gpu_available;

/// Feature key-value map for a single nucleus.
pub type FeatureMap = HashMap<String, f64>;

/// Extract Phase-2 morphology features for a batch of nucleus masks.
///
/// Current output keys per mask:
/// - `area`
/// - `perimeter`
/// - `centroid_row`
/// - `centroid_col`
pub fn extract_morphology_batch(
    image_shape: (usize, usize, usize),
    masks: &[Array2<bool>],
    use_gpu: bool,
) -> Result<Vec<FeatureMap>> {
    validate_image_shape(&[image_shape.0, image_shape.1, image_shape.2])?;

    if use_gpu && !is_gpu_available() {
        return Err(FeaturizerError::CudaError(
            "GPU requested but no compatible WGPU adapter is available".to_string(),
        ));
    }

    for (idx, mask) in masks.iter().enumerate() {
        validate_mask_shape(mask.shape())?;
        let (mh, mw) = mask.dim();
        if mh != image_shape.0 || mw != image_shape.1 {
            return Err(FeaturizerError::InvalidDimensions {
                expected: format!("Mask {} shape ({}, {})", idx, image_shape.0, image_shape.1),
                got: format!("({}, {})", mh, mw),
            });
        }
    }

    let features_per_mask: Vec<Result<FeatureMap>> = masks
        .par_iter()
        .enumerate()
        .map(|(idx, mask)| {
            let tuples =
                morphology::calculate_morphological_features(&mask.view()).map_err(|err| {
                    FeaturizerError::FeatureComputationFailed {
                        feature: format!("morphology(mask_index={idx})"),
                        reason: err.to_string(),
                    }
                })?;
            Ok(tuples.into_iter().collect())
        })
        .collect();

    features_per_mask.into_iter().collect()
}
