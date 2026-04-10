//! NuQR Featurizer: High-performance histopathology feature extraction
//!
//! This library provides fast, GPU-accelerated feature extraction for histopathology images.
//! It extracts morphological, texture, color, and spatial features from nucleus segmentations.

use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

mod core;
pub mod features;
pub mod gpu;
pub mod stain_norm; // GPU acceleration (WGPU)

// Re-export core types
pub use core::{
    extract_patch, init_logging, rgb_to_grayscale, FeatureConfig, FeatureVector, FeaturizerError,
    ImagePatch, NucleusMask, Result,
};
pub use gpu::is_gpu_available;
pub use stain_norm::{normalize_staining_default, VahadaneStainNormalizer};

#[derive(Clone, Copy)]
struct BoundingBox {
    min_row: usize,
    max_row: usize,
    min_col: usize,
    max_col: usize,
}

impl BoundingBox {
    fn new(row: usize, col: usize) -> Self {
        Self {
            min_row: row,
            max_row: row,
            min_col: col,
            max_col: col,
        }
    }

    fn include(&mut self, row: usize, col: usize) {
        self.min_row = self.min_row.min(row);
        self.max_row = self.max_row.max(row);
        self.min_col = self.min_col.min(col);
        self.max_col = self.max_col.max(col);
    }
}

fn morphology_features_with_offset(
    mask: &ndarray::ArrayView2<'_, bool>,
    row_offset: usize,
    col_offset: usize,
) -> Result<HashMap<String, f64>> {
    let tuples = features::morphology::calculate_morphological_features(mask)?;
    let mut feature_map = HashMap::with_capacity(tuples.len());
    for (key, value) in tuples {
        let adjusted = match key.as_str() {
            "centroid_row" => value + row_offset as f64,
            "centroid_col" => value + col_offset as f64,
            _ => value,
        };
        feature_map.insert(key, adjusted);
    }
    Ok(feature_map)
}

fn extract_morphology_from_instance_map(
    instance_map: &ndarray::ArrayView2<'_, u32>,
) -> Result<Vec<HashMap<String, f64>>> {
    use ndarray::Array2;

    let mut bboxes: BTreeMap<u32, BoundingBox> = BTreeMap::new();
    for ((row, col), &instance_id) in instance_map.indexed_iter() {
        if instance_id == 0 {
            continue;
        }
        bboxes
            .entry(instance_id)
            .and_modify(|bbox| bbox.include(row, col))
            .or_insert_with(|| BoundingBox::new(row, col));
    }

    let mut outputs = Vec::with_capacity(bboxes.len());
    for (instance_id, bbox) in bboxes {
        let crop_h = bbox.max_row - bbox.min_row + 1;
        let crop_w = bbox.max_col - bbox.min_col + 1;
        let mut cropped_mask = Array2::<bool>::from_elem((crop_h, crop_w), false);

        for row in bbox.min_row..=bbox.max_row {
            for col in bbox.min_col..=bbox.max_col {
                if instance_map[[row, col]] == instance_id {
                    cropped_mask[[row - bbox.min_row, col - bbox.min_col]] = true;
                }
            }
        }

        let feature_map =
            morphology_features_with_offset(&cropped_mask.view(), bbox.min_row, bbox.min_col)
                .map_err(|err| FeaturizerError::FeatureComputationFailed {
                    feature: format!("morphology(instance_id={instance_id})"),
                    reason: err.to_string(),
                })?;
        outputs.push(feature_map);
    }

    Ok(outputs)
}

/// Python module initialization
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging
    init_logging();

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add GPU availability check
    #[pyfn(m)]
    fn check_gpu() -> bool {
        is_gpu_available()
    }

    #[pyfn(m)]
    fn get_gpu_device_count() -> usize {
        if is_gpu_available() {
            1
        } else {
            0
        }
    }

    #[pyfn(m)]
    fn normalize_staining<'py>(
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
        use numpy::{IntoPyArray, PyReadonlyArray3, PyUntypedArrayMethods};
        use pyo3::exceptions::PyTypeError;

        let image_arr: PyReadonlyArray3<u8> = image.extract().map_err(|err| {
            PyTypeError::new_err(format!("image must be a 3D numpy uint8 array: {err}"))
        })?;
        core::numpy_interop::validate_image_shape(image_arr.shape()).map_err(pyo3::PyErr::from)?;

        let normalized =
            normalize_staining_default(&image_arr.as_array()).map_err(pyo3::PyErr::from)?;
        Ok(normalized.into_pyarray_bound(py))
    }

    #[pyfn(m)]
    #[pyo3(signature = (image, masks, use_gpu=None))]
    fn extract_features<'py>(
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        masks: &Bound<'py, PyAny>,
        use_gpu: Option<bool>,
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
        use pyo3::exceptions::PyTypeError;
        use pyo3::types::{PyDict, PyList};

        let image_arr: PyReadonlyArray3<u8> = image.extract()?;
        let image_shape = image_arr.shape();
        core::numpy_interop::validate_image_shape(image_shape).map_err(pyo3::PyErr::from)?;

        let use_gpu = use_gpu.unwrap_or(false);
        if use_gpu && !is_gpu_available() {
            return Err(pyo3::PyErr::from(FeaturizerError::CudaError(
                "GPU requested but no compatible WGPU adapter is available".to_string(),
            )));
        }

        let results: Vec<HashMap<String, f64>> = if let Ok(instance_map_arr) =
            masks.extract::<PyReadonlyArray2<u32>>()
        {
            core::numpy_interop::validate_mask_shape(instance_map_arr.shape())
                .map_err(pyo3::PyErr::from)?;
            let map_shape = instance_map_arr.shape();
            if map_shape[0] != image_shape[0] || map_shape[1] != image_shape[1] {
                return Err(pyo3::PyErr::from(FeaturizerError::InvalidDimensions {
                    expected: format!("({}, {})", image_shape[0], image_shape[1]),
                    got: format!("({}, {})", map_shape[0], map_shape[1]),
                }));
            }
            extract_morphology_from_instance_map(&instance_map_arr.as_array())
                .map_err(pyo3::PyErr::from)?
        } else {
            let masks_list = masks.downcast::<PyList>().map_err(|_| {
                    PyTypeError::new_err(
                        "masks must be either a 2D numpy uint32 instance map or a list of 2D numpy bool arrays",
                    )
                })?;
            let mut out = Vec::with_capacity(masks_list.len());
            for (idx, mask_obj) in masks_list.iter().enumerate() {
                let mask_arr: PyReadonlyArray2<bool> = mask_obj.extract().map_err(|err| {
                    PyTypeError::new_err(format!(
                        "masks[{idx}] must be a 2D numpy bool array: {err}"
                    ))
                })?;
                core::numpy_interop::validate_mask_shape(mask_arr.shape())
                    .map_err(pyo3::PyErr::from)?;
                let (mh, mw) = (mask_arr.shape()[0], mask_arr.shape()[1]);
                if mh != image_shape[0] || mw != image_shape[1] {
                    return Err(pyo3::PyErr::from(FeaturizerError::InvalidDimensions {
                        expected: format!("({}, {})", image_shape[0], image_shape[1]),
                        got: format!("({}, {})", mh, mw),
                    }));
                }
                let feature_map = morphology_features_with_offset(&mask_arr.as_array(), 0, 0)
                    .map_err(|err| FeaturizerError::FeatureComputationFailed {
                        feature: format!("morphology(mask_index={idx})"),
                        reason: err.to_string(),
                    })?;
                out.push(feature_map);
            }
            out
        };

        let py_results = PyList::empty_bound(py);
        for feature_map in results {
            let py_dict = PyDict::new_bound(py);
            for (key, value) in feature_map {
                py_dict.set_item(key, value)?;
            }
            py_results.append(py_dict)?;
        }

        Ok(py_results)
    }

    Ok(())
}
