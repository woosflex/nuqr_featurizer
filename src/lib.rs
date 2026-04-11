//! NuQR Featurizer: High-performance histopathology feature extraction
//!
//! This library provides fast, GPU-accelerated feature extraction for histopathology images.
//! It extracts morphological, texture, color, and spatial features from nucleus segmentations.

use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3};
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

mod core;
pub mod features;
pub mod gpu;
pub mod stain_norm;

// Re-export core types
pub use core::{
    extract_patch, init_logging, rgb_to_grayscale, FeatureConfig, FeatureVector, FeaturizerError,
    ImagePatch, NucleusMask, Result,
};
pub use gpu::is_gpu_available;
pub use stain_norm::{normalize_staining_default, VahadaneStainNormalizer};

const PATCH_PADDING: usize = 10;

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

    fn padded(self, image_h: usize, image_w: usize, padding: usize) -> Self {
        Self {
            min_row: self.min_row.saturating_sub(padding),
            max_row: (self.max_row + padding).min(image_h - 1),
            min_col: self.min_col.saturating_sub(padding),
            max_col: (self.max_col + padding).min(image_w - 1),
        }
    }
}

fn feature_error(feature: String, err: FeaturizerError) -> FeaturizerError {
    FeaturizerError::FeatureComputationFailed {
        feature,
        reason: err.to_string(),
    }
}

fn morphology_features_with_offset(
    mask: &ArrayView2<'_, bool>,
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

fn python_style_hu_moments(mask: &ArrayView2<'_, bool>, area: f64) -> Result<HashMap<String, f64>> {
    let mut out: HashMap<String, f64> = (1..=7).map(|i| (format!("hu_moment_{i}"), 0.0)).collect();

    if area <= 10.0 {
        return Ok(out);
    }
    if mask.iter().filter(|&&v| v).count() <= 5 {
        return Ok(out);
    }

    // Mirror the Python reference behavior exactly, including the
    // moments_normalized(binary_mask, order=3) call pattern.
    let raw_moments = features::moments::moments_raw(mask, 3);
    if raw_moments[[0, 0]] <= 1e-10 {
        return Ok(out);
    }
    let (h, w) = mask.dim();
    if h <= 3 || w <= 3 {
        return Ok(out);
    }
    let mu00: f64 = if mask[[0, 0]] { 1.0 } else { 0.0 };
    let mut nu = Array2::<f64>::zeros((4, 4));
    for p in 0..=3 {
        for q in 0..=3 {
            if p + q < 2 {
                nu[[p, q]] = f64::NAN;
            } else {
                let mu_pq = if mask[[p, q]] { 1.0 } else { 0.0 };
                if mu00 == 0.0 {
                    nu[[p, q]] = if mu_pq == 0.0 {
                        f64::NAN
                    } else {
                        f64::INFINITY
                    };
                } else {
                    let exponent = (p + q) as f64 / 2.0 + 1.0;
                    nu[[p, q]] = mu_pq / mu00.powf(exponent);
                }
            }
        }
    }
    let hu = features::moments::moments_hu(&nu.view());
    for i in 1..=7 {
        let key = format!("hu_moment_{i}");
        let value = hu[i - 1];
        out.insert(
            key,
            if value.is_finite() && value.abs() < 1e10 {
                value
            } else {
                0.0
            },
        );
    }
    Ok(out)
}

fn build_instance_regions(
    instance_map: &ArrayView2<'_, u32>,
) -> Vec<(u32, BoundingBox, Array2<bool>)> {
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

    let mut out = Vec::with_capacity(bboxes.len());
    for (instance_id, bbox) in bboxes {
        let crop_h = bbox.max_row - bbox.min_row + 1;
        let crop_w = bbox.max_col - bbox.min_col + 1;
        let mut tight_mask = Array2::<bool>::from_elem((crop_h, crop_w), false);
        for row in bbox.min_row..=bbox.max_row {
            for col in bbox.min_col..=bbox.max_col {
                if instance_map[[row, col]] == instance_id {
                    tight_mask[[row - bbox.min_row, col - bbox.min_col]] = true;
                }
            }
        }
        out.push((instance_id, bbox, tight_mask));
    }
    out
}

fn build_patch_mask(
    instance_map: &ArrayView2<'_, u32>,
    instance_id: u32,
    bounds: BoundingBox,
) -> Array2<bool> {
    let patch_h = bounds.max_row - bounds.min_row + 1;
    let patch_w = bounds.max_col - bounds.min_col + 1;
    let mut mask = Array2::<bool>::from_elem((patch_h, patch_w), false);
    for row in bounds.min_row..=bounds.max_row {
        for col in bounds.min_col..=bounds.max_col {
            if instance_map[[row, col]] == instance_id {
                mask[[row - bounds.min_row, col - bounds.min_col]] = true;
            }
        }
    }
    mask
}

fn crop_rgb_patch(image: &ArrayView3<'_, u8>, bounds: BoundingBox) -> Array3<u8> {
    image
        .slice(s![
            bounds.min_row..=bounds.max_row,
            bounds.min_col..=bounds.max_col,
            ..
        ])
        .to_owned()
}

fn to_f32_grayscale(gray_u8: &Array2<u8>) -> Array2<f32> {
    gray_u8.mapv(|v| v as f32)
}

fn normalized_feature_name(key: &str) -> String {
    match key {
        "ccsm_mean_dist_to_edge" => "ccsm_mean_dist_to_boundary".to_string(),
        "ccsm_texture_contrast" => "ccsm_contrast".to_string(),
        "ccsm_texture_correlation" => "ccsm_correlation".to_string(),
        "ccsm_texture_energy" => "ccsm_energy".to_string(),
        "ccsm_texture_homogeneity" => "ccsm_homogeneity".to_string(),
        _ => key.to_string(),
    }
}

fn extend_prefixed_features(
    target: &mut HashMap<String, f64>,
    prefix: &str,
    features_map: HashMap<String, f64>,
) {
    for (key, value) in features_map {
        let normalized = normalized_feature_name(&key);
        target.insert(format!("{prefix}{normalized}"), value);
    }
}

fn compute_prefixed_patch_features(
    out: &mut HashMap<String, f64>,
    rgb_patch: &ArrayView3<'_, u8>,
    mask_patch: &ArrayView2<'_, bool>,
    use_gpu: bool,
    prefix: &str,
) -> Result<()> {
    let gray_u8 = rgb_to_grayscale(rgb_patch);
    let gray_f32 = to_f32_grayscale(&gray_u8);

    extend_prefixed_features(
        out,
        prefix,
        features::intensity::calculate_intensity_features(&gray_f32.view(), mask_patch)?,
    );
    extend_prefixed_features(
        out,
        prefix,
        features::glcm::calculate_glcm_features_with_gpu(&gray_f32.view(), mask_patch, use_gpu)?,
    );
    extend_prefixed_features(
        out,
        prefix,
        features::lbp::calculate_lbp_features(&gray_f32.view(), mask_patch)?,
    );
    extend_prefixed_features(
        out,
        prefix,
        features::he_color::calculate_he_color_features(rgb_patch, mask_patch)?,
    );
    extend_prefixed_features(
        out,
        prefix,
        features::hog::calculate_hog_features(&gray_f32.view(), mask_patch, use_gpu)?,
    );
    extend_prefixed_features(
        out,
        prefix,
        features::ccsm::calculate_ccsm_features_with_gpu(&gray_f32.view(), mask_patch, use_gpu)?,
    );
    Ok(())
}

fn add_nearest_neighbor_distances(feature_maps: &mut [HashMap<String, f64>]) {
    let centroids: Vec<(f64, f64)> = feature_maps
        .iter()
        .map(|map| {
            (
                map.get("centroid_row").copied().unwrap_or(0.0),
                map.get("centroid_col").copied().unwrap_or(0.0),
            )
        })
        .collect();

    for (idx, map) in feature_maps.iter_mut().enumerate() {
        let distance = if centroids.len() <= 1 {
            0.0
        } else {
            let (cy, cx) = centroids[idx];
            let mut min_dist = f64::INFINITY;
            for (j, &(oy, ox)) in centroids.iter().enumerate() {
                if j == idx {
                    continue;
                }
                let d = ((cy - oy).powi(2) + (cx - ox).powi(2)).sqrt();
                if d < min_dist {
                    min_dist = d;
                }
            }
            if min_dist.is_finite() {
                min_dist
            } else {
                0.0
            }
        };
        map.insert("distance_to_nearest_neighbor".to_string(), distance);
    }
}

fn normalized_image_for_feature_extraction(image: &ArrayView3<'_, u8>) -> Array3<u8> {
    let enable_norm = std::env::var("NUQR_ENABLE_STAIN_NORMALIZATION")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "yes" || v == "on"
        })
        .unwrap_or(false);

    if enable_norm {
        normalize_staining_default(image).unwrap_or_else(|_| image.to_owned())
    } else {
        image.to_owned()
    }
}

fn extract_all_features_from_instance_map(
    image: &ArrayView3<'_, u8>,
    instance_map: &ArrayView2<'_, u32>,
    use_gpu: bool,
) -> Result<Vec<HashMap<String, f64>>> {
    let normalized_image = normalized_image_for_feature_extraction(image);
    let normalized_view = normalized_image.view();
    let (image_h, image_w, _) = image.dim();

    let regions = build_instance_regions(instance_map);
    let mut outputs = Vec::with_capacity(regions.len());

    for (instance_id, bbox, _tight_mask) in regions {
        let patch_bounds = bbox.padded(image_h, image_w, PATCH_PADDING);
        let patch_mask = build_patch_mask(instance_map, instance_id, patch_bounds);

        let mut feature_map = morphology_features_with_offset(
            &patch_mask.view(),
            patch_bounds.min_row,
            patch_bounds.min_col,
        )
        .map_err(|err| feature_error(format!("morphology(instance_id={instance_id})"), err))?;
        feature_map.insert("nucleus_id".to_string(), instance_id as f64);

        let area = feature_map.get("area").copied().unwrap_or(0.0);
        let hu = python_style_hu_moments(&patch_mask.view(), area)
            .map_err(|err| feature_error(format!("hu_moments(instance_id={instance_id})"), err))?;
        feature_map.extend(hu);

        let pre_rgb_patch = crop_rgb_patch(image, patch_bounds);
        compute_prefixed_patch_features(
            &mut feature_map,
            &pre_rgb_patch.view(),
            &patch_mask.view(),
            use_gpu,
            "pre_norm_",
        )
        .map_err(|err| feature_error(format!("pre_norm(instance_id={instance_id})"), err))?;

        let advanced_shape = features::shape::calculate_advanced_shape_features(&patch_mask.view())
            .map_err(|err| {
                feature_error(format!("advanced_shape(instance_id={instance_id})"), err)
            })?;
        feature_map.extend(advanced_shape);
        let neis = features::neis::calculate_neis_features(&patch_mask.view())
            .map_err(|err| feature_error(format!("neis(instance_id={instance_id})"), err))?;
        feature_map.extend(neis);

        let post_rgb_patch = crop_rgb_patch(&normalized_view, patch_bounds);
        compute_prefixed_patch_features(
            &mut feature_map,
            &post_rgb_patch.view(),
            &patch_mask.view(),
            use_gpu,
            "post_norm_",
        )
        .map_err(|err| feature_error(format!("post_norm(instance_id={instance_id})"), err))?;

        outputs.push(feature_map);
    }

    add_nearest_neighbor_distances(&mut outputs);
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
        let image_view = image_arr.as_array();

        let use_gpu = use_gpu.unwrap_or(false);
        if use_gpu && !is_gpu_available() {
            return Err(pyo3::PyErr::from(FeaturizerError::CudaError(
                "GPU requested but no compatible WGPU adapter is available".to_string(),
            )));
        }

        let mut results: Vec<HashMap<String, f64>> = if let Ok(instance_map_arr) =
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
            extract_all_features_from_instance_map(
                &image_view,
                &instance_map_arr.as_array(),
                use_gpu,
            )
            .map_err(pyo3::PyErr::from)?
        } else {
            let masks_list = masks.downcast::<PyList>().map_err(|_| {
                PyTypeError::new_err(
                    "masks must be either a 2D numpy uint32 instance map or a list of 2D numpy bool arrays",
                )
            })?;

            let normalized_image = normalized_image_for_feature_extraction(&image_view);
            let normalized_view = normalized_image.view();
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

                let mask_view = mask_arr.as_array();
                let mut feature_map = morphology_features_with_offset(&mask_view, 0, 0)
                    .map_err(|err| feature_error(format!("morphology(mask_index={idx})"), err))
                    .map_err(pyo3::PyErr::from)?;
                feature_map.insert("nucleus_id".to_string(), (idx + 1) as f64);
                let area = feature_map.get("area").copied().unwrap_or(0.0);
                feature_map.extend(
                    python_style_hu_moments(&mask_view, area)
                        .map_err(|err| feature_error(format!("hu_moments(mask_index={idx})"), err))
                        .map_err(pyo3::PyErr::from)?,
                );

                let (pre_rgb_patch, _, pre_patch_mask) =
                    extract_patch(&image_view, &mask_view, PATCH_PADDING)
                        .map_err(|err| feature_error(format!("pre_patch(mask_index={idx})"), err))
                        .map_err(pyo3::PyErr::from)?;
                compute_prefixed_patch_features(
                    &mut feature_map,
                    &pre_rgb_patch.view(),
                    &pre_patch_mask.view(),
                    use_gpu,
                    "pre_norm_",
                )
                .map_err(|err| feature_error(format!("pre_norm(mask_index={idx})"), err))
                .map_err(pyo3::PyErr::from)?;
                feature_map.extend(
                    features::shape::calculate_advanced_shape_features(&pre_patch_mask.view())
                        .map_err(|err| {
                            feature_error(format!("advanced_shape(mask_index={idx})"), err)
                        })
                        .map_err(pyo3::PyErr::from)?,
                );
                feature_map.extend(
                    features::neis::calculate_neis_features(&pre_patch_mask.view())
                        .map_err(|err| feature_error(format!("neis(mask_index={idx})"), err))
                        .map_err(pyo3::PyErr::from)?,
                );

                let (post_rgb_patch, _, post_patch_mask) =
                    extract_patch(&normalized_view, &mask_view, PATCH_PADDING)
                        .map_err(|err| feature_error(format!("post_patch(mask_index={idx})"), err))
                        .map_err(pyo3::PyErr::from)?;
                compute_prefixed_patch_features(
                    &mut feature_map,
                    &post_rgb_patch.view(),
                    &post_patch_mask.view(),
                    use_gpu,
                    "post_norm_",
                )
                .map_err(|err| feature_error(format!("post_norm(mask_index={idx})"), err))
                .map_err(pyo3::PyErr::from)?;

                out.push(feature_map);
            }
            out
        };

        add_nearest_neighbor_distances(&mut results);

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
