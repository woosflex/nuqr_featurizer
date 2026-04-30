use std::collections::HashMap;

use ndarray::{Array2, Array3};
use nuxplore::features::ccsm::calculate_ccsm_features_with_gpu;
use nuxplore::features::glcm::calculate_glcm_features_with_gpu;
use nuxplore::features::he_color::calculate_he_color_features;
use nuxplore::features::hog::calculate_hog_features;
use nuxplore::features::intensity::calculate_intensity_features;
use nuxplore::features::lbp::calculate_lbp_features;
use nuxplore::features::moments::calculate_hu_moments;
use nuxplore::features::morphology::calculate_morphological_features;
use nuxplore::features::neis::calculate_neis_features;
use nuxplore::features::shape::calculate_advanced_shape_features;
use nuxplore::features::spatial::calculate_nearest_neighbor_distance;
use nuxplore::{extract_patch, normalize_staining_default, rgb_to_grayscale};

fn paint_disk_u32(map: &mut Array2<u32>, label: u32, cy: isize, cx: isize, radius: isize) {
    let (h, w) = map.dim();
    let r2 = radius * radius;
    for y in 0..h {
        for x in 0..w {
            let dy = y as isize - cy;
            let dx = x as isize - cx;
            if dy * dy + dx * dx <= r2 {
                map[[y, x]] = label;
            }
        }
    }
}

fn paint_disk_bool(mask: &mut Array2<bool>, cy: isize, cx: isize, radius: isize) {
    let (h, w) = mask.dim();
    let r2 = radius * radius;
    for y in 0..h {
        for x in 0..w {
            let dy = y as isize - cy;
            let dx = x as isize - cx;
            if dy * dy + dx * dx <= r2 {
                mask[[y, x]] = true;
            }
        }
    }
}

fn make_synthetic_rgb(h: usize, w: usize) -> Array3<u8> {
    let mut rgb = Array3::<u8>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            rgb[[y, x, 0]] = ((x * 3 + y * 5) % 256) as u8;
            rgb[[y, x, 1]] = ((x * 7 + y * 11 + 37) % 256) as u8;
            rgb[[y, x, 2]] = ((x * 13 + y * 17 + 71) % 256) as u8;
        }
    }
    rgb
}

fn tuple_features_to_map(features: Vec<(String, f64)>) -> HashMap<String, f64> {
    features.into_iter().collect()
}

fn nearest_neighbor_for_index(index: usize, centroids: &[(f64, f64)]) -> f64 {
    if centroids.len() <= 1 {
        return 0.0;
    }
    let mut neighbors = Array2::<f64>::zeros((centroids.len() - 1, 2));
    let mut row = 0;
    for (i, &(cy, cx)) in centroids.iter().enumerate() {
        if i == index {
            continue;
        }
        neighbors[[row, 0]] = cy;
        neighbors[[row, 1]] = cx;
        row += 1;
    }
    calculate_nearest_neighbor_distance(centroids[index], &neighbors.view()).unwrap()
}

fn assert_maps_close(
    cpu: &HashMap<String, f64>,
    gpu: &HashMap<String, f64>,
    abs_tol: f64,
    rel_tol: f64,
) {
    assert_eq!(cpu.len(), gpu.len(), "feature map size mismatch");
    for (key, &cpu_val) in cpu {
        let gpu_val = *gpu
            .get(key)
            .unwrap_or_else(|| panic!("missing key in GPU map: {key}"));
        let diff = (cpu_val - gpu_val).abs();
        let tol = abs_tol + rel_tol * cpu_val.abs().max(gpu_val.abs());
        assert!(
            diff <= tol,
            "feature '{key}' differs too much: cpu={cpu_val}, gpu={gpu_val}, diff={diff}, tol={tol}"
        );
    }
}

#[test]
fn test_full_pipeline_end_to_end_cpu() {
    let (h, w) = (96, 96);
    let rgb = make_synthetic_rgb(h, w);

    // Label 1 intentionally touches image boundaries to exercise contour padding logic.
    let mut instance_map = Array2::<u32>::zeros((h, w));
    paint_disk_u32(&mut instance_map, 1, 8, 8, 12);
    paint_disk_u32(&mut instance_map, 2, 70, 70, 11);

    let labels = [1_u32, 2_u32];
    let mut feature_maps = Vec::<HashMap<String, f64>>::new();
    let mut centroids = Vec::<(f64, f64)>::new();

    for label in labels {
        let full_mask = instance_map.mapv(|id| id == label);
        assert!(
            full_mask.iter().any(|&v| v),
            "expected non-empty mask for {label}"
        );

        let (rgb_patch, gray_patch_u8, patch_mask) =
            extract_patch(&rgb.view(), &full_mask.view(), 4).unwrap();
        let gray_patch = gray_patch_u8.mapv(|v| v as f32);

        let mut features = tuple_features_to_map(
            calculate_morphological_features(&full_mask.view())
                .expect("morphology should succeed in integration pipeline"),
        );
        features.extend(calculate_hu_moments(&full_mask.view()).unwrap());
        features
            .extend(calculate_intensity_features(&gray_patch.view(), &patch_mask.view()).unwrap());
        features.extend(
            calculate_glcm_features_with_gpu(&gray_patch.view(), &patch_mask.view(), false)
                .unwrap(),
        );
        features.extend(calculate_lbp_features(&gray_patch.view(), &patch_mask.view()).unwrap());
        features
            .extend(calculate_hog_features(&gray_patch.view(), &patch_mask.view(), false).unwrap());
        features
            .extend(calculate_he_color_features(&rgb_patch.view(), &patch_mask.view()).unwrap());
        features.extend(calculate_advanced_shape_features(&patch_mask.view()).unwrap());
        features.extend(calculate_neis_features(&patch_mask.view()).unwrap());
        features.extend(
            calculate_ccsm_features_with_gpu(&gray_patch.view(), &patch_mask.view(), false)
                .unwrap(),
        );

        centroids.push((
            *features.get("centroid_row").unwrap(),
            *features.get("centroid_col").unwrap(),
        ));
        feature_maps.push(features);
    }

    for idx in 0..feature_maps.len() {
        let nnd = nearest_neighbor_for_index(idx, &centroids);
        feature_maps[idx].insert("distance_to_nearest_neighbor".to_string(), nnd);
    }

    assert_eq!(feature_maps.len(), 2);
    let required_keys = [
        "area",
        "hu_moment_1",
        "mean_intensity",
        "glcm_contrast",
        "lbp_entropy",
        "hog_mean",
        "mean_hematoxylin",
        "convexity",
        "neis_irregularity_score",
        "ccsm_condensed_area_ratio",
        "distance_to_nearest_neighbor",
    ];

    for features in &feature_maps {
        assert!(
            features.len() >= 60,
            "expected broad integrated feature set, got {}",
            features.len()
        );
        for &key in &required_keys {
            assert!(features.contains_key(key), "missing required key: {key}");
        }
        for (key, &value) in features {
            assert!(value.is_finite(), "non-finite feature {key}={value}");
        }
        assert!(
            features["distance_to_nearest_neighbor"] > 0.0,
            "distance_to_nearest_neighbor should be positive"
        );
    }

    let normalized = normalize_staining_default(&rgb.view()).unwrap();
    assert_eq!(normalized.dim(), rgb.dim());
}

#[test]
fn test_gpu_flag_paths_match_cpu_tolerances() {
    let side = 72;
    let rgb = make_synthetic_rgb(side, side);
    let gray_u8 = rgb_to_grayscale(&rgb.view());
    let gray = gray_u8.mapv(|v| v as f32);

    let mut mask = Array2::<bool>::from_elem((side, side), false);
    paint_disk_bool(&mut mask, 36, 36, 20);

    let hog_cpu = calculate_hog_features(&gray.view(), &mask.view(), false).unwrap();
    let hog_gpu = calculate_hog_features(&gray.view(), &mask.view(), true).unwrap();
    assert_maps_close(&hog_cpu, &hog_gpu, 1e-3, 1e-2);

    let glcm_cpu = calculate_glcm_features_with_gpu(&gray.view(), &mask.view(), false).unwrap();
    let glcm_gpu = calculate_glcm_features_with_gpu(&gray.view(), &mask.view(), true).unwrap();
    assert_maps_close(&glcm_cpu, &glcm_gpu, 1e-4, 1e-3);

    let ccsm_cpu = calculate_ccsm_features_with_gpu(&gray.view(), &mask.view(), false).unwrap();
    let ccsm_gpu = calculate_ccsm_features_with_gpu(&gray.view(), &mask.view(), true).unwrap();
    assert_maps_close(&ccsm_cpu, &ccsm_gpu, 5e-3, 5e-2);
}
