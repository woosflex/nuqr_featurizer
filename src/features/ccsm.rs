//! Cytoplasm-to-Stroma Spatial Model (CCSM) feature extraction.
//!
//! Port of `calculate_ccsm_features` from `Final_Code_Features_13.10.py`.

use std::collections::HashMap;

use image::{GrayImage, Luma};
use imageproc::drawing::draw_polygon_mut;
use imageproc::geometry::convex_hull;
use imageproc::point::Point;
use imageproc::region_labelling::{connected_components, Connectivity};
use ndarray::{Array2, ArrayView2};

use crate::core::{FeaturizerError, Result};
use crate::features::ccsm_clahe::{clahe_ccsm_with_gpu, clahe_u8_batch_with_gpu};
use crate::features::ccsm_distance_transform::{
    euclidean_distance_transform_batch_with_gpu, euclidean_distance_transform_with_gpu,
};
use crate::features::ccsm_gmm::GaussianMixture1D;
use crate::features::ccsm_morphops::apply_ccsm_morphops;
use crate::features::shape::largest_external_contour;

/// Compute CCSM feature set (11 features).
pub fn calculate_ccsm_features(
    grayscale_patch: &ArrayView2<f32>,
    mask: &ArrayView2<bool>,
) -> Result<HashMap<String, f64>> {
    calculate_ccsm_features_with_gpu(grayscale_patch, mask, false)
}

/// Compute CCSM feature set (11 features) with optional GPU acceleration.
pub fn calculate_ccsm_features_with_gpu(
    grayscale_patch: &ArrayView2<f32>,
    mask: &ArrayView2<bool>,
    use_gpu: bool,
) -> Result<HashMap<String, f64>> {
    if grayscale_patch.shape() != mask.shape() {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("{:?}", mask.shape()),
            got: format!("{:?}", grayscale_patch.shape()),
        });
    }
    let (h, w) = mask.dim();
    if h == 0 || w == 0 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "Non-zero dimensions".to_string(),
            got: format!("({}, {})", h, w),
        });
    }

    let img_u8 = grayscale_to_u8(grayscale_patch);
    let enhanced = clahe_ccsm_with_gpu(&img_u8.view(), use_gpu)?;
    let dist_map = euclidean_distance_transform_with_gpu(mask, use_gpu)?;
    calculate_ccsm_features_from_intermediates(&enhanced.view(), mask, &dist_map.view())
}

/// Compute CCSM features for a batch, reusing batched GPU preprocessing when enabled.
pub fn calculate_ccsm_features_batch_with_gpu(
    grayscale_patches: &[Array2<f32>],
    masks: &[Array2<bool>],
    use_gpu: bool,
) -> Result<Vec<HashMap<String, f64>>> {
    if grayscale_patches.len() != masks.len() {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("{} masks", grayscale_patches.len()),
            got: format!("{} masks", masks.len()),
        });
    }
    if grayscale_patches.is_empty() {
        return Ok(Vec::new());
    }

    let mut patch_u8_batch = Vec::with_capacity(grayscale_patches.len());
    for (idx, (patch, mask)) in grayscale_patches.iter().zip(masks.iter()).enumerate() {
        if patch.shape() != mask.shape() {
            return Err(FeaturizerError::InvalidDimensions {
                expected: format!("Mask {} shape {:?}", idx, patch.shape()),
                got: format!("{:?}", mask.shape()),
            });
        }
        let (h, w) = patch.dim();
        if h == 0 || w == 0 {
            return Err(FeaturizerError::InvalidDimensions {
                expected: "Non-zero dimensions".to_string(),
                got: format!("({}, {})", h, w),
            });
        }
        patch_u8_batch.push(grayscale_to_u8(&patch.view()));
    }

    let enhanced_batch = clahe_u8_batch_with_gpu(&patch_u8_batch, 0.03, 16, use_gpu)?;
    let dist_batch = euclidean_distance_transform_batch_with_gpu(masks, use_gpu)?;

    let mut out = Vec::with_capacity(grayscale_patches.len());
    for i in 0..grayscale_patches.len() {
        out.push(calculate_ccsm_features_from_intermediates(
            &enhanced_batch[i].view(),
            &masks[i].view(),
            &dist_batch[i].view(),
        )?);
    }
    Ok(out)
}

fn calculate_ccsm_features_from_intermediates(
    enhanced: &ArrayView2<u8>,
    mask: &ArrayView2<bool>,
    dist_map: &ArrayView2<f64>,
) -> Result<HashMap<String, f64>> {
    if enhanced.shape() != mask.shape() || dist_map.shape() != mask.shape() {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("enhanced and dist_map shape {:?}", mask.shape()),
            got: format!(
                "enhanced {:?}, dist_map {:?}",
                enhanced.shape(),
                dist_map.shape()
            ),
        });
    }

    let (h, w) = mask.dim();
    let mut features = default_ccsm_features();
    if !mask.iter().any(|&v| v) {
        return Ok(features);
    }

    let mut masked_vals = Vec::<f64>::new();
    masked_vals.reserve(mask.iter().filter(|&&v| v).count());
    for ((r, c), &m) in mask.indexed_iter() {
        if m {
            masked_vals.push(enhanced[[r, c]] as f64);
        }
    }
    if masked_vals.len() <= 1 {
        return Ok(features);
    }

    // 2-component GMM on masked intensities.
    let model = GaussianMixture1D::fit(&masked_vals, 2, 500, 1e-4)?;
    let labels = model.predict_labels(&masked_vals);
    let condensed_label = model.condensed_component_index();

    // Rebuild condensed chromatin mask.
    let mut condensed = Array2::<bool>::from_elem((h, w), false);
    let mut idx = 0usize;
    for ((r, c), &m) in mask.indexed_iter() {
        if m {
            condensed[[r, c]] = labels[idx] == condensed_label;
            idx += 1;
        }
    }

    let condensed = apply_ccsm_morphops(&condensed.view(), 10)?;
    let total_area = mask.iter().filter(|&&v| v).count() as f64;
    let condensed_area = condensed.iter().filter(|&&v| v).count() as f64;
    features.insert(
        "ccsm_condensed_area_ratio".to_string(),
        safe_div(condensed_area, total_area),
    );

    let components = component_pixels(&condensed.view());
    features.insert("ccsm_num_clumps".to_string(), components.len() as f64);

    if components.is_empty() {
        return Ok(features);
    }

    let mut areas = Vec::<f64>::with_capacity(components.len());
    let mut eccentricities = Vec::<f64>::with_capacity(components.len());
    let mut solidities = Vec::<f64>::with_capacity(components.len());
    let mut centroids = Vec::<(f64, f64)>::with_capacity(components.len());

    for pixels in components.values() {
        let area = pixels.len() as f64;
        areas.push(area);

        let (cy, cx) = centroid_from_pixels(pixels);
        centroids.push((cy, cx));
        eccentricities.push(eccentricity_from_pixels(pixels));

        let mut comp_mask = Array2::<bool>::from_elem((h, w), false);
        for &(r, c) in pixels {
            comp_mask[[r, c]] = true;
        }
        solidities.push(solidity_from_mask(&comp_mask.view(), area));
    }

    features.insert("ccsm_mean_clump_area".to_string(), mean(&areas));
    features.insert(
        "ccsm_mean_clump_eccentricity".to_string(),
        mean(&eccentricities),
    );
    features.insert("ccsm_mean_clump_solidity".to_string(), mean(&solidities));

    // Mean distance to edge for condensed pixels, inside nucleus mask.
    let mut dists = Vec::<f64>::new();
    for ((r, c), &v) in condensed.indexed_iter() {
        if v {
            dists.push(dist_map[[r, c]]);
        }
    }
    features.insert("ccsm_mean_dist_to_edge".to_string(), mean(&dists));

    features.insert(
        "ccsm_mean_nnd".to_string(),
        mean_nearest_neighbor(&centroids),
    );

    // Texture on condensed chromatin image: enhanced_patch * condensed_mask
    let mut cc_img = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            if condensed[[y, x]] {
                cc_img[[y, x]] = enhanced[[y, x]];
            }
        }
    }

    let glcm = graycomatrix_0deg(&cc_img.view(), true, true);
    let (contrast, correlation, energy, homogeneity) = glcm_props(&glcm.view());
    features.insert("ccsm_texture_contrast".to_string(), contrast);
    features.insert("ccsm_texture_correlation".to_string(), correlation);
    features.insert("ccsm_texture_energy".to_string(), energy);
    features.insert("ccsm_texture_homogeneity".to_string(), homogeneity);

    for v in features.values_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    Ok(features)
}

fn default_ccsm_features() -> HashMap<String, f64> {
    [
        "ccsm_condensed_area_ratio",
        "ccsm_num_clumps",
        "ccsm_mean_clump_area",
        "ccsm_mean_clump_eccentricity",
        "ccsm_mean_clump_solidity",
        "ccsm_mean_dist_to_edge",
        "ccsm_mean_nnd",
        "ccsm_texture_contrast",
        "ccsm_texture_correlation",
        "ccsm_texture_energy",
        "ccsm_texture_homogeneity",
    ]
    .iter()
    .map(|&k| (k.to_string(), 0.0))
    .collect()
}

fn grayscale_to_u8(grayscale: &ArrayView2<f32>) -> Array2<u8> {
    let (h, w) = grayscale.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = grayscale[[y, x]].clamp(0.0, 255.0).round() as u8;
        }
    }
    out
}

fn bool_to_gray(mask: &ArrayView2<bool>) -> GrayImage {
    let (h, w) = mask.dim();
    let mut img = GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(
                x as u32,
                y as u32,
                Luma([if mask[[y, x]] { 255_u8 } else { 0_u8 }]),
            );
        }
    }
    img
}

fn component_pixels(mask: &ArrayView2<bool>) -> HashMap<u32, Vec<(usize, usize)>> {
    let gray = bool_to_gray(mask);
    let labels = connected_components(&gray, Connectivity::Four, Luma([0_u8]));

    let mut components: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for (x, y, pixel) in labels.enumerate_pixels() {
        let label = pixel[0];
        if label > 0 {
            components
                .entry(label)
                .or_default()
                .push((y as usize, x as usize));
        }
    }
    components
}

fn centroid_from_pixels(pixels: &[(usize, usize)]) -> (f64, f64) {
    if pixels.is_empty() {
        return (0.0, 0.0);
    }
    let n = pixels.len() as f64;
    let sum_r = pixels.iter().map(|&(r, _)| r as f64).sum::<f64>();
    let sum_c = pixels.iter().map(|&(_, c)| c as f64).sum::<f64>();
    (sum_r / n, sum_c / n)
}

fn eccentricity_from_pixels(pixels: &[(usize, usize)]) -> f64 {
    if pixels.len() < 2 {
        return 0.0;
    }
    let (cy, cx) = centroid_from_pixels(pixels);
    let n = pixels.len() as f64;

    let mut mu20 = 0.0;
    let mut mu02 = 0.0;
    let mut mu11 = 0.0;
    for &(r, c) in pixels {
        let dy = r as f64 - cy;
        let dx = c as f64 - cx;
        mu20 += dx * dx;
        mu02 += dy * dy;
        mu11 += dx * dy;
    }
    mu20 /= n;
    mu02 /= n;
    mu11 /= n;

    let trace = mu20 + mu02;
    let disc = ((mu20 - mu02) * (mu20 - mu02) + 4.0 * mu11 * mu11).sqrt();
    let l1 = ((trace + disc) * 0.5).max(0.0);
    let l2 = ((trace - disc) * 0.5).max(0.0);

    if l1 <= 1e-12 {
        0.0
    } else {
        (1.0 - (l2 / l1).clamp(0.0, 1.0)).sqrt()
    }
}

fn solidity_from_mask(mask: &ArrayView2<bool>, area: f64) -> f64 {
    if area <= 0.0 {
        return 0.0;
    }
    let Some(contour) = largest_external_contour(mask) else {
        return 0.0;
    };
    let hull = convex_hull(contour);
    if hull.len() < 3 {
        return 0.0;
    }
    let convex_area = convex_hull_pixel_area(&hull, mask.dim());
    safe_div(area, convex_area)
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

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn safe_div(num: f64, den: f64) -> f64 {
    if den.abs() <= 1e-12 || !num.is_finite() || !den.is_finite() {
        0.0
    } else {
        num / den
    }
}

fn mean_nearest_neighbor(points: &[(f64, f64)]) -> f64 {
    if points.len() <= 1 {
        return 0.0;
    }
    let mut mins = Vec::with_capacity(points.len());
    for (i, &(r1, c1)) in points.iter().enumerate() {
        let mut best = f64::INFINITY;
        for (j, &(r2, c2)) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = ((r1 - r2).powi(2) + (c1 - c2).powi(2)).sqrt();
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            mins.push(best);
        }
    }
    mean(&mins)
}

fn graycomatrix_0deg(image: &ArrayView2<u8>, symmetric: bool, normed: bool) -> Array2<f64> {
    const LEVELS: usize = 256;
    let (h, w) = image.dim();
    let mut p = Array2::<f64>::zeros((LEVELS, LEVELS));

    if w < 2 || h == 0 {
        return p;
    }

    for y in 0..h {
        for x in 0..(w - 1) {
            let i = image[[y, x]] as usize;
            let j = image[[y, x + 1]] as usize;
            p[[i, j]] += 1.0;
            if symmetric && i != j {
                p[[j, i]] += 1.0;
            }
        }
    }

    if normed {
        let sum = p.sum();
        if sum > 0.0 {
            p.mapv_inplace(|v| v / sum);
        }
    }
    p
}

fn glcm_props(p: &ArrayView2<f64>) -> (f64, f64, f64, f64) {
    let levels = p.nrows();
    let mut contrast = 0.0;
    let mut homogeneity = 0.0;
    let mut asm = 0.0;

    let mut mu_i = 0.0;
    let mut mu_j = 0.0;
    for i in 0..levels {
        for j in 0..levels {
            let v = p[[i, j]];
            mu_i += i as f64 * v;
            mu_j += j as f64 * v;
        }
    }

    let mut var_i = 0.0;
    let mut var_j = 0.0;
    let mut corr_num = 0.0;
    for i in 0..levels {
        for j in 0..levels {
            let v = p[[i, j]];
            let d = i as f64 - j as f64;
            contrast += v * d * d;
            homogeneity += v / (1.0 + d * d);
            asm += v * v;

            var_i += v * (i as f64 - mu_i).powi(2);
            var_j += v * (j as f64 - mu_j).powi(2);
            corr_num += v * (i as f64 - mu_i) * (j as f64 - mu_j);
        }
    }

    let correlation = if var_i <= 1e-12 || var_j <= 1e-12 {
        0.0
    } else {
        corr_num / (var_i.sqrt() * var_j.sqrt())
    };

    let energy = asm.sqrt();
    (contrast, correlation, energy, homogeneity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccsm_empty_mask_returns_defaults() {
        let patch = Array2::<f32>::from_elem((16, 16), 120.0);
        let mask = Array2::<bool>::from_elem((16, 16), false);
        let f = calculate_ccsm_features(&patch.view(), &mask.view()).unwrap();
        assert_eq!(f.len(), 11);
        assert!(f.values().all(|v| *v == 0.0));
    }

    #[test]
    fn test_ccsm_invalid_shape_error() {
        let patch = Array2::<f32>::from_elem((8, 8), 90.0);
        let mask = Array2::<bool>::from_elem((7, 8), true);
        let err = calculate_ccsm_features(&patch.view(), &mask.view()).unwrap_err();
        assert!(matches!(err, FeaturizerError::InvalidDimensions { .. }));
    }

    #[test]
    fn test_ccsm_features_finite_on_synthetic() {
        let mut patch = Array2::<f32>::from_elem((32, 32), 30.0);
        let mut mask = Array2::<bool>::from_elem((32, 32), false);

        for y in 6..26 {
            for x in 6..26 {
                mask[[y, x]] = true;
                patch[[y, x]] = if x < 16 { 65.0 } else { 190.0 };
            }
        }

        let f = calculate_ccsm_features(&patch.view(), &mask.view()).unwrap();
        assert_eq!(f.len(), 11);
        assert!(f.values().all(|v| v.is_finite()));
        assert!((0.0..=1.0).contains(&f["ccsm_condensed_area_ratio"]));
    }

    #[test]
    fn test_ccsm_gpu_flag_small_patch_matches_cpu() {
        let mut patch = Array2::<f32>::from_elem((24, 24), 40.0);
        let mut mask = Array2::<bool>::from_elem((24, 24), false);
        for y in 6..18 {
            for x in 6..18 {
                mask[[y, x]] = true;
                patch[[y, x]] = if x < 12 { 80.0 } else { 170.0 };
            }
        }

        let cpu = calculate_ccsm_features_with_gpu(&patch.view(), &mask.view(), false).unwrap();
        let gpu_flag = calculate_ccsm_features_with_gpu(&patch.view(), &mask.view(), true).unwrap();
        assert_eq!(cpu, gpu_flag);
    }

    #[test]
    fn test_ccsm_batch_matches_individual_calls() {
        let mut patch1 = Array2::<f32>::from_elem((24, 24), 35.0);
        let mut patch2 = Array2::<f32>::from_elem((24, 24), 45.0);
        let mut mask1 = Array2::<bool>::from_elem((24, 24), false);
        let mut mask2 = Array2::<bool>::from_elem((24, 24), false);

        for y in 6..18 {
            for x in 6..18 {
                mask1[[y, x]] = true;
                patch1[[y, x]] = if x < 12 { 75.0 } else { 165.0 };
            }
        }
        for y in 5..19 {
            for x in 7..17 {
                mask2[[y, x]] = true;
                patch2[[y, x]] = if y < 12 { 90.0 } else { 155.0 };
            }
        }

        let patches = vec![patch1.clone(), patch2.clone()];
        let masks = vec![mask1.clone(), mask2.clone()];

        let batch = calculate_ccsm_features_batch_with_gpu(&patches, &masks, true).unwrap();
        let single0 =
            calculate_ccsm_features_with_gpu(&patch1.view(), &mask1.view(), true).unwrap();
        let single1 =
            calculate_ccsm_features_with_gpu(&patch2.view(), &mask2.view(), true).unwrap();

        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], single0);
        assert_eq!(batch[1], single1);
    }
}
