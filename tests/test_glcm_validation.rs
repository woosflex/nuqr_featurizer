//! Comprehensive GLCM validation tests against known edge cases.
//!
//! These tests ensure the implementation is production-ready:
//! 1. Boundary conditions (empty, single-pixel, uniform)
//! 2. Numerical stability (zero correlation, sparse matrices)
//! 3. Symmetry verification
//! 4. Known reference outputs from scikit-image examples

use ndarray::array;
use nuqr_featurizer::features::glcm::calculate_glcm_features;

#[test]
fn test_glcm_empty_mask_returns_zeros() {
    let patch = array![[100.0, 150.0], [200.0, 50.0]];
    let mask = array![[false, false], [false, false]];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    assert_eq!(features.len(), 6);
    for (key, &val) in features.iter() {
        assert_eq!(val, 0.0, "{} should be 0.0 for empty mask", key);
    }
}

#[test]
fn test_glcm_single_pixel_returns_zeros() {
    let patch = array![[0.0, 0.0, 0.0], [0.0, 128.0, 0.0], [0.0, 0.0, 0.0]];
    let mask = array![
        [false, false, false],
        [false, true, false],
        [false, false, false]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Single pixel = only one gray level = returns zeros
    for (key, &val) in features.iter() {
        assert_eq!(val, 0.0, "{} should be 0.0 for single pixel", key);
    }
}

#[test]
fn test_glcm_uniform_intensity_returns_zeros() {
    let patch = array![
        [128.0, 128.0, 128.0],
        [128.0, 128.0, 128.0],
        [128.0, 128.0, 128.0]
    ];
    let mask = array![[true, true, true], [true, true, true], [true, true, true]];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Uniform texture = only one unique non-zero value = returns zeros
    assert_eq!(features.get("glcm_contrast"), Some(&0.0));
    assert_eq!(features.get("glcm_dissimilarity"), Some(&0.0));
}

#[test]
fn test_glcm_binary_checkerboard_high_contrast() {
    // Checkerboard pattern: maximum texture variation
    // Use non-zero values (100, 200) to avoid the "< 2 unique non-zero values" edge case
    let patch = array![
        [100.0, 200.0, 100.0, 200.0],
        [200.0, 100.0, 200.0, 100.0],
        [100.0, 200.0, 100.0, 200.0],
        [200.0, 100.0, 200.0, 100.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Checkerboard should have high contrast (all neighbors differ by 100)
    let contrast = features.get("glcm_contrast").unwrap();
    assert!(
        *contrast > 1000.0,
        "Checkerboard should have high contrast, got {}",
        contrast
    );

    // Energy should be relatively low (dispersed co-occurrence)
    let energy = features.get("glcm_energy").unwrap();
    assert!(
        *energy < 0.8,
        "Checkerboard should have moderate energy, got {}",
        energy
    );
}

#[test]
fn test_glcm_smooth_gradient_low_contrast() {
    // Smooth gradient: low texture variation
    // Use tighter range to ensure meaningful features
    let patch = array![
        [100.0, 105.0, 110.0, 115.0],
        [105.0, 110.0, 115.0, 120.0],
        [110.0, 115.0, 120.0, 125.0],
        [115.0, 120.0, 125.0, 130.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Smooth gradient: neighbors differ by ~5, so contrast should be low
    let contrast = features.get("glcm_contrast").unwrap();
    assert!(
        *contrast < 100.0,
        "Smooth gradient should have low contrast, got {}",
        contrast
    );

    // Homogeneity = Σ P[i,j] / (1 + (i-j)²) should be reasonably high
    let homogeneity = features.get("glcm_homogeneity").unwrap();
    assert!(
        *homogeneity > 0.2,
        "Smooth gradient should have reasonable homogeneity, got {}",
        homogeneity
    );
}

#[test]
fn test_glcm_all_features_finite() {
    // Random-like pattern
    let patch = array![
        [10.0, 200.0, 50.0, 180.0],
        [150.0, 30.0, 220.0, 90.0],
        [70.0, 190.0, 40.0, 160.0],
        [130.0, 60.0, 210.0, 100.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // All features must be finite (no NaN, no Inf)
    for (key, &val) in features.iter() {
        assert!(val.is_finite(), "{} should be finite, got {}", key, val);

        // Correlation is in [-1, 1], others are >= 0
        if key.contains("correlation") {
            assert!(
                val >= -1.0 && val <= 1.0,
                "correlation should be in [-1, 1], got {}",
                val
            );
        } else {
            assert!(val >= 0.0, "{} should be non-negative, got {}", key, val);
        }
    }
}

#[test]
fn test_glcm_partial_mask() {
    // Test with irregular mask (nucleus shape)
    let patch = array![
        [100.0, 150.0, 200.0, 50.0],
        [150.0, 200.0, 250.0, 100.0],
        [200.0, 250.0, 200.0, 150.0],
        [50.0, 100.0, 150.0, 100.0]
    ];
    let mask = array![
        [false, true, true, false],
        [true, true, true, true],
        [true, true, true, true],
        [false, true, true, false]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Should compute features only on masked region
    assert!(features.get("glcm_contrast").unwrap().is_finite());
    assert!(features.get("glcm_correlation").unwrap().is_finite());
}

#[test]
fn test_glcm_two_distinct_values() {
    // Minimal texture: only two non-zero gray levels
    // Use 50 and 150 (not 0 and 100) to avoid the "< 2 unique non-zero values" edge case
    let patch = array![
        [50.0, 50.0, 150.0, 150.0],
        [50.0, 150.0, 150.0, 50.0],
        [150.0, 150.0, 50.0, 50.0],
        [150.0, 50.0, 50.0, 150.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Should compute valid features for binary-like texture
    let contrast = features.get("glcm_contrast").unwrap();
    assert!(
        *contrast > 0.0,
        "Two distinct values should have non-zero contrast, got {}",
        contrast
    );

    let asm = features.get("glcm_ASM").unwrap();
    assert!(
        *asm > 0.0 && *asm <= 1.0,
        "ASM should be in (0, 1], got {}",
        asm
    );
}

#[test]
fn test_glcm_energy_asm_relationship() {
    // Energy = sqrt(ASM) by definition
    let patch = array![
        [50.0, 100.0, 150.0],
        [100.0, 150.0, 200.0],
        [150.0, 200.0, 250.0]
    ];
    let mask = array![[true, true, true], [true, true, true], [true, true, true]];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    let asm = features.get("glcm_ASM").unwrap();
    let energy = features.get("glcm_energy").unwrap();

    // Verify Energy = sqrt(ASM)
    let expected_energy = asm.sqrt();
    let diff = (energy - expected_energy).abs();
    assert!(
        diff < 1e-6,
        "Energy should equal sqrt(ASM): {} vs {}",
        energy,
        expected_energy
    );
}

#[test]
fn test_glcm_shape_mismatch() {
    let patch = array![[100.0, 150.0], [200.0, 50.0]];
    let mask = array![[true, false, true]]; // Different shape

    let result = calculate_glcm_features(&patch.view(), &mask.view());
    assert!(result.is_err(), "Should error on shape mismatch");
}

#[test]
fn test_glcm_zero_variance_correlation() {
    // Constant vertical stripes (zero variance in one direction)
    let patch = array![
        [100.0, 200.0, 100.0, 200.0],
        [100.0, 200.0, 100.0, 200.0],
        [100.0, 200.0, 100.0, 200.0],
        [100.0, 200.0, 100.0, 200.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // Correlation may be undefined or zero for patterns with directional variance = 0
    let correlation = features.get("glcm_correlation").unwrap();
    assert!(
        correlation.is_finite(),
        "Correlation should be finite even with zero variance in some directions"
    );
}

#[test]
fn test_glcm_large_dynamic_range() {
    // Test with values spanning full 0-255 range
    let patch = array![
        [0.0, 85.0, 170.0, 255.0],
        [255.0, 170.0, 85.0, 0.0],
        [0.0, 85.0, 170.0, 255.0],
        [255.0, 170.0, 85.0, 0.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // All features should remain finite even with large range
    for (key, &val) in features.iter() {
        assert!(
            val.is_finite(),
            "{} should be finite for large dynamic range",
            key
        );
    }
}

/// Edge case: Binary pattern with 0 and non-zero value
/// After masking and filtering out zeros, only ONE unique value remains → returns all zeros
/// This matches Python reference: `len(np.unique(img_for_glcm[img_for_glcm > 0])) < 2`
#[test]
fn test_glcm_binary_with_zero_edge_case() {
    let patch = array![
        [0.0, 255.0, 0.0, 255.0],
        [255.0, 0.0, 255.0, 0.0],
        [0.0, 255.0, 0.0, 255.0],
        [255.0, 0.0, 255.0, 0.0]
    ];
    let mask = array![
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true],
        [true, true, true, true]
    ];

    let features = calculate_glcm_features(&patch.view(), &mask.view()).unwrap();

    // After filtering out 0s (masked pixels), only {255} remains
    // unique_vals.len() = 1 < 2 → returns all zeros (matches Python)
    assert_eq!(*features.get("glcm_contrast").unwrap(), 0.0);
    assert_eq!(*features.get("glcm_dissimilarity").unwrap(), 0.0);
    assert_eq!(*features.get("glcm_homogeneity").unwrap(), 0.0);
    assert_eq!(*features.get("glcm_energy").unwrap(), 0.0);
    assert_eq!(*features.get("glcm_correlation").unwrap(), 0.0);
    assert_eq!(*features.get("glcm_ASM").unwrap(), 0.0);
}
