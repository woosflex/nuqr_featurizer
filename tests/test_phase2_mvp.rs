use ndarray::Array2;
use nuxplore::features::extract_morphology_batch;
use nuxplore::FeaturizerError;

fn feature<'a>(map: &'a std::collections::HashMap<String, f64>, key: &str) -> &'a f64 {
    map.get(key)
        .unwrap_or_else(|| panic!("missing feature key: {key}"))
}

#[test]
fn test_extract_morphology_batch_single_square() {
    let image_shape = (16, 16, 3);
    let mut mask = Array2::from_elem((16, 16), false);
    for r in 5..8 {
        for c in 5..8 {
            mask[[r, c]] = true;
        }
    }

    let out = extract_morphology_batch(image_shape, &[mask], false).unwrap();
    assert_eq!(out.len(), 1);

    let f = &out[0];
    assert_eq!(*feature(f, "area"), 9.0);
    assert!((*feature(f, "centroid_row") - 6.0).abs() < 1e-12);
    assert!((*feature(f, "centroid_col") - 6.0).abs() < 1e-12);
    assert!((*feature(f, "perimeter") - 8.0).abs() < 1e-12);
}

#[test]
fn test_extract_morphology_batch_dimension_mismatch() {
    let image_shape = (20, 20, 3);
    let mask = Array2::from_elem((8, 8), true);

    let err = extract_morphology_batch(image_shape, &[mask], false).unwrap_err();
    assert!(matches!(err, FeaturizerError::InvalidDimensions { .. }));
}

#[test]
fn test_extract_morphology_batch_empty_mask_error() {
    let image_shape = (10, 10, 3);
    let empty = Array2::from_elem((10, 10), false);

    let err = extract_morphology_batch(image_shape, &[empty], false).unwrap_err();
    assert!(matches!(
        err,
        FeaturizerError::FeatureComputationFailed { .. }
    ));
}
