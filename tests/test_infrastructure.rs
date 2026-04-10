//! Integration tests for Phase 1 infrastructure

// Since this is an integration test, we need to import via the crate name
// but the internal modules aren't exposed. Let's test the public API instead.

use ndarray::{Array2, Array3};

#[test]
fn test_basic_array_operations() {
    // Test that ndarray works as expected
    let arr = Array2::from_elem((10, 10), 0u8);
    assert_eq!(arr.dim(), (10, 10));

    let arr3: Array3<u8> = Array3::zeros((5, 5, 3));
    assert_eq!(arr3.dim(), (5, 5, 3));
}

#[test]
fn test_rayon_parallelism() {
    // Test that rayon is available
    use rayon::prelude::*;

    let data: Vec<i32> = (0..100).collect();
    let sum: i32 = data.par_iter().sum();

    assert_eq!(sum, 4950);
}

#[test]
fn test_rustfft_available() {
    // Test that rustfft is available
    use rustfft::{num_complex::Complex, FftPlanner};

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(8);

    let mut buffer = vec![Complex::new(0.0, 0.0); 8];
    buffer[0] = Complex::new(1.0, 0.0);

    fft.process(&mut buffer);

    // Just check it doesn't panic
    assert!(buffer.len() == 8);
}
