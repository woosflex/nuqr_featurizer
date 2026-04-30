use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use std::sync::Once;

use nuxplore::features::ccsm_clahe::clahe_u8_with_gpu;
use nuxplore::features::glcm::calculate_glcm_features_with_gpu;
use nuxplore::features::hog::calculate_hog_features;
use nuxplore::{init_logging, is_gpu_available};

static INIT_LOGGING: Once = Once::new();

fn init_bench_logging() {
    INIT_LOGGING.call_once(init_logging);
}

fn make_grayscale_patch(side: usize) -> Array2<f32> {
    let mut patch = Array2::<f32>::zeros((side, side));
    for r in 0..side {
        for c in 0..side {
            let base = ((r * 13 + c * 7) % 256) as f32;
            patch[[r, c]] = (base + ((r ^ c) % 17) as f32).clamp(0.0, 255.0);
        }
    }
    patch
}

fn make_u8_patch(side: usize) -> Array2<u8> {
    let mut patch = Array2::<u8>::zeros((side, side));
    for r in 0..side {
        for c in 0..side {
            patch[[r, c]] = ((r * 11 + c * 5 + (r ^ c)) % 256) as u8;
        }
    }
    patch
}

fn make_circular_mask(side: usize) -> Array2<bool> {
    let mut mask = Array2::<bool>::from_elem((side, side), false);
    let cy = (side as isize) / 2;
    let cx = (side as isize) / 2;
    let radius = ((side as f64) * 0.35) as isize;
    let r2 = radius * radius;
    for r in 0..side as isize {
        for c in 0..side as isize {
            let dy = r - cy;
            let dx = c - cx;
            if dy * dy + dx * dx <= r2 {
                mask[[r as usize, c as usize]] = true;
            }
        }
    }
    mask
}

fn bench_hog_cpu_gpu(c: &mut Criterion) {
    init_bench_logging();
    let gpu = is_gpu_available();
    let mut group = c.benchmark_group("gpu_features_hog");

    for side in [64_usize, 128, 256] {
        let patch = make_grayscale_patch(side);
        let mask = make_circular_mask(side);

        group.bench_with_input(BenchmarkId::new("cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_hog_features(black_box(&patch.view()), black_box(&mask.view()), false)
                    .unwrap()
            })
        });

        if gpu {
            let _ = calculate_hog_features(&patch.view(), &mask.view(), true);
            group.bench_with_input(BenchmarkId::new("gpu", side), &side, |b, _| {
                b.iter(|| {
                    calculate_hog_features(black_box(&patch.view()), black_box(&mask.view()), true)
                        .unwrap()
                })
            });
        }
    }

    group.finish();
}

fn bench_glcm_cpu_gpu(c: &mut Criterion) {
    init_bench_logging();
    let gpu = is_gpu_available();
    let mut group = c.benchmark_group("gpu_features_glcm");

    for side in [64_usize, 128, 192] {
        let patch = make_grayscale_patch(side);
        let mask = make_circular_mask(side);

        group.bench_with_input(BenchmarkId::new("cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_glcm_features_with_gpu(
                    black_box(&patch.view()),
                    black_box(&mask.view()),
                    false,
                )
                .unwrap()
            })
        });

        if gpu {
            let _ = calculate_glcm_features_with_gpu(&patch.view(), &mask.view(), true);
            group.bench_with_input(BenchmarkId::new("gpu", side), &side, |b, _| {
                b.iter(|| {
                    calculate_glcm_features_with_gpu(
                        black_box(&patch.view()),
                        black_box(&mask.view()),
                        true,
                    )
                    .unwrap()
                })
            });
        }
    }

    group.finish();
}

fn bench_clahe_cpu_gpu(c: &mut Criterion) {
    init_bench_logging();
    let gpu = is_gpu_available();
    let mut group = c.benchmark_group("gpu_features_clahe");

    for side in [64_usize, 128, 256] {
        let image = make_u8_patch(side);

        group.bench_with_input(BenchmarkId::new("cpu", side), &side, |b, _| {
            b.iter(|| clahe_u8_with_gpu(black_box(&image.view()), 0.03, 16, false).unwrap())
        });

        if gpu {
            let _ = clahe_u8_with_gpu(&image.view(), 0.03, 16, true);
            group.bench_with_input(BenchmarkId::new("gpu", side), &side, |b, _| {
                b.iter(|| clahe_u8_with_gpu(black_box(&image.view()), 0.03, 16, true).unwrap())
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hog_cpu_gpu,
    bench_glcm_cpu_gpu,
    bench_clahe_cpu_gpu
);
criterion_main!(benches);
