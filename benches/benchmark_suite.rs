use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{s, Array2, Array3};
use std::sync::Once;
use std::time::Duration;

use nuxplore::features::ccsm::calculate_ccsm_features_with_gpu;
use nuxplore::features::ccsm_clahe::clahe_u8_with_gpu;
use nuxplore::features::extract_morphology_batch;
use nuxplore::features::glcm::calculate_glcm_features_with_gpu;
use nuxplore::features::he_color::calculate_he_color_features;
use nuxplore::features::hog::calculate_hog_features;
use nuxplore::features::intensity::calculate_intensity_features;
use nuxplore::features::lbp::calculate_lbp_features;
use nuxplore::features::neis::calculate_neis_features;
use nuxplore::features::shape::calculate_advanced_shape_features;
use nuxplore::features::spatial::calculate_nearest_neighbor_distance;
use nuxplore::{init_logging, is_gpu_available};

static INIT_LOGGING: Once = Once::new();

fn init_bench_logging() {
    INIT_LOGGING.call_once(init_logging);
}

fn make_grayscale_patch(side: usize, seed: usize) -> Array2<f32> {
    let mut patch = Array2::<f32>::zeros((side, side));
    for r in 0..side {
        for c in 0..side {
            let base = ((r * 13 + c * 7 + seed * 17 + ((r ^ c) * (seed % 11 + 1))) % 256) as f32;
            let wave = (((r * 3 + c * 5 + seed * 29) % 37) as f32) * 0.75;
            patch[[r, c]] = (base * 0.8 + wave).clamp(0.0, 255.0);
        }
    }
    patch
}

fn make_u8_patch(side: usize, seed: usize) -> Array2<u8> {
    let mut patch = Array2::<u8>::zeros((side, side));
    for r in 0..side {
        for c in 0..side {
            patch[[r, c]] = ((r * 11 + c * 5 + seed * 19 + (r ^ c)) % 256) as u8;
        }
    }
    patch
}

fn make_rgb_patch(side: usize, seed: usize) -> Array3<u8> {
    let mut rgb = Array3::<u8>::zeros((side, side, 3));
    for r in 0..side {
        for c in 0..side {
            rgb[[r, c, 0]] = ((r * 3 + c * 5 + seed * 7) % 256) as u8;
            rgb[[r, c, 1]] = ((r * 11 + c * 13 + seed * 17) % 256) as u8;
            rgb[[r, c, 2]] = ((r * 19 + c * 23 + seed * 29) % 256) as u8;
        }
    }
    rgb
}

fn make_circular_mask(side: usize, seed: usize) -> Array2<bool> {
    let mut mask = Array2::<bool>::from_elem((side, side), false);
    let cy = side as isize / 2 + (seed as isize % 3) - 1;
    let cx = side as isize / 2 + (seed as isize % 5) - 2;
    let radius = ((side as f64) * (0.30 + ((seed % 7) as f64) * 0.01)) as isize;
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

fn make_masks_batch(mask_count: usize, side: usize) -> Vec<Array2<bool>> {
    (0..mask_count)
        .map(|i| make_circular_mask(side, i))
        .collect::<Vec<_>>()
}

fn make_centroids(count: usize) -> Array2<f64> {
    let mut centroids = Array2::<f64>::zeros((count, 2));
    for i in 0..count {
        centroids[[i, 0]] = ((i * 13) % 997) as f64 * 0.25 + (i % 7) as f64;
        centroids[[i, 1]] = ((i * 17) % 991) as f64 * 0.25 + (i % 11) as f64;
    }
    centroids
}

fn bench_texture_and_intensity(c: &mut Criterion) {
    init_bench_logging();
    let gpu = is_gpu_available();
    let mut group = c.benchmark_group("suite_texture_intensity");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    for side in [64_usize, 128, 192] {
        let grayscale = make_grayscale_patch(side, side);
        let mask = make_circular_mask(side, side + 1);
        let u8_patch = make_u8_patch(side, side + 2);

        group.bench_with_input(BenchmarkId::new("intensity_cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_intensity_features(black_box(&grayscale.view()), black_box(&mask.view()))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("lbp_cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_lbp_features(black_box(&grayscale.view()), black_box(&mask.view()))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("hog_cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_hog_features(black_box(&grayscale.view()), black_box(&mask.view()), false)
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("glcm_cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_glcm_features_with_gpu(
                    black_box(&grayscale.view()),
                    black_box(&mask.view()),
                    false,
                )
                .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("clahe_cpu", side), &side, |b, _| {
            b.iter(|| clahe_u8_with_gpu(black_box(&u8_patch.view()), 0.03, 16, false).unwrap())
        });

        if gpu {
            let _ = calculate_hog_features(&grayscale.view(), &mask.view(), true);
            let _ = calculate_glcm_features_with_gpu(&grayscale.view(), &mask.view(), true);
            let _ = clahe_u8_with_gpu(&u8_patch.view(), 0.03, 16, true);

            group.bench_with_input(BenchmarkId::new("hog_gpu", side), &side, |b, _| {
                b.iter(|| {
                    calculate_hog_features(
                        black_box(&grayscale.view()),
                        black_box(&mask.view()),
                        true,
                    )
                    .unwrap()
                })
            });
            group.bench_with_input(BenchmarkId::new("glcm_gpu", side), &side, |b, _| {
                b.iter(|| {
                    calculate_glcm_features_with_gpu(
                        black_box(&grayscale.view()),
                        black_box(&mask.view()),
                        true,
                    )
                    .unwrap()
                })
            });
            group.bench_with_input(BenchmarkId::new("clahe_gpu", side), &side, |b, _| {
                b.iter(|| clahe_u8_with_gpu(black_box(&u8_patch.view()), 0.03, 16, true).unwrap())
            });
        }
    }

    group.finish();
}

fn bench_shape_color_neis(c: &mut Criterion) {
    init_bench_logging();
    let mut group = c.benchmark_group("suite_shape_color_neis");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(3));

    for side in [96_usize, 160] {
        let rgb = make_rgb_patch(side, side);
        let mask = make_circular_mask(side, side + 3);
        let grayscale = make_grayscale_patch(side, side + 4);

        group.bench_with_input(BenchmarkId::new("he_color_cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_he_color_features(black_box(&rgb.view()), black_box(&mask.view()))
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("shape_cpu", side), &side, |b, _| {
            b.iter(|| calculate_advanced_shape_features(black_box(&mask.view())).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("neis_cpu", side), &side, |b, _| {
            b.iter(|| calculate_neis_features(black_box(&mask.view())).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("ccsm_cpu", side), &side, |b, _| {
            b.iter(|| {
                calculate_ccsm_features_with_gpu(
                    black_box(&grayscale.view()),
                    black_box(&mask.view()),
                    false,
                )
                .unwrap()
            })
        });

        if is_gpu_available() {
            let _ = calculate_ccsm_features_with_gpu(&grayscale.view(), &mask.view(), true);
            group.bench_with_input(BenchmarkId::new("ccsm_gpu", side), &side, |b, _| {
                b.iter(|| {
                    calculate_ccsm_features_with_gpu(
                        black_box(&grayscale.view()),
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

fn bench_morphology_batch(c: &mut Criterion) {
    init_bench_logging();
    let mut group = c.benchmark_group("suite_morphology_batch");
    let side = 128usize;
    let image_shape = (side, side, 3);
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(3));

    for mask_count in [32_usize, 128, 512] {
        let masks = make_masks_batch(mask_count, side);
        group.throughput(Throughput::Elements(mask_count as u64));
        group.bench_with_input(
            BenchmarkId::new("morph_batch_cpu", mask_count),
            &mask_count,
            |b, _| {
                b.iter(|| extract_morphology_batch(image_shape, black_box(&masks), false).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_spatial(c: &mut Criterion) {
    init_bench_logging();
    let mut group = c.benchmark_group("suite_spatial");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));

    for count in [32_usize, 256, 2048, 8192] {
        let all = make_centroids(count + 1);
        let center = (all[[0, 0]], all[[0, 1]]);
        let others = all.slice(s![1.., ..]).to_owned();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("nearest_neighbor_cpu", count),
            &count,
            |b, _| {
                b.iter(|| {
                    calculate_nearest_neighbor_distance(center, black_box(&others.view())).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_texture_and_intensity,
    bench_shape_color_neis,
    bench_morphology_batch,
    bench_spatial
);
criterion_main!(benches);
