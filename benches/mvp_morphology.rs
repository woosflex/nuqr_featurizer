use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use nuqr_featurizer::features::extract_morphology_batch;

fn generate_masks(mask_count: usize, side: usize) -> Vec<Array2<bool>> {
    let mut masks = Vec::with_capacity(mask_count);
    for i in 0..mask_count {
        let mut mask = Array2::from_elem((side, side), false);
        let offset = (i * 7) % (side.saturating_sub(12).max(1));
        for r in (offset + 2)..(offset + 10).min(side) {
            for c in (offset + 2)..(offset + 10).min(side) {
                mask[[r, c]] = true;
            }
        }
        masks.push(mask);
    }
    masks
}

fn bench_phase2_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase2_morphology");
    let image_shape = (64, 64, 3);

    for mask_count in [32_usize, 128, 512] {
        let masks = generate_masks(mask_count, 64);
        group.bench_with_input(
            BenchmarkId::new("rust_rayon", mask_count),
            &mask_count,
            |b, _| {
                b.iter(|| extract_morphology_batch(image_shape, black_box(&masks), false).unwrap())
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_phase2_morphology);
criterion_main!(benches);
