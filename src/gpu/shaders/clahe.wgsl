// CLAHE (Contrast Limited Adaptive Histogram Equalization) shader.
//
// This implements per-pixel bilinear interpolation between four neighboring
// tile mappings, with clipping + redistribution logic aligned to the CPU path.

struct ClaheParams {
    width: u32,
    height: u32,
    kernel_size: u32,
    n_tiles_x: u32,
    n_tiles_y: u32,
    clip_limit: f32,
    _pad0: u32,
    _pad1: u32,
}

struct InterpIndices {
    i0: u32,
    i1: u32,
    w: f32,
}

const N_BINS: u32 = 256u;

@group(0) @binding(0) var<uniform> params: ClaheParams;
@group(0) @binding(1) var<storage, read> image: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn interpolation_indices(coord: u32, n_tiles: u32, tile_size: u32) -> InterpIndices {
    var g = f32(coord) / f32(tile_size) - 0.5;
    var i0 = i32(floor(g));
    var i1 = i0 + 1;
    var w = g - f32(i0);

    if (i0 < 0) {
        i0 = 0;
        i1 = 0;
        w = 0.0;
    }
    if (i1 >= i32(n_tiles)) {
        i1 = i32(n_tiles) - 1;
        i0 = i1;
        w = 0.0;
    }

    return InterpIndices(u32(i0), u32(i1), clamp(w, 0.0, 1.0));
}

fn clip_histogram(hist: ptr<function, array<u32, 256>>, clip_limit: u32) {
    if (clip_limit == 0u) {
        return;
    }

    var n_excess: i32 = 0;
    for (var i: u32 = 0u; i < N_BINS; i = i + 1u) {
        let value = (*hist)[i];
        if (value > clip_limit) {
            n_excess = n_excess + i32(value - clip_limit);
            (*hist)[i] = clip_limit;
        }
    }

    if (n_excess <= 0) {
        return;
    }

    let bin_incr: u32 = u32(n_excess) / N_BINS;
    var upper: u32 = 0u;
    if (clip_limit > bin_incr) {
        upper = clip_limit - bin_incr;
    }

    if (bin_incr > 0u) {
        for (var i: u32 = 0u; i < N_BINS; i = i + 1u) {
            let value = (*hist)[i];
            if (value < upper) {
                (*hist)[i] = value + bin_incr;
                n_excess = n_excess - i32(bin_incr);
            } else if (value >= upper && value < clip_limit) {
                n_excess = n_excess + i32(value) - i32(clip_limit);
                (*hist)[i] = clip_limit;
            }
        }
    }

    loop {
        if (n_excess <= 0) {
            break;
        }

        let prev = n_excess;
        var n_under: u32 = 0u;
        for (var i: u32 = 0u; i < N_BINS; i = i + 1u) {
            if ((*hist)[i] < clip_limit) {
                n_under = n_under + 1u;
            }
        }
        if (n_under == 0u) {
            break;
        }

        var step_size: u32 = n_under / u32(n_excess);
        if (step_size < 1u) {
            step_size = 1u;
        }

        var idx: u32 = 0u;
        loop {
            if (idx >= N_BINS || n_excess <= 0) {
                break;
            }
            if ((*hist)[idx] < clip_limit) {
                (*hist)[idx] = (*hist)[idx] + 1u;
                n_excess = n_excess - 1;
            }
            idx = idx + step_size;
        }

        if (prev == n_excess) {
            break;
        }
    }
}

fn tile_map_value(ty: u32, tx: u32, pixel_bin: u32) -> f32 {
    let y0 = ty * params.kernel_size;
    let x0 = tx * params.kernel_size;
    let y1 = min((ty + 1u) * params.kernel_size, params.height);
    let x1 = min((tx + 1u) * params.kernel_size, params.width);

    let tile_h = y1 - y0;
    let tile_w = x1 - x0;
    let tile_pixels = tile_h * tile_w;
    if (tile_pixels == 0u) {
        return f32(pixel_bin);
    }

    var clip: u32 = tile_pixels;
    if (params.clip_limit > 0.0) {
        let clipped = u32(params.clip_limit * f32(tile_pixels));
        clip = max(clipped, 1u);
    }

    var hist: array<u32, 256>;
    for (var i: u32 = 0u; i < N_BINS; i = i + 1u) {
        hist[i] = 0u;
    }

    for (var y: u32 = y0; y < y1; y = y + 1u) {
        for (var x: u32 = x0; x < x1; x = x + 1u) {
            let idx = y * params.width + x;
            let b = image[idx] & 255u;
            hist[b] = hist[b] + 1u;
        }
    }

    clip_histogram(&hist, clip);

    var cumsum: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= N_BINS || i > pixel_bin) {
            break;
        }
        cumsum = cumsum + hist[i];
        i = i + 1u;
    }

    let mapped = floor((f32(cumsum) * 255.0) / f32(tile_pixels));
    return clamp(mapped, 0.0, 255.0);
}

@compute @workgroup_size(8, 8, 1)
fn clahe_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let p = image[idx] & 255u;

    let yi = interpolation_indices(y, params.n_tiles_y, params.kernel_size);
    let xi = interpolation_indices(x, params.n_tiles_x, params.kernel_size);

    let v00 = tile_map_value(yi.i0, xi.i0, p);
    let v01 = tile_map_value(yi.i0, xi.i1, p);
    let v10 = tile_map_value(yi.i1, xi.i0, p);
    let v11 = tile_map_value(yi.i1, xi.i1, p);

    let top = (1.0 - xi.w) * v00 + xi.w * v01;
    let bottom = (1.0 - xi.w) * v10 + xi.w * v11;
    let value = clamp(round((1.0 - yi.w) * top + yi.w * bottom), 0.0, 255.0);
    output[idx] = u32(value);
}
