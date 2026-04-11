// GLCM (Gray-Level Co-occurrence Matrix) compute shader for WGPU
//
// This shader accumulates symmetric, distance=1 co-occurrence counts for 4 angles:
// - 0°   => (dr, dc) = ( 0,  1)
// - 45°  => (dr, dc) = ( 1,  1)
// - 90°  => (dr, dc) = ( 1,  0)
// - 135° => (dr, dc) = ( 1, -1)

struct GlcmParams {
    width: u32,
    height: u32,
    symmetric: u32,
    _pad: u32,
}

const LEVELS: u32 = 256u;

@group(0) @binding(0) var<uniform> params: GlcmParams;
@group(0) @binding(1) var<storage, read> image: array<u32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> glcm_counts: array<atomic<u32>>;

fn accumulate_pair(x: u32, y: u32, i: u32, angle_idx: u32, dc: i32, dr: i32) {
    let nx = i32(x) + dc;
    let ny = i32(y) + dr;
    if (nx < 0 || ny < 0 || nx >= i32(params.width) || ny >= i32(params.height)) {
        return;
    }

    let nidx = u32(ny) * params.width + u32(nx);
    if (mask[nidx] == 0u) {
        return;
    }

    let j = image[nidx] & 255u;
    let plane_base = angle_idx * LEVELS * LEVELS;
    let ij = plane_base + i * LEVELS + j;
    atomicAdd(&glcm_counts[ij], 1u);

    if (params.symmetric != 0u) {
        let ji = plane_base + j * LEVELS + i;
        atomicAdd(&glcm_counts[ji], 1u);
    }
}

@compute @workgroup_size(16, 16, 1)
fn glcm_accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    if (mask[idx] == 0u) {
        return;
    }
    let i = image[idx] & 255u;

    // 0°, 45°, 90°, 135°
    accumulate_pair(x, y, i, 0u, 1, 0);
    accumulate_pair(x, y, i, 1u, 1, 1);
    accumulate_pair(x, y, i, 2u, 0, 1);
    accumulate_pair(x, y, i, 3u, -1, 1);
}
