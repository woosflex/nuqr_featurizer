// Euclidean Distance Transform (EDT) compute shader for WGPU.
//
// Input:
// - mask: 1 for foreground (inside nucleus), 0 for background
//
// Output:
// - distances: Euclidean distance to nearest background pixel for each foreground pixel
//   and 0 for background pixels.

struct EdtParams {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: EdtParams;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn edt_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    if (mask[idx] == 0u) {
        distances[idx] = 0.0;
        return;
    }

    var min_dist2: f32 = 3.402823e38;
    var found_background: bool = false;

    for (var yy: u32 = 0u; yy < params.height; yy = yy + 1u) {
        for (var xx: u32 = 0u; xx < params.width; xx = xx + 1u) {
            let nidx = yy * params.width + xx;
            if (mask[nidx] == 0u) {
                found_background = true;
                let dx = f32(i32(xx) - i32(x));
                let dy = f32(i32(yy) - i32(y));
                let d2 = dx * dx + dy * dy;
                if (d2 < min_dist2) {
                    min_dist2 = d2;
                }
            }
        }
    }

    if (found_background) {
        distances[idx] = sqrt(min_dist2);
    } else {
        distances[idx] = 0.0;
    }
}
