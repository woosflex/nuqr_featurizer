// HOG (Histogram of Oriented Gradients) compute shaders for WGPU
// 
// This file contains 3 compute shaders:
// 1. gradient_compute: Compute gradient magnitude and orientation for each pixel
// 2. histogram_accumulate: Bin gradients into 8-orientation histograms per cell
// 3. normalize_l2hys: L2-Hys normalization of cell histograms
//
// Parameters:
// - orientations = 8 (unsigned 180°)
// - pixels_per_cell = (8, 8)
// - cells_per_block = (1, 1)

// ============================================================================
// Shader 1: Gradient Computation
// ============================================================================

struct GradientParams {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: GradientParams;
@group(0) @binding(1) var<storage, read> image: array<f32>;      // Input: masked grayscale image
@group(0) @binding(2) var<storage, read_write> gradients: array<vec2<f32>>; // Output: (magnitude, orientation)

@compute @workgroup_size(16, 16)
fn gradient_compute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Central difference gradients
    var g_row: f32 = 0.0;
    var g_col: f32 = 0.0;
    
    // g_row: vertical gradient (row direction)
    if (y > 0u && y < params.height - 1u) {
        let idx_up = (y - 1u) * params.width + x;
        let idx_down = (y + 1u) * params.width + x;
        g_row = image[idx_down] - image[idx_up];
    }
    
    // g_col: horizontal gradient (column direction)
    if (x > 0u && x < params.width - 1u) {
        let idx_left = y * params.width + (x - 1u);
        let idx_right = y * params.width + (x + 1u);
        g_col = image[idx_right] - image[idx_left];
    }
    
    // Magnitude: sqrt(g_row^2 + g_col^2)
    let mag = sqrt(g_col * g_col + g_row * g_row);
    
    // Orientation: atan2(g_row, g_col) in degrees, range [0, 180)
    var angle = atan2(g_row, g_col) * (180.0 / 3.14159265359);
    if (angle < 0.0) {
        angle = angle + 180.0;
    }
    if (angle >= 180.0) {
        angle = angle - 180.0;
    }
    
    gradients[idx] = vec2<f32>(mag, angle);
}

// ============================================================================
// Shader 2: Histogram Accumulation
// ============================================================================

struct HistogramParams {
    width: u32,
    height: u32,
    n_cells_row: u32,
    n_cells_col: u32,
}

const ORIENTATIONS: u32 = 8u;
const CELL_SIZE: u32 = 8u;
const ORIENT_STEP: f32 = 22.5;  // 180.0 / 8

@group(0) @binding(0) var<uniform> hist_params: HistogramParams;
@group(0) @binding(1) var<storage, read> gradients_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> mask_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> histograms: array<f32>;  // Output: [n_cells_row * n_cells_col * 8]

@compute @workgroup_size(8, 1, 1)  // 8 threads per workgroup = 1 per orientation
fn histogram_accumulate(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let cell_col = global_id.x / ORIENTATIONS;
    let cell_row = global_id.y;
    let orient_idx = local_id.x;  // 0-7
    
    if (cell_row >= hist_params.n_cells_row || cell_col >= hist_params.n_cells_col) {
        return;
    }
    
    // Cell center coordinates
    let center_r = CELL_SIZE / 2u + cell_row * CELL_SIZE;
    let center_c = CELL_SIZE / 2u + cell_col * CELL_SIZE;
    
    // Orientation bin range for this thread
    let orient_start = ORIENT_STEP * f32(orient_idx + 1u);
    let orient_end = ORIENT_STEP * f32(orient_idx);
    
    // Accumulate magnitudes in this bin (iterate over 8x8 cell)
    let range_start = i32(CELL_SIZE / 2u);
    let range_end = i32((CELL_SIZE + 1u) / 2u);
    var sum_mag: f32 = 0.0;
    
    for (var dr: i32 = -range_start; dr < range_end; dr = dr + 1) {
        let rr = i32(center_r) + dr;
        if (rr < 0 || rr >= i32(hist_params.height)) {
            continue;
        }
        
        for (var dc: i32 = -range_start; dc < range_end; dc = dc + 1) {
            let cc = i32(center_c) + dc;
            if (cc < 0 || cc >= i32(hist_params.width)) {
                continue;
            }
            
            let pixel_idx = u32(rr) * hist_params.width + u32(cc);
            let grad = gradients_in[pixel_idx];
            let mag = grad.x;
            let ang = grad.y;

            if (mask_in[pixel_idx] != 0u && ang >= orient_end && ang < orient_start) {
                sum_mag = sum_mag + mag;
            }
        }
    }

    // Normalize by cell area to match CPU path.
    let hist_base = (cell_row * hist_params.n_cells_col + cell_col) * ORIENTATIONS;
    histograms[hist_base + orient_idx] = sum_mag / f32(CELL_SIZE * CELL_SIZE);
}

// ============================================================================
// Shader 3: L2-Hys Normalization
// ============================================================================

struct NormalizeParams {
    n_cells: u32,
}

const HOG_EPS: f32 = 0.00001;

@group(0) @binding(0) var<uniform> norm_params: NormalizeParams;
@group(0) @binding(1) var<storage, read_write> histograms_inout: array<f32>;

@compute @workgroup_size(256)
fn normalize_l2hys(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    
    if (cell_idx >= norm_params.n_cells) {
        return;
    }
    
    let hist_base = cell_idx * ORIENTATIONS;
    
    // Step 1: Compute L2 norm
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < ORIENTATIONS; i = i + 1u) {
        let val = histograms_inout[hist_base + i];
        sum_sq = sum_sq + val * val;
    }
    
    var norm = sqrt(sum_sq + HOG_EPS);
    
    // Normalize
    for (var i: u32 = 0u; i < ORIENTATIONS; i = i + 1u) {
        histograms_inout[hist_base + i] = histograms_inout[hist_base + i] / norm;
    }
    
    // Step 2: Clip to 0.2
    for (var i: u32 = 0u; i < ORIENTATIONS; i = i + 1u) {
        let val = histograms_inout[hist_base + i];
        if (val > 0.2) {
            histograms_inout[hist_base + i] = 0.2;
        }
    }
    
    // Step 3: Renormalize
    sum_sq = 0.0;
    for (var i: u32 = 0u; i < ORIENTATIONS; i = i + 1u) {
        let val = histograms_inout[hist_base + i];
        sum_sq = sum_sq + val * val;
    }
    norm = sqrt(sum_sq + HOG_EPS);
    
    for (var i: u32 = 0u; i < ORIENTATIONS; i = i + 1u) {
        histograms_inout[hist_base + i] = histograms_inout[hist_base + i] / norm;
    }
}
