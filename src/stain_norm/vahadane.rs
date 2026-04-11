//! Vahadane stain normalization (SNMF-based).
//!
//! Ported from `Final_Code_Features_13.10.py`:
//! - `_jit_snmf_updates`
//! - `VahadaneStainNormalizer`

use ndarray::{arr2, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Zip};

use crate::core::{FeaturizerError, Result};

/// Reference stain matrix used in the Python implementation.
pub const REFERENCE_STAIN_MATRIX_V: [[f32; 2]; 3] = [[0.65, 0.07], [0.70, 0.99], [0.29, 0.11]];

/// Reference max stain concentrations used in the Python implementation.
pub const REFERENCE_MAX_CONCENTRATIONS_V: [f32; 2] = [0.8, 0.6];

/// Vahadane stain normalizer configuration and reference parameters.
#[derive(Debug, Clone)]
pub struct VahadaneStainNormalizer {
    reference_stain_matrix_v: Array2<f32>,       // (3, num_stains)
    reference_max_concentrations_v: Array1<f32>, // (num_stains,)
    num_stains: usize,
    lambda_reg: f32,
    eps: f32,
    max_iter: usize,
}

impl Default for VahadaneStainNormalizer {
    fn default() -> Self {
        Self::new(
            arr2(&REFERENCE_STAIN_MATRIX_V),
            Array1::from_vec(REFERENCE_MAX_CONCENTRATIONS_V.to_vec()),
            2,
            0.01,
            1e-8,
            200,
        )
        .expect("default Vahadane configuration must be valid")
    }
}

impl VahadaneStainNormalizer {
    /// Create a configured Vahadane normalizer.
    pub fn new(
        reference_stain_matrix_v: Array2<f32>,
        reference_max_concentrations_v: Array1<f32>,
        num_stains: usize,
        lambda_reg: f32,
        eps: f32,
        max_iter: usize,
    ) -> Result<Self> {
        if num_stains == 0 {
            return Err(FeaturizerError::InvalidInput(
                "num_stains must be >= 1".to_string(),
            ));
        }
        if reference_stain_matrix_v.nrows() != 3 || reference_stain_matrix_v.ncols() != num_stains {
            return Err(FeaturizerError::InvalidDimensions {
                expected: format!("(3, {num_stains})"),
                got: format!(
                    "({}, {})",
                    reference_stain_matrix_v.nrows(),
                    reference_stain_matrix_v.ncols()
                ),
            });
        }
        if reference_max_concentrations_v.len() != num_stains {
            return Err(FeaturizerError::InvalidDimensions {
                expected: format!("({num_stains},)"),
                got: format!("({})", reference_max_concentrations_v.len()),
            });
        }
        if !lambda_reg.is_finite() || lambda_reg < 0.0 {
            return Err(FeaturizerError::InvalidInput(
                "lambda_reg must be finite and >= 0".to_string(),
            ));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(FeaturizerError::InvalidInput(
                "eps must be finite and > 0".to_string(),
            ));
        }
        if max_iter == 0 {
            return Err(FeaturizerError::InvalidInput(
                "max_iter must be > 0".to_string(),
            ));
        }
        if !reference_stain_matrix_v.iter().all(|v| v.is_finite()) {
            return Err(FeaturizerError::InvalidInput(
                "reference_stain_matrix_v must contain finite values".to_string(),
            ));
        }
        if !reference_max_concentrations_v
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        {
            return Err(FeaturizerError::InvalidInput(
                "reference_max_concentrations_v must contain finite positive values".to_string(),
            ));
        }

        Ok(Self {
            reference_stain_matrix_v,
            reference_max_concentrations_v,
            num_stains,
            lambda_reg,
            eps,
            max_iter,
        })
    }

    /// Normalize an RGB image using Vahadane stain normalization.
    pub fn normalize(&self, img_rgb: &ArrayView3<u8>) -> Result<Array3<u8>> {
        let (height, width, channels) = img_rgb.dim();
        if height == 0 || width == 0 {
            return Err(FeaturizerError::InvalidDimensions {
                expected: "Non-zero image dimensions".to_string(),
                got: format!("({}, {}, {})", height, width, channels),
            });
        }
        if channels != 3 {
            return Err(FeaturizerError::InvalidDimensions {
                expected: "(H, W, 3)".to_string(),
                got: format!("({}, {}, {})", height, width, channels),
            });
        }

        let od = rgb_to_optical_density(img_rgb, self.eps);
        let n_pixels = height * width;
        let mut v = Array2::<f32>::zeros((3, n_pixels));
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                v[[0, idx]] = od[[y, x, 0]];
                v[[1, idx]] = od[[y, x, 1]];
                v[[2, idx]] = od[[y, x, 2]];
            }
        }

        let (mut w_source, mut h_source) = self.snmf_multiplicative_updates(&v.view())?;

        // Keep stain order consistent with Python reference.
        if self.num_stains >= 2 && w_source[[2, 0]] < w_source[[2, 1]] {
            let col0 = w_source.column(0).to_owned();
            let col1 = w_source.column(1).to_owned();
            w_source.column_mut(0).assign(&col1);
            w_source.column_mut(1).assign(&col0);

            let row0 = h_source.row(0).to_owned();
            let row1 = h_source.row(1).to_owned();
            h_source.row_mut(0).assign(&row1);
            h_source.row_mut(1).assign(&row0);
        }

        let mut current_max = Array1::<f32>::zeros(self.num_stains);
        for s in 0..self.num_stains {
            let p99 = percentile_linear(&h_source.row(s), 99.0);
            current_max[s] = if p99 > self.eps { p99 } else { self.eps };
        }

        let mut h_normalized = h_source;
        for s in 0..self.num_stains {
            let scale = self.reference_max_concentrations_v[s] / current_max[s];
            h_normalized
                .row_mut(s)
                .mapv_inplace(|x| (x * scale).max(self.eps));
        }

        let od_normalized_flat = self.reference_stain_matrix_v.dot(&h_normalized); // (3, N)
        if !od_normalized_flat.iter().all(|v| v.is_finite()) {
            return Err(FeaturizerError::NumericalError(
                "Non-finite values in normalized optical density".to_string(),
            ));
        }

        let mut normalized = Array3::<u8>::zeros((height, width, 3));
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                for ch in 0..3 {
                    let od_val = od_normalized_flat[[ch, idx]];
                    let rgb_val = 255.0 * (-od_val).exp();
                    normalized[[y, x, ch]] = rgb_val.clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(normalized)
    }

    /// Run SNMF multiplicative updates with internally-initialized W/H.
    pub fn snmf_multiplicative_updates(
        &self,
        v: &ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let (w_init, h_init) = self.initial_wh(v)?;
        snmf_update_loop(
            v,
            &w_init,
            &h_init,
            self.max_iter,
            self.lambda_reg,
            self.eps,
        )
    }

    fn initial_wh(&self, v: &ArrayView2<f32>) -> Result<(Array2<f32>, Array2<f32>)> {
        let (m, n) = v.dim();
        if m == 0 || n == 0 {
            return Err(FeaturizerError::InvalidDimensions {
                expected: "Non-zero V dimensions".to_string(),
                got: format!("({}, {})", m, n),
            });
        }

        // Match Python reference initialization:
        // W = np.random.rand(m, k) + eps ; H = np.random.rand(k, n) + eps
        let mut rng = XorShift64::new(0x9E37_79B9_7F4A_7C15);
        let mut w = Array2::<f32>::zeros((m, self.num_stains));
        for i in 0..m {
            for j in 0..self.num_stains {
                w[[i, j]] = rng.next_f32() + self.eps;
            }
        }

        normalize_w_columns(&mut w, self.eps);

        let mut h = Array2::<f32>::zeros((self.num_stains, n));
        for i in 0..self.num_stains {
            for j in 0..n {
                h[[i, j]] = rng.next_f32() + self.eps;
            }
        }

        Ok((w, h))
    }
}

/// Core SNMF multiplicative update loop.
///
/// Port of `_jit_snmf_updates`:
/// - `H = H * (W.T @ V) / (W.T @ W @ H + lambda_reg + eps)`
/// - `W = W * (V @ H.T) / (W @ H @ H.T + eps)`
/// - `W` column-wise L2 normalization each iteration
pub fn snmf_update_loop(
    v: &ArrayView2<f32>,
    w_init: &Array2<f32>,
    h_init: &Array2<f32>,
    max_iter: usize,
    lambda_reg: f32,
    eps: f32,
) -> Result<(Array2<f32>, Array2<f32>)> {
    if max_iter == 0 {
        return Err(FeaturizerError::InvalidInput(
            "max_iter must be > 0".to_string(),
        ));
    }
    if !lambda_reg.is_finite() || lambda_reg < 0.0 {
        return Err(FeaturizerError::InvalidInput(
            "lambda_reg must be finite and >= 0".to_string(),
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(FeaturizerError::InvalidInput(
            "eps must be finite and > 0".to_string(),
        ));
    }

    let (m, n) = v.dim();
    let (w_m, k) = w_init.dim();
    let (h_k, h_n) = h_init.dim();

    if m == 0 || n == 0 || k == 0 {
        return Err(FeaturizerError::InvalidDimensions {
            expected: "Non-zero matrix dimensions".to_string(),
            got: format!("V=({}, {}), W=({}, {}), H=({}, {})", m, n, w_m, k, h_k, h_n),
        });
    }
    if w_m != m || h_k != k || h_n != n {
        return Err(FeaturizerError::InvalidDimensions {
            expected: format!("W=({}, {}), H=({}, {})", m, k, k, n),
            got: format!("W=({}, {}), H=({}, {})", w_m, k, h_k, h_n),
        });
    }

    let mut w = w_init.to_owned();
    let mut h = h_init.to_owned();

    for _ in 0..max_iter {
        // H update
        let wt = w.t().to_owned();
        let numerator_h = wt.dot(v); // (k, n)
        let wt_w = wt.dot(&w); // (k, k)
        let mut denominator_h = wt_w.dot(&h); // (k, n)
        denominator_h.mapv_inplace(|x| x + lambda_reg + eps);

        Zip::from(&mut h)
            .and(&numerator_h)
            .and(&denominator_h)
            .for_each(|h_ij, &num, &den| {
                let ratio = num / den.max(eps);
                let next = *h_ij * ratio;
                *h_ij = if next.is_finite() && next > eps {
                    next
                } else {
                    eps
                };
            });

        // W update
        let ht = h.t().to_owned();
        let numerator_w = v.dot(&ht); // (m, k)
        let hh_t = h.dot(&ht); // (k, k)
        let mut denominator_w = w.dot(&hh_t); // (m, k)
        denominator_w.mapv_inplace(|x| x + eps);

        Zip::from(&mut w)
            .and(&numerator_w)
            .and(&denominator_w)
            .for_each(|w_ij, &num, &den| {
                let ratio = num / den.max(eps);
                let next = *w_ij * ratio;
                *w_ij = if next.is_finite() && next > eps {
                    next
                } else {
                    eps
                };
            });

        // Column-wise W normalization
        normalize_w_columns(&mut w, eps);
    }

    if !w.iter().all(|x| x.is_finite()) || !h.iter().all(|x| x.is_finite()) {
        return Err(FeaturizerError::NumericalError(
            "SNMF produced non-finite values".to_string(),
        ));
    }

    Ok((w, h))
}

/// Convenience wrapper using default Vahadane settings.
pub fn normalize_staining_default(img_rgb: &ArrayView3<u8>) -> Result<Array3<u8>> {
    VahadaneStainNormalizer::default().normalize(img_rgb)
}

fn rgb_to_optical_density(img_rgb: &ArrayView3<u8>, eps: f32) -> Array3<f32> {
    let (h, w, c) = img_rgb.dim();
    let mut od = Array3::<f32>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                let pix = img_rgb[[y, x, ch]] as f32;
                od[[y, x, ch]] = -((pix + eps) / 255.0).ln();
            }
        }
    }
    od
}

fn normalize_w_columns(w: &mut Array2<f32>, eps: f32) {
    let (_, cols) = w.dim();
    for col in 0..cols {
        let norm = w.column(col).iter().map(|v| v * v).sum::<f32>().sqrt();
        let denom = norm + eps;
        w.column_mut(col).mapv_inplace(|x| x / denom);
    }
}

#[derive(Clone, Copy)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xA5A5_A5A5_A5A5_A5A5
            } else {
                seed
            },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32; // 24 random bits
        (v as f32) / ((1u32 << 24) as f32)
    }
}

fn percentile_linear(values: &ArrayView1<f32>, q_percent: f32) -> f32 {
    let mut vec = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if vec.is_empty() {
        return 0.0;
    }
    vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if vec.len() == 1 {
        return vec[0];
    }

    let q = (q_percent / 100.0).clamp(0.0, 1.0);
    let rank = q * (vec.len() as f32 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        vec[lo]
    } else {
        let w = rank - lo as f32;
        vec[lo] * (1.0 - w) + vec[hi] * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn frobenius_norm(a: &Array2<f32>) -> f32 {
        a.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn test_snmf_update_loop_shapes_and_finiteness() {
        let w_true = arr2(&REFERENCE_STAIN_MATRIX_V);
        let h_true = array![
            [0.2, 0.3, 0.4, 0.1, 0.8, 0.5, 0.3, 0.7],
            [0.7, 0.6, 0.2, 0.9, 0.1, 0.4, 0.8, 0.2]
        ];
        let v = w_true.dot(&h_true);

        let normalizer = VahadaneStainNormalizer::default();
        let (w0, h0) = normalizer.initial_wh(&v.view()).unwrap();
        let before = frobenius_norm(&(v.clone() - w0.dot(&h0)));

        let (w, h) = snmf_update_loop(&v.view(), &w0, &h0, 60, 0.01, 1e-8).unwrap();
        let after = frobenius_norm(&(v - w.dot(&h)));

        assert_eq!(w.dim(), (3, 2));
        assert_eq!(h.dim(), (2, 8));
        assert!(w.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(h.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(after <= before * 1.05);

        for col in 0..2 {
            let norm = w.column(col).iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_normalize_staining_default_preserves_shape() {
        let mut img = Array3::<u8>::zeros((12, 10, 3));
        for y in 0..12 {
            for x in 0..10 {
                img[[y, x, 0]] = (50 + y * 3 + x) as u8;
                img[[y, x, 1]] = (90 + y + x * 2) as u8;
                img[[y, x, 2]] = (130 + (x % 4) * 10) as u8;
            }
        }

        let out = normalize_staining_default(&img.view()).unwrap();
        assert_eq!(out.dim(), img.dim());
    }

    #[test]
    fn test_normalize_invalid_channels_error() {
        let img = Array3::<u8>::zeros((8, 8, 1));
        let err = normalize_staining_default(&img.view()).unwrap_err();
        assert!(matches!(err, FeaturizerError::InvalidDimensions { .. }));
    }

    #[test]
    fn test_percentile_linear_matches_expected_points() {
        let arr = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let p50 = percentile_linear(&arr.view(), 50.0);
        let p99 = percentile_linear(&arr.view(), 99.0);
        assert!((p50 - 3.0).abs() < 1e-6);
        assert!(p99 >= 4.9 && p99 <= 5.0);
    }
}
