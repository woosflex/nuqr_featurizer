//! 1D Gaussian Mixture Model (EM) for CCSM condensed chromatin segmentation.

use crate::core::{FeaturizerError, Result};

const MIN_VARIANCE: f64 = 1e-6;

/// Simple 1D Gaussian Mixture Model fitted with EM.
///
/// This model is used by CCSM to separate condensed versus diffuse chromatin
/// intensity populations.
#[derive(Debug, Clone)]
pub struct GaussianMixture1D {
    /// Mixture weights for each component.
    pub weights: Vec<f64>,
    /// Component means.
    pub means: Vec<f64>,
    /// Component variances (clamped to a minimum for stability).
    pub variances: Vec<f64>,
    /// Whether convergence criterion was met before reaching `max_iter`.
    pub converged: bool,
    /// Number of EM iterations actually executed.
    pub n_iter: usize,
}

impl GaussianMixture1D {
    /// Fit a 1D Gaussian mixture model using expectation-maximization.
    ///
    /// # Arguments
    /// * `values` - Input samples.
    /// * `n_components` - Number of Gaussian components.
    /// * `max_iter` - Maximum EM iterations.
    /// * `tol` - Convergence tolerance on log-likelihood delta.
    pub fn fit(values: &[f64], n_components: usize, max_iter: usize, tol: f64) -> Result<Self> {
        if values.len() < 2 {
            return Err(FeaturizerError::InvalidInput(
                "GMM requires at least 2 samples".to_string(),
            ));
        }
        if n_components < 1 || n_components > values.len() {
            return Err(FeaturizerError::InvalidInput(format!(
                "Invalid n_components: {n_components}"
            )));
        }
        if max_iter == 0 {
            return Err(FeaturizerError::InvalidInput(
                "max_iter must be > 0".to_string(),
            ));
        }

        let n = values.len();
        let mut rng = NpRandomState::new(0);
        let centers0 = kmeans_plusplus_init_1d(values, n_components, &mut rng);
        let (mut means, init_labels) = lloyd_refine_1d(values, centers0, 20);

        let global_mean = values.iter().sum::<f64>() / n as f64;
        let global_var = (values
            .iter()
            .map(|&v| (v - global_mean) * (v - global_mean))
            .sum::<f64>()
            / n as f64)
            .max(MIN_VARIANCE);

        let eps = f64::EPSILON;
        let mut weights = vec![1.0 / n_components as f64; n_components];
        let mut variances = vec![0.0_f64; n_components];
        let mut counts = vec![0usize; n_components];
        let mut sums = vec![0.0_f64; n_components];
        for (i, &label) in init_labels.iter().enumerate() {
            if label < n_components {
                counts[label] += 1;
                sums[label] += values[i];
            }
        }
        for k in 0..n_components {
            if counts[k] > 0 {
                means[k] = sums[k] / counts[k] as f64;
                weights[k] = counts[k] as f64 / n as f64;
            }
        }
        for (i, &x) in values.iter().enumerate() {
            let k = init_labels[i];
            if k < n_components && counts[k] > 0 {
                let d = x - means[k];
                variances[k] += d * d;
            }
        }
        for k in 0..n_components {
            if counts[k] > 0 {
                variances[k] = variances[k] / counts[k] as f64 + MIN_VARIANCE;
            } else {
                variances[k] = global_var + MIN_VARIANCE;
            }
        }
        let wsum = weights.iter().sum::<f64>().max(1e-12);
        for w in &mut weights {
            *w /= wsum;
        }
        let mut responsibilities = vec![0.0_f64; n * n_components];

        let mut prev_lower_bound = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0usize;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            // E-step.
            let mut ll_sum = 0.0_f64;
            for (i, &x) in values.iter().enumerate() {
                let base = i * n_components;
                let mut max_log = f64::NEG_INFINITY;
                for k in 0..n_components {
                    let lp = weights[k].ln() + log_gaussian(x, means[k], variances[k]);
                    responsibilities[base + k] = lp;
                    if lp > max_log {
                        max_log = lp;
                    }
                }

                let mut sum_exp = 0.0_f64;
                for k in 0..n_components {
                    let v = (responsibilities[base + k] - max_log).exp();
                    responsibilities[base + k] = v;
                    sum_exp += v;
                }

                let denom = sum_exp.max(1e-300);
                ll_sum += max_log + denom.ln();
                for k in 0..n_components {
                    responsibilities[base + k] /= denom;
                }
            }

            // M-step.
            for k in 0..n_components {
                let mut nk = 0.0_f64;
                let mut mean_num = 0.0_f64;
                for (i, &x) in values.iter().enumerate() {
                    let r = responsibilities[i * n_components + k];
                    nk += r;
                    mean_num += r * x;
                }

                nk += 10.0 * eps;
                weights[k] = nk / n as f64;
                means[k] = mean_num / nk;

                let mut var_num = 0.0_f64;
                for (i, &x) in values.iter().enumerate() {
                    let r = responsibilities[i * n_components + k];
                    let d = x - means[k];
                    var_num += r * d * d;
                }
                variances[k] = var_num / nk + MIN_VARIANCE;
            }

            // Keep weights normalized.
            let wsum = weights.iter().sum::<f64>().max(1e-12);
            for w in &mut weights {
                *w /= wsum;
            }

            // Match sklearn GaussianMixture convergence criterion:
            // lower_bound is mean log-likelihood (not sum).
            let lower_bound = ll_sum / n as f64;
            if (lower_bound - prev_lower_bound).abs() < tol {
                converged = true;
                break;
            }
            prev_lower_bound = lower_bound;
        }

        Ok(Self {
            weights,
            means,
            variances,
            converged,
            n_iter,
        })
    }

    /// Predict most-likely component index for each sample.
    pub fn predict_labels(&self, values: &[f64]) -> Vec<usize> {
        let k = self.means.len();
        let mut labels = vec![0usize; values.len()];
        for (i, &x) in values.iter().enumerate() {
            let mut best = 0usize;
            let mut best_log = f64::NEG_INFINITY;
            for c in 0..k {
                let lp = self.weights[c].ln() + log_gaussian(x, self.means[c], self.variances[c]);
                if lp > best_log {
                    best_log = lp;
                    best = c;
                }
            }
            labels[i] = best;
        }
        labels
    }

    /// Return the component index with the lowest mean intensity.
    ///
    /// In CCSM this corresponds to the condensed chromatin population.
    pub fn condensed_component_index(&self) -> usize {
        self.means
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

#[inline]
fn log_gaussian(x: f64, mean: f64, var: f64) -> f64 {
    let v = var.max(MIN_VARIANCE);
    let d = x - mean;
    -0.5 * (d * d / v + (2.0 * std::f64::consts::PI * v).ln())
}

fn kmeans_plusplus_init_1d(
    values: &[f64],
    n_components: usize,
    rng: &mut NpRandomState,
) -> Vec<f64> {
    let n = values.len();
    let mut centers = Vec::with_capacity(n_components);
    centers.push(values[rng.gen_index(n)]);

    while centers.len() < n_components {
        let mut dist2 = vec![0.0_f64; n];
        let mut current_potential = 0.0_f64;
        for (i, &x) in values.iter().enumerate() {
            let best = centers
                .iter()
                .map(|&c| {
                    let d = x - c;
                    d * d
                })
                .fold(f64::INFINITY, f64::min);
            dist2[i] = best;
            current_potential += best;
        }

        if current_potential <= 1e-12 {
            centers.push(values[rng.gen_index(n)]);
            continue;
        }

        // Match sklearn's kmeans++ local trial strategy:
        // n_local_trials = 2 + ln(n_clusters)
        let n_local_trials = 2 + (n_components as f64).ln() as usize;
        let mut best_candidate = 0usize;
        let mut best_potential = f64::INFINITY;

        for _ in 0..n_local_trials.max(1) {
            let mut target = rng.random_sample() * current_potential;
            let mut candidate = n - 1;
            for (i, &d2) in dist2.iter().enumerate() {
                target -= d2;
                if target <= 0.0 {
                    candidate = i;
                    break;
                }
            }

            let c = values[candidate];
            let mut candidate_potential = 0.0_f64;
            for (i, &x) in values.iter().enumerate() {
                let d = x - c;
                let d2 = d * d;
                candidate_potential += dist2[i].min(d2);
            }

            if candidate_potential < best_potential {
                best_potential = candidate_potential;
                best_candidate = candidate;
            }
        }

        centers.push(values[best_candidate]);
    }

    centers
}

fn lloyd_refine_1d(
    values: &[f64],
    mut centers: Vec<f64>,
    max_iter: usize,
) -> (Vec<f64>, Vec<usize>) {
    let k = centers.len();
    let mut labels = vec![0usize; values.len()];

    for _ in 0..max_iter {
        let mut changed = false;
        for (i, &x) in values.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (j, &c) in centers.iter().enumerate() {
                let d = x - c;
                let d2 = d * d;
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_idx = j;
                }
            }
            if labels[i] != best_idx {
                labels[i] = best_idx;
                changed = true;
            }
        }

        let mut counts = vec![0usize; k];
        let mut sums = vec![0.0_f64; k];
        for (i, &x) in values.iter().enumerate() {
            let lbl = labels[i];
            counts[lbl] += 1;
            sums[lbl] += x;
        }

        let mut max_shift = 0.0_f64;
        for j in 0..k {
            if counts[j] > 0 {
                let next = sums[j] / counts[j] as f64;
                max_shift = max_shift.max((next - centers[j]).abs());
                centers[j] = next;
            }
        }

        if !changed || max_shift < 1e-9 {
            break;
        }
    }

    (centers, labels)
}

struct NpRandomState {
    mt: [u32; 624],
    index: usize,
}

impl NpRandomState {
    fn new(seed: u32) -> Self {
        let mut mt = [0_u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            let prev = mt[i - 1];
            let v = 1812433253_u64
                .wrapping_mul((prev ^ (prev >> 30)) as u64)
                .wrapping_add(i as u64);
            mt[i] = (v & 0xFFFF_FFFF) as u32;
        }
        Self { mt, index: 624 }
    }

    fn twist(&mut self) {
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7FFF_FFFF;
        const MATRIX_A: u32 = 0x9908_B0DF;

        for i in 0..624 {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % 624] & LOWER_MASK);
            let mut xa = x >> 1;
            if (x & 1) != 0 {
                xa ^= MATRIX_A;
            }
            self.mt[i] = self.mt[(i + 397) % 624] ^ xa;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.mt[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^= y >> 18;
        y
    }

    #[inline]
    fn random_sample(&mut self) -> f64 {
        // Match NumPy RandomState random_sample bit-assembly.
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        ((a << 26) + b) as f64 / ((1_u64 << 53) as f64)
    }

    #[inline]
    fn gen_index(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            ((self.random_sample() * upper as f64).floor() as usize).min(upper - 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmm_bimodal_fit() {
        let mut data = Vec::new();
        for i in 0..200 {
            data.push(40.0 + (i % 5) as f64);
        }
        for i in 0..200 {
            data.push(180.0 + (i % 5) as f64);
        }

        let model = GaussianMixture1D::fit(&data, 2, 200, 1e-6).unwrap();
        assert_eq!(model.means.len(), 2);
        let mut means = model.means.clone();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(means[0] < 80.0);
        assert!(means[1] > 140.0);
    }

    #[test]
    fn test_gmm_predict_labels_length() {
        let data = vec![10.0, 11.0, 12.0, 200.0, 201.0, 202.0];
        let model = GaussianMixture1D::fit(&data, 2, 100, 1e-6).unwrap();
        let labels = model.predict_labels(&data);
        assert_eq!(labels.len(), data.len());
    }

    #[test]
    fn test_gmm_constant_data_stable() {
        let data = vec![77.0; 128];
        let model = GaussianMixture1D::fit(&data, 2, 100, 1e-6).unwrap();
        assert!(model.means.iter().all(|m| m.is_finite()));
        assert!(model
            .variances
            .iter()
            .all(|v| v.is_finite() && *v >= MIN_VARIANCE));
    }

    #[test]
    fn test_condensed_component_index() {
        let data = vec![5.0, 6.0, 7.0, 200.0, 210.0, 220.0];
        let model = GaussianMixture1D::fit(&data, 2, 200, 1e-6).unwrap();
        let idx = model.condensed_component_index();
        assert!(idx < 2);
    }
}
