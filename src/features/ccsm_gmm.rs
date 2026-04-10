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
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut means = vec![0.0; n_components];
        for (k, mean) in means.iter_mut().enumerate().take(n_components) {
            let pos = (((k as f64 + 0.5) / n_components as f64) * (n as f64 - 1.0))
                .round()
                .clamp(0.0, (n - 1) as f64) as usize;
            *mean = sorted[pos];
        }

        let global_mean = values.iter().sum::<f64>() / n as f64;
        let global_var = (values
            .iter()
            .map(|&v| (v - global_mean) * (v - global_mean))
            .sum::<f64>()
            / n as f64)
            .max(MIN_VARIANCE);

        let mut weights = vec![1.0 / n_components as f64; n_components];
        let mut variances = vec![global_var; n_components];
        let mut responsibilities = vec![0.0_f64; n * n_components];

        let mut prev_ll = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0usize;

        for iter in 0..max_iter {
            n_iter = iter + 1;

            // E-step.
            let mut ll = 0.0_f64;
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
                ll += max_log + denom.ln();
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

                nk = nk.max(1e-12);
                weights[k] = (nk / n as f64).clamp(1e-12, 1.0);
                means[k] = mean_num / nk;

                let mut var_num = 0.0_f64;
                for (i, &x) in values.iter().enumerate() {
                    let r = responsibilities[i * n_components + k];
                    let d = x - means[k];
                    var_num += r * d * d;
                }
                variances[k] = (var_num / nk).max(MIN_VARIANCE);
            }

            // Keep weights normalized.
            let wsum = weights.iter().sum::<f64>().max(1e-12);
            for w in &mut weights {
                *w /= wsum;
            }

            if (ll - prev_ll).abs() < tol {
                converged = true;
                break;
            }
            prev_ll = ll;
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
