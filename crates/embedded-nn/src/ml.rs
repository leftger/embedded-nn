//! Classical machine learning classifiers (Support Vector Machines, Gaussian Naive Bayes) and audio feature extraction (Mel filterbank, MFCC).

#[inline(always)]
fn math_log(x: f32) -> f32 {
    #[cfg(feature = "libm")]
    {
        libm::logf(x)
    }
    #[cfg(not(feature = "libm"))]
    {
        x.ln()
    }
}

#[inline(always)]
fn math_exp(x: f32) -> f32 {
    #[cfg(feature = "libm")]
    {
        libm::expf(x)
    }
    #[cfg(not(feature = "libm"))]
    {
        x.exp()
    }
}

#[inline(always)]
fn math_pow(x: f32, y: f32) -> f32 {
    #[cfg(feature = "libm")]
    {
        libm::powf(x, y)
    }
    #[cfg(not(feature = "libm"))]
    {
        x.powf(y)
    }
}

#[inline(always)]
fn math_tanh(x: f32) -> f32 {
    #[cfg(feature = "libm")]
    {
        libm::tanhf(x)
    }
    #[cfg(not(feature = "libm"))]
    {
        x.tanh()
    }
}

#[inline(always)]
fn math_cos(x: f32) -> f32 {
    #[cfg(feature = "libm")]
    {
        libm::cosf(x)
    }
    #[cfg(not(feature = "libm"))]
    {
        x.cos()
    }
}

/// Kernel function type for Support Vector Machine (SVM) classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmKernelType {
    /// Linear kernel: `K(x, y) = x^T * y`
    Linear,
    /// Polynomial kernel: `K(x, y) = (gamma * x^T * y + coef0)^degree`
    Polynomial,
    /// Radial Basis Function (RBF) kernel: `K(x, y) = exp(-gamma * ||x - y||^2)`
    Rbf,
    /// Sigmoid kernel: `K(x, y) = tanh(gamma * x^T * y + coef0)`
    Sigmoid,
}

/// Gaussian Naive Bayes classifier instance for floating-point feature vectors.
pub struct GaussianNaiveBayesInstanceF32<'a> {
    /// Number of output classes.
    pub num_classes: u32,
    /// Number of input features per sample.
    pub num_features: u32,
    /// Mean parameters matrix of shape `num_classes * num_features`.
    pub theta: &'a [f32],
    /// Variance parameters matrix of shape `num_classes * num_features`.
    pub sigma: &'a [f32],
    /// Prior probability distribution slice for each class (`num_classes`).
    pub class_prior: &'a [f32],
    /// Epsilon factor added to variances for numerical stability.
    pub epsilon: f32,
}

impl<'a> GaussianNaiveBayesInstanceF32<'a> {
    /// Predict class label for feature vector `x`. Populates `log_probs` with class log-likelihoods.
    pub fn predict(&self, x: &[f32], log_probs: &mut [f32]) -> u32 {
        let num_classes = self.num_classes as usize;
        let num_features = self.num_features as usize;
        assert!(x.len() >= num_features, "Input feature dimension mismatch");
        assert!(
            log_probs.len() >= num_classes,
            "log_probs buffer size mismatch"
        );

        let mut best_class = 0;
        let mut max_log_prob = f32::NEG_INFINITY;

        for c in 0..num_classes {
            let prior = self.class_prior[c];
            let mut log_prob = if prior > 0.0 { math_log(prior) } else { -1e9 };

            for f in 0..num_features {
                let idx = c * num_features + f;
                let mean = self.theta[idx];
                let var = self.sigma[idx] + self.epsilon;
                let diff = x[f] - mean;

                let pi = core::f32::consts::PI;
                let log_term = math_log(2.0 * pi * var);
                log_prob -= 0.5 * (log_term + (diff * diff) / var);
            }

            log_probs[c] = log_prob;
            if log_prob > max_log_prob {
                max_log_prob = log_prob;
                best_class = c as u32;
            }
        }

        best_class
    }
}

/// Support Vector Machine (SVM) binary classifier instance.
pub struct SvmInstanceF32<'a> {
    /// Dimension of each support vector / input feature vector.
    pub num_vector_dim: u32,
    /// Total number of support vectors stored.
    pub num_support_vectors: u32,
    /// Decision function bias term (intercept).
    pub intercept: f32,
    /// Dual coefficients array of length `num_support_vectors`.
    pub dual_coefs: &'a [f32],
    /// Support vectors matrix of shape `num_support_vectors * num_vector_dim`.
    pub support_vectors: &'a [f32],
    /// Kernel function type.
    pub kernel_type: SvmKernelType,
    /// Kernel gamma coefficient.
    pub gamma: f32,
    /// Kernel coef0 constant (for Polynomial and Sigmoid).
    pub coef0: f32,
    /// Polynomial degree (for Polynomial kernel).
    pub degree: i32,
}

impl<'a> SvmInstanceF32<'a> {
    /// Predict binary class label (0 or 1) for input feature vector `x`.
    pub fn predict(&self, x: &[f32], out_label: &mut u32) -> crate::types::Result<()> {
        let dim = self.num_vector_dim as usize;
        let num_sv = self.num_support_vectors as usize;
        if x.len() < dim {
            return Err(crate::types::Error::ArgumentError);
        }

        let mut sum = self.intercept;
        for i in 0..num_sv {
            let sv = &self.support_vectors[i * dim..(i + 1) * dim];
            let k_val = match self.kernel_type {
                SvmKernelType::Linear => {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += x[d] * sv[d];
                    }
                    dot
                }
                SvmKernelType::Polynomial => {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += x[d] * sv[d];
                    }
                    let base = self.gamma * dot + self.coef0;
                    math_pow(base, self.degree as f32)
                }
                SvmKernelType::Rbf => {
                    let mut dist_sq = 0.0f32;
                    for d in 0..dim {
                        let diff = x[d] - sv[d];
                        dist_sq += diff * diff;
                    }
                    math_exp(-self.gamma * dist_sq)
                }
                SvmKernelType::Sigmoid => {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += x[d] * sv[d];
                    }
                    math_tanh(self.gamma * dot + self.coef0)
                }
            };

            sum += self.dual_coefs[i] * k_val;
        }

        *out_label = if sum >= 0.0 { 1 } else { 0 };
        Ok(())
    }
}

/// Converts frequency in Hertz to Mel scale.
pub fn hz_to_mel(hz: f32) -> f32 {
    let inner = 1.0 + hz / 700.0;
    2595.0 * math_log(inner) / math_log(10.0)
}

/// Converts Mel scale to frequency in Hertz.
pub fn mel_to_hz(mel: f32) -> f32 {
    let pow_arg = mel / 2595.0;
    700.0 * (math_pow(10.0, pow_arg) - 1.0)
}

/// Computes Mel-filterbank energies from FFT magnitude spectrum.
pub fn mel_filterbank_f32(
    fft_mag: &[f32],
    sample_rate: f32,
    min_freq: f32,
    max_freq: f32,
    mel_energies: &mut [f32],
) {
    let num_filters = mel_energies.len();
    let num_bins = fft_mag.len();
    let min_mel = hz_to_mel(min_freq);
    let max_mel = hz_to_mel(max_freq);
    let mel_step = (max_mel - min_mel) / (num_filters + 1) as f32;

    for filter_idx in 0..num_filters {
        let left_mel = min_mel + filter_idx as f32 * mel_step;
        let center_mel = min_mel + (filter_idx + 1) as f32 * mel_step;
        let right_mel = min_mel + (filter_idx + 2) as f32 * mel_step;

        let left_hz = mel_to_hz(left_mel);
        let center_hz = mel_to_hz(center_mel);
        let right_hz = mel_to_hz(right_mel);

        let left_bin = (left_hz / sample_rate * (2 * num_bins) as f32) as usize;
        let center_bin = (center_hz / sample_rate * (2 * num_bins) as f32) as usize;
        let right_bin = (right_hz / sample_rate * (2 * num_bins) as f32) as usize;

        let mut energy = 0.0f32;

        if center_bin > left_bin {
            for bin in left_bin..center_bin.min(num_bins) {
                let weight = (bin - left_bin) as f32 / (center_bin - left_bin) as f32;
                energy += fft_mag[bin] * weight;
            }
        }

        if right_bin > center_bin {
            for bin in center_bin..right_bin.min(num_bins) {
                let weight = (right_bin - bin) as f32 / (right_bin - center_bin) as f32;
                energy += fft_mag[bin] * weight;
            }
        }

        mel_energies[filter_idx] = energy;
    }
}

/// Computes Mel-Frequency Cepstral Coefficients (MFCCs) using Discrete Cosine Transform (DCT-II).
pub fn mfcc_f32(mel_energies: &[f32], mfccs: &mut [f32]) {
    let num_filters = mel_energies.len();
    let num_cepstral = mfccs.len();

    for i in 0..num_cepstral {
        let mut sum = 0.0f32;
        for j in 0..num_filters {
            let val = mel_energies[j] + 1e-6;
            let log_e = math_log(val);
            let angle = (j as f32 + 0.5) * i as f32 * core::f32::consts::PI / num_filters as f32;
            let cos_factor = math_cos(angle);
            sum += log_e * cos_factor;
        }
        mfccs[i] = sum;
    }
}
