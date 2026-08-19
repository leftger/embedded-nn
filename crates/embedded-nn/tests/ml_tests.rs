#![cfg(feature = "libm")]

use embedded_nn::*;

#[test]
fn test_gaussian_naive_bayes() {
    let theta = [0.0f32, 0.0, 5.0, 5.0];
    let sigma = [1.0f32, 1.0, 1.0, 1.0];
    let priors = [0.5f32, 0.5];

    let gnb = GaussianNaiveBayesInstanceF32 {
        num_classes: 2,
        num_features: 2,
        theta: &theta,
        sigma: &sigma,
        class_prior: &priors,
        epsilon: 1e-9,
    };

    let mut probs = [0.0f32; 2];
    let cls = gnb.predict(&[4.8, 5.1], &mut probs);
    assert_eq!(cls, 1);
}

#[test]
fn test_svm_classifier() {
    let sv = [0.0f32, 0.0, 2.0, 2.0];
    let dual_coefs = [-1.0f32, 1.0];

    let svm = SvmInstanceF32 {
        num_vector_dim: 2,
        num_support_vectors: 2,
        intercept: 0.0,
        dual_coefs: &dual_coefs,
        support_vectors: &sv,
        kernel_type: SvmKernelType::Linear,
        gamma: 1.0,
        coef0: 0.0,
        degree: 1,
    };

    let mut res = 0;
    assert!(svm.predict(&[3.0, 3.0], &mut res).is_ok());
    assert_eq!(res, 1);
}

#[test]
fn test_mel_filterbank_and_mfcc() {
    let mel = hz_to_mel(1000.0);
    let hz = mel_to_hz(mel);
    assert!((hz - 1000.0).abs() < 1e-3);

    let fft_mag = [1.0f32; 128];
    let mut mel_energies = [0.0f32; 10];
    mel_filterbank_f32(&fft_mag, 16000.0, 300.0, 8000.0, &mut mel_energies);
    assert!(mel_energies[0] >= 0.0);

    let mut mfccs = [0.0f32; 5];
    mfcc_f32(&mel_energies, &mut mfccs);
    assert_eq!(mfccs.len(), 5);
}
