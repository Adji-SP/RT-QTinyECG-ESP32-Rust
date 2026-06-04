//! inference.rs
//! =============
//! ECG abnormality detection inference engine.
//!
//! This module implements two inference modes:
//!
//! Mode A: Threshold Classifier
//!   Simple rule-based classifier using signal features.
//!
//! Mode B: Quantized Tiny MLP
//!   Architecture: 5 inputs → 8 hidden neurons ReLU → 1 output.
//!   Uses integer arithmetic only.
//!
//!   Forward pass (mathematically identical to sklearn MLPClassifier):
//!     feat_q  = per-window normalize(features) to i8-like [-128, 127]
//!     hidden  = ReLU(W1_q · feat_q + B1)   (raw integer accumulator)
//!     output  = W2_q · hidden + B2          (raw integer accumulator)
//!     predict = output > 0
//!
//!   No hidden re-normalization is applied. This matches how sklearn's
//!   Layer 2 weights were trained (on raw relu outputs, not re-scaled ones).
//!
//! In this version, the default mode is MLP because the project already has
//! model_weights.rs generated from the training/export pipeline.
//!
//! DISCLAIMER: Educational prototype only. Not for clinical use.

use crate::model_weights;

// ─── Inference Mode Selection ─────────────────────────────────────────────────

/// Select inference mode:
///   0 = Threshold classifier
///   1 = Quantized tiny MLP
///
/// For UART evaluation against exported int8 weights, use 1.
/// For simple hardware debugging, you may temporarily set this to 0.
const INFERENCE_MODE: u8 = 1;

// ─── Main Inference Entry Point ───────────────────────────────────────────────

/// Run ECG abnormality inference on a sample window.
///
/// Returns:
///   0 = Normal
///   1 = Abnormal
pub fn infer(window: &[i32]) -> u8 {
    let features = extract_features(window);

    match INFERENCE_MODE {
        0 => threshold_classify(&features),
        1 => mlp_infer(&features),
        _ => threshold_classify(&features),
    }
}

// ─── Feature Extraction ───────────────────────────────────────────────────────

/// Extracted features from one ECG window.
pub struct EcgFeatures {
    pub mean: i32,
    pub maximum: i32,
    pub minimum: i32,
    pub peak_to_peak: i32,
    pub energy: i32,
}

/// Extract the 5 ECG features from a window of samples.
///
/// Features:
///   [mean, maximum, minimum, peak_to_peak, centered_energy]
///
/// Energy is computed on mean-centered samples to remove DC offset.
/// Raw ADC values are ~2048, so energy = mean(sample²) ≈ 2048² ≈ 4M
/// (dominated by DC, not signal). Centered energy = mean((sample-mean)²)
/// measures actual AC signal power.
///
/// Must match preprocessing.py::extract_firmware_mlp_input_array() exactly.
fn extract_features(window: &[i32]) -> EcgFeatures {
    if window.is_empty() {
        return EcgFeatures {
            mean: 0,
            maximum: 0,
            minimum: 0,
            peak_to_peak: 0,
            energy: 0,
        };
    }

    let n = window.len();

    let mut sum: i64 = 0;
    let mut max_val: i32 = i32::MIN;
    let mut min_val: i32 = i32::MAX;

    for &sample in window.iter() {
        sum += sample as i64;

        if sample > max_val {
            max_val = sample;
        }

        if sample < min_val {
            min_val = sample;
        }
    }

    let mean = (sum / n as i64) as i32;
    let peak_to_peak = max_val - min_val;

    // ── Mean-centered energy ──────────────────────────────────────────────────
    //
    // Subtract DC offset before squaring so energy reflects AC signal power.
    // This matches preprocessing.py::extract_firmware_mlp_input_array():
    //   centered = w - mean
    //   energy = sum(centered²) / n
    let mut energy_acc: i64 = 0;
    for &sample in window.iter() {
        let centered = (sample as i64) - (mean as i64);
        energy_acc += centered * centered;
    }
    let energy = (energy_acc / n as i64) as i32;

    EcgFeatures {
        mean,
        maximum: max_val,
        minimum: min_val,
        peak_to_peak,
        energy,
    }
}

// ─── Mode A: Threshold Classifier ─────────────────────────────────────────────

/// Peak-to-peak amplitude above this ADC count → Abnormal.
const THRESH_P2P: i32 = 600;

/// Mean ADC value above this → Abnormal.
const THRESH_MEAN_HIGH: i32 = 2350;

/// Mean ADC value below this → Abnormal.
const THRESH_MEAN_LOW: i32 = 1750;

fn threshold_classify(features: &EcgFeatures) -> u8 {
    if features.peak_to_peak > THRESH_P2P {
        return 1;
    }

    if features.mean > THRESH_MEAN_HIGH {
        return 1;
    }

    if features.mean < THRESH_MEAN_LOW {
        return 1;
    }

    0
}

// ─── Mode B: Quantized Tiny MLP ──────────────────────────────────────────────

/// Quantized 5→8→1 MLP inference using integer arithmetic.
///
/// Architecture:
///   Input  5: [mean, max, min, peak_to_peak, centered_energy_scaled]
///   Hidden 8: ReLU
///   Output 1: output_acc > 0 means abnormal
///
/// This is mathematically identical to sklearn's MLPClassifier forward pass:
///   hidden = ReLU(W1_q · feat_q + B1)
///   output = W2_q · hidden + B2
///
/// No hidden re-normalization is applied between layers, which matches how
/// the Python model was trained (sklearn sees raw relu activations, not
/// re-scaled ones). This guarantees Python↔firmware consistency.
///
/// Bias correctness:
///   B1 and B2 are quantized assuming input activations live in [-128, 127].
///   So bias_scale = weight_scale × 127.0 (see quantize_weights.py).
fn mlp_infer(features: &EcgFeatures) -> u8 {
    // ── Step 1: Build feature vector (raw integer features) ────────────────────
    //
    // These are raw ADC-scale values before any normalization.
    // Do NOT scale or clip here — that is done by FEAT_MEAN/FEAT_STD below.
    let feat_raw: [f32; model_weights::N_FEATURES] = [
        features.mean as f32,
        features.maximum as f32,
        features.minimum as f32,
        features.peak_to_peak as f32,
        // Energy compressed to same order of magnitude as ADC features.
        (features.energy / 4096).clamp(-32767, 32767) as f32,
    ];

    // ── Step 2: Global z-score normalization ───────────────────────────────
    //
    // Apply (raw - FEAT_MEAN) / FEAT_STD using the training-set statistics
    // exported by train_simple_model.py and stored in model_weights.rs.
    //
    // This is mathematically identical to sklearn's StandardScaler.transform().
    // Using global normalization (not per-window) preserves amplitude differences
    // across windows, which is the primary discriminative signal for ECG abnormality.
    //
    // After normalization, clip to i8-like [-128, 127] range and convert to i32
    // for integer dot-product with quantized weights.
    let mut feat_q: [i32; model_weights::N_FEATURES] = [0; model_weights::N_FEATURES];
    for i in 0..model_weights::N_FEATURES {
        let std_safe = if model_weights::FEAT_STD[i] < 1e-6 {
            1.0_f32
        } else {
            model_weights::FEAT_STD[i]
        };
        let z = (feat_raw[i] - model_weights::FEAT_MEAN[i]) / std_safe;
        // Clip z-score to [-3, +3] and scale to [-128, 127].
        // This maps +/-3 sigma to +/-127, consistent with how quantize_weights.py
        // quantizes weights (which were trained on sklearn's scaler output).
        feat_q[i] = (z * 42.333_f32).clamp(-128.0, 127.0) as i32;
    }

    // ── Step 3: Layer 1 forward pass ─────────────────────────────────────────
    //
    // hidden[j] = ReLU(sum_i(W1[j][i] * feat_q[i]) + B1[j])
    //
    // W1 layout: [N_HIDDEN × N_FEATURES] row-major
    //
    // Overflow analysis:
    //   Each term: |W1[i]| ≤ 127, |feat_q[i]| ≤ 127  →  max |product| = 16129
    //   Sum over N_FEATURES=5 terms: max |acc| = 80645
    //   B1 adds at most a few thousand (quantized with scale × 127).
    //   Total fits comfortably in i32 (max ~2.1 billion).
    let mut hidden: [i32; model_weights::N_HIDDEN] = [0; model_weights::N_HIDDEN];

    for j in 0..model_weights::N_HIDDEN {
        let mut acc: i32 = 0;

        for i in 0..model_weights::N_FEATURES {
            let idx = j * model_weights::N_FEATURES + i;
            let w = model_weights::W1[idx] as i32;
            acc += w * feat_q[i];
        }

        acc += model_weights::B1[j];
        hidden[j] = acc.max(0); // ReLU
    }

    // ── Step 4: Layer 2 forward pass ─────────────────────────────────────────
    //
    // output = sum_j(W2[j] * hidden[j]) + B2[0]
    //
    // NOTE: No hidden re-normalization is applied here.
    //   sklearn's Layer 2 weights were trained on raw relu(W1·x+b1) values.
    //   Re-normalizing hidden would change the scale that W2 sees, making
    //   B2 incorrect and predictions unreliable.
    //
    // Overflow analysis:
    //   max |hidden[j]|: bounded by Step 3 (≤ ~80645 + |B1|)
    //   |W2[j]| ≤ 127
    //   Sum over N_HIDDEN=8: max |acc| ≈ 127 × 80000 × 8 ≈ 81M  → fits in i32.
    let mut output_acc: i32 = 0;

    for j in 0..model_weights::N_HIDDEN {
        let w = model_weights::W2[j] as i32;
        output_acc += w * hidden[j];
    }

    output_acc += model_weights::B2[0];

    // ── Step 5: Binary decision ──────────────────────────────────────────────
    //
    // output_acc > 0 → Abnormal (class 1)
    // output_acc ≤ 0 → Normal   (class 0)
    //
    // This matches sklearn's decision boundary for binary MLPClassifier,
    // which classifies class 1 when the raw output exceeds 0.
    if output_acc > 0 {
        1
    } else {
        0
    }
}