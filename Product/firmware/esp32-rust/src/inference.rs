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
///   [mean, maximum, minimum, peak_to_peak, energy]
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
    let mut energy_acc: i64 = 0;

    for &sample in window.iter() {
        sum += sample as i64;

        if sample > max_val {
            max_val = sample;
        }

        if sample < min_val {
            min_val = sample;
        }

        energy_acc += (sample as i64) * (sample as i64);
    }

    let mean = (sum / n as i64) as i32;
    let peak_to_peak = max_val - min_val;
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
///   Input  5: [mean, max, min, peak_to_peak, energy_scaled]
///   Hidden 8: ReLU
///   Output 1: output_acc > 0 means abnormal
fn mlp_infer(features: &EcgFeatures) -> u8 {
    // ── Step 1: Build feature vector ─────────────────────────────────────────
    //
    // Energy is much larger than the other ADC-scale features, so compress it.
    let feat: [i32; model_weights::N_FEATURES] = [
        features.mean,
        features.maximum,
        features.minimum,
        features.peak_to_peak,
        (features.energy / 4096).clamp(-32767, 32767),
    ];

    // ── Step 2: Normalize features to i8-like range [-128, 127] ─────────────
    //
    // This is a lightweight embedded normalization.
    // It keeps the model fully integer-only.
    let mut feat_max: i32 = 1;

    for &f in feat.iter() {
        let abs_f = f.abs();

        if abs_f > feat_max {
            feat_max = abs_f;
        }
    }

    let mut feat_q: [i32; model_weights::N_FEATURES] = [0; model_weights::N_FEATURES];

    for i in 0..model_weights::N_FEATURES {
        let scaled = (feat[i] as i64 * 127i64) / feat_max as i64;
        feat_q[i] = scaled.clamp(-128, 127) as i32;
    }

    // ── Step 3: Layer 1 forward pass ─────────────────────────────────────────
    //
    // hidden[j] = ReLU(sum_i(W1[j][i] * feat_q[i]) + B1[j])
    //
    // W1 layout:
    //   [N_HIDDEN x N_FEATURES] row-major
    let mut hidden: [i32; model_weights::N_HIDDEN] = [0; model_weights::N_HIDDEN];

    for j in 0..model_weights::N_HIDDEN {
        let mut acc: i32 = 0;

        for i in 0..model_weights::N_FEATURES {
            let idx = j * model_weights::N_FEATURES + i;
            let w = model_weights::W1[idx] as i32;
            acc += w * feat_q[i];
        }

        acc += model_weights::B1[j];
        hidden[j] = acc.max(0);
    }

    // ── Step 4: Re-quantize hidden layer to i8-like range ────────────────────
    let mut h_max: i32 = 1;

    for &h in hidden.iter() {
        if h > h_max {
            h_max = h;
        }
    }

    let mut hidden_q: [i32; model_weights::N_HIDDEN] = [0; model_weights::N_HIDDEN];

    for j in 0..model_weights::N_HIDDEN {
        let scaled = (hidden[j] as i64 * 127i64) / h_max as i64;
        hidden_q[j] = scaled.clamp(-128, 127) as i32;
    }

    // ── Step 5: Layer 2 forward pass ─────────────────────────────────────────
    //
    // output = sum_j(W2[j] * hidden_q[j]) + B2[0]
    let mut output_acc: i32 = 0;

    for j in 0..model_weights::N_HIDDEN {
        let w = model_weights::W2[j] as i32;
        output_acc += w * hidden_q[j];
    }

    output_acc += model_weights::B2[0];

    // ── Step 6: Binary decision ──────────────────────────────────────────────
    if output_acc > 0 {
        1
    } else {
        0
    }
}