//! inference.rs
//! =============
//! ECG abnormality detection inference engine — local test version.
//!
//! This is a direct port of firmware/esp32-rust/src/inference.rs adapted
//! to run on a standard Rust host (std). The inference logic is byte-for-byte
//! identical to the firmware so results can be compared against ESP32 output.
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
//! DISCLAIMER: Educational prototype only. Not for clinical use.

use crate::model_weights;

// ─── Inference Mode Selection ─────────────────────────────────────────────────

/// Select inference mode:
///   0 = Threshold classifier
///   1 = Quantized tiny MLP
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
pub fn extract_features(window: &[i32]) -> EcgFeatures {
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
///   B1 and B2 are quantized with input_scale = 42.333 = 127/3.
///   This matches Step 2: feat_q = z * 42.333, so W_q·feat_q runs at
///   scale w_scale * 42.333. Bias must live at the same scale.
///   (See quantize_weights.py :: quantize_bias_int32)
pub fn mlp_infer(features: &EcgFeatures) -> u8 {
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
    if output_acc > 0 {
        1
    } else {
        0
    }
}

// ─── Debug helpers ────────────────────────────────────────────────────────────

/// Run inference and also return the raw intermediate values for debugging.
///
/// Returns (prediction, feat_q, hidden, output_acc) to allow inspection of
/// the internal state of the MLP without needing a debugger.
pub fn mlp_infer_debug(
    features: &EcgFeatures,
) -> (u8, [i32; model_weights::N_FEATURES], [i32; model_weights::N_HIDDEN], i32) {
    let feat_raw: [f32; model_weights::N_FEATURES] = [
        features.mean as f32,
        features.maximum as f32,
        features.minimum as f32,
        features.peak_to_peak as f32,
        (features.energy / 4096).clamp(-32767, 32767) as f32,
    ];

    let mut feat_q: [i32; model_weights::N_FEATURES] = [0; model_weights::N_FEATURES];
    for i in 0..model_weights::N_FEATURES {
        let std_safe = if model_weights::FEAT_STD[i] < 1e-6 {
            1.0_f32
        } else {
            model_weights::FEAT_STD[i]
        };
        let z = (feat_raw[i] - model_weights::FEAT_MEAN[i]) / std_safe;
        feat_q[i] = (z * 42.333_f32).clamp(-128.0, 127.0) as i32;
    }

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

    let mut output_acc: i32 = 0;
    for j in 0..model_weights::N_HIDDEN {
        let w = model_weights::W2[j] as i32;
        output_acc += w * hidden[j];
    }
    output_acc += model_weights::B2[0];

    let prediction = if output_acc > 0 { 1u8 } else { 0u8 };
    (prediction, feat_q, hidden, output_acc)
}
