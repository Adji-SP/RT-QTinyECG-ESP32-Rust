//! inference.rs
//! =============
//! ECG abnormality detection inference engine.
//!
//! This module implements two inference modes:
//!
//! # Mode A: Threshold Classifier (default)
//! Simple rule-based classifier using signal features:
//!   - Peak-to-peak amplitude exceeds threshold → Abnormal
//!   - Mean value too high (elevated baseline) → Abnormal
//!   - Mean value too low (depressed baseline) → Abnormal
//! No model weights required. Fast and predictable.
//!
//! # Mode B: Quantized Tiny MLP
//! Small Multi-Layer Perceptron with int8 weights:
//!   Architecture: 5 inputs → 8 hidden neurons (ReLU) → 1 output
//!   Uses integer arithmetic only (no float).
//!   Weights loaded from model_weights.rs.
//!
//! # Inference mode selection:
//! Change `INFERENCE_MODE` constant below to switch between modes.
//!   0 = Threshold classifier (fast, no weights needed)
//!   1 = Quantized MLP (requires trained weights in model_weights.rs)
//!
//! # Python equivalent:
//! python/realtime_ecg_simulator.py → threshold_inference(), mlp_inference_manual()
//!
//! # Performance on ESP32 (estimated at 240 MHz):
//!   Threshold classifier: ~5–20 µs
//!   Quantized MLP:       ~20–60 µs
//! Both are well within the 4 ms (4000 µs) sampling interval budget.

use crate::model_weights;

// ─── Inference Mode Selection ─────────────────────────────────────────────────

/// Select inference mode:
///   0 = Threshold classifier (simple, fast, no weights)
///   1 = Quantized tiny MLP (uses model_weights.rs)
///
/// Change this constant to switch modes.
/// Rebuild firmware after changing: cargo build --release
const INFERENCE_MODE: u8 = 0;

// ─── Main Inference Entry Point ───────────────────────────────────────────────

/// Run ECG abnormality inference on a sample window.
///
/// # Arguments
/// * `window` - Slice of filtered ADC samples (length = RING_BUF_SIZE = 128)
///
/// # Returns
/// * `0` = Normal ECG
/// * `1` = Abnormal ECG → triggers alert
///
/// # How it works
/// 1. Extract features from the window (mean, max, min, peak-to-peak, energy)
/// 2. Run the selected classifier
/// 3. Return prediction
pub fn infer(window: &[i32]) -> u8 {
    // Extract features from the window
    let features = extract_features(window);

    // Select inference mode
    match INFERENCE_MODE {
        0 => threshold_classify(&features),
        1 => mlp_infer(&features),
        _ => threshold_classify(&features), // fallback
    }
}

// ─── Feature Extraction ───────────────────────────────────────────────────────

/// Extracted features from one ECG window.
///
/// These 5 features are the input to both classifiers.
/// They are computed in integer arithmetic to avoid float operations.
///
/// Python equivalent:
///   python/preprocessing.py → extract_features()
pub struct EcgFeatures {
    /// Mean ADC value in the window (baseline level)
    pub mean: i32,
    /// Maximum ADC value in the window
    pub maximum: i32,
    /// Minimum ADC value in the window
    pub minimum: i32,
    /// Peak-to-peak amplitude: max - min (main heartbeat amplitude feature)
    pub peak_to_peak: i32,
    /// Signal energy: mean of squared values, scaled to avoid overflow
    /// Computed as sum(x²) / N, divided by 16 for range compression.
    pub energy: i32,
}

/// Extract the 5 ECG features from a window of samples.
///
/// All computation uses integer arithmetic (i64 intermediate for safety).
///
/// # Arguments
/// * `window` - Slice of filtered ADC values
///
/// # Returns
/// `EcgFeatures` struct with all computed features.
fn extract_features(window: &[i32]) -> EcgFeatures {
    if window.is_empty() {
        return EcgFeatures {
            mean: 0, maximum: 0, minimum: 0, peak_to_peak: 0, energy: 0,
        };
    }

    let n = window.len();

    // Compute mean using i64 accumulator
    let mut sum: i64 = 0;
    let mut max_val: i32 = i32::MIN;
    let mut min_val: i32 = i32::MAX;
    let mut energy_acc: i64 = 0;

    for &sample in window.iter() {
        sum += sample as i64;

        if sample > max_val { max_val = sample; }
        if sample < min_val { min_val = sample; }

        // Energy: accumulate squared values (using i64 to prevent overflow)
        // Max: 4095² × 128 = 2,148,007,680 → fits in i64 (max ~9.2 × 10^18)
        energy_acc += (sample as i64) * (sample as i64);
    }

    let mean        = (sum / n as i64) as i32;
    let peak_to_peak = max_val - min_val;
    // Scale energy by dividing by N and then by 1024 to bring into i32 range
    // At max: 2,148,007,680 / 128 = 16,781,310 → fits comfortably in i32
    let energy      = (energy_acc / n as i64) as i32;

    EcgFeatures {
        mean,
        maximum: max_val,
        minimum: min_val,
        peak_to_peak,
        energy,
    }
}

// ─── Mode A: Threshold Classifier ────────────────────────────────────────────

/// Threshold-based ECG abnormality classifier.
///
/// This is the simplest possible classifier. Three rules:
///
/// Rule 1: HIGH AMPLITUDE (erratic beats / artifact)
///   peak_to_peak > THRESH_P2P → Abnormal
///   A very high amplitude suggests erratic heartbeat or motion artifact.
///
/// Rule 2: ELEVATED MEAN (ST elevation simulation)
///   mean > THRESH_MEAN_HIGH → Abnormal
///   Elevated baseline simulates ST elevation seen in cardiac events.
///
/// Rule 3: DEPRESSED MEAN (unusual baseline)
///   mean < THRESH_MEAN_LOW → Abnormal
///   Depressed baseline simulates unusual signal loss.
///
/// If none of the rules trigger → Normal.
///
/// # Tuning these thresholds:
/// Adjust the constants below based on your ECG signal characteristics.
/// The Python simulator uses the same thresholds in realtime_ecg_simulator.py.
///
/// # Arguments
/// * `features` - Extracted ECG features from the current window
///
/// # Returns
/// * `0` = Normal
/// * `1` = Abnormal

// Threshold parameters (tune these for your signal)
/// Peak-to-peak amplitude above this ADC count → Abnormal
/// At 250 Hz, 128 samples ≈ 512 ms window
/// Normal ECG at 2048 center ± ~600 ADC units (≈ ±0.48V at 12-bit/3.3V)
const THRESH_P2P: i32 = 600;

/// Mean ADC value above this → Elevated baseline (Abnormal)
/// ADC midpoint = 2048. Elevated mean ≈ 2048 + 300 = 2348
const THRESH_MEAN_HIGH: i32 = 2350;

/// Mean ADC value below this → Depressed baseline (Abnormal)
const THRESH_MEAN_LOW: i32 = 1750;

fn threshold_classify(features: &EcgFeatures) -> u8 {
    // Rule 1: High amplitude (irregular beats / noise spike)
    if features.peak_to_peak > THRESH_P2P {
        return 1; // Abnormal
    }

    // Rule 2: Elevated mean baseline (simulated ST elevation)
    if features.mean > THRESH_MEAN_HIGH {
        return 1; // Abnormal
    }

    // Rule 3: Depressed mean (unusual signal)
    if features.mean < THRESH_MEAN_LOW {
        return 1; // Abnormal
    }

    0 // Normal
}

// ─── Mode B: Quantized Tiny MLP ──────────────────────────────────────────────

/// Quantized 5→8→1 MLP inference using integer arithmetic.
///
/// Architecture:
///   Input  (5):  [mean, max, min, peak_to_peak, energy]
///   Hidden (8):  ReLU activation, int8 weights from W1/B1
///   Output (1):  Linear → threshold at 0 for binary decision
///
/// Quantization:
///   All weights stored as i8 (range -128 to +127).
///   Computation uses i32/i64 accumulators.
///   No floating point used in the hot path.
///
/// Input preparation:
///   Features are normalized to i8 range [-128, 127] by scaling
///   relative to the largest feature value.
///
/// Output decision:
///   output_q > 0 → Abnormal (1)
///   output_q ≤ 0 → Normal   (0)
///
/// Python equivalent:
///   python/realtime_ecg_simulator.py → mlp_inference_manual()
///
/// # Arguments
/// * `features` - Extracted ECG features
///
/// # Returns
/// * `0` = Normal
/// * `1` = Abnormal
fn mlp_infer(features: &EcgFeatures) -> u8 {
    // ── Step 1: Build feature vector ─────────────────────────────────────────
    // Feature array: [mean, max, min, peak_to_peak, energy]
    // We use the raw i32 values and quantize them to i8 below.
    let feat: [i32; 5] = [
        features.mean,
        features.maximum,
        features.minimum,
        features.peak_to_peak,
        // Energy is a large value (up to ~16M). Scale it down to ADC range.
        // Divide by 4096 to bring it roughly into 0–4095 range.
        (features.energy / 4096).max(-32767).min(32767),
    ];

    // ── Step 2: Normalize features to i8 range ────────────────────────────────
    // Find the max absolute value for symmetric scaling.
    let mut feat_max: i32 = 1; // Avoid divide-by-zero
    for &f in feat.iter() {
        let abs_f = f.abs();
        if abs_f > feat_max {
            feat_max = abs_f;
        }
    }

    // Scale each feature to [-127, +127]
    let mut feat_q: [i32; 5] = [0; 5];
    for i in 0..5 {
        let scaled = (feat[i] as i64 * 127i64) / feat_max as i64;
        feat_q[i] = scaled.max(-128).min(127) as i32;
    }

    // ── Step 3: Layer 1 forward pass ─────────────────────────────────────────
    // hidden[j] = ReLU( sum_i( W1[j][i] * feat_q[i] ) + B1[j] )
    //
    // W1 is stored row-major: W1[j * N_FEATURES + i]
    // model_weights::W1 has shape [N_HIDDEN × N_FEATURES] = [8 × 5] = 40 elements.
    //
    // Accumulate in i32 (int8 × int8 → int16, sum of 5 → max 5 × 127 × 127 = 80645, fits i32).
    let mut hidden: [i32; 8] = [0; 8];

    for j in 0..model_weights::N_HIDDEN {
        let mut acc: i32 = 0;
        for i in 0..model_weights::N_FEATURES {
            // W1 is stored as i8, convert to i32 for arithmetic
            let w: i32 = model_weights::W1[j * model_weights::N_FEATURES + i] as i32;
            acc += w * feat_q[i];
        }
        // Add bias (i32)
        acc += model_weights::B1[j];
        // ReLU activation: max(0, acc)
        hidden[j] = acc.max(0);
    }

    // ── Step 4: Re-quantize hidden layer to i8 ────────────────────────────────
    // Find max of hidden for scaling
    let mut h_max: i32 = 1;
    for &h in hidden.iter() {
        if h > h_max { h_max = h; }
    }

    let mut hidden_q: [i32; 8] = [0; 8];
    for j in 0..8 {
        let scaled = (hidden[j] as i64 * 127i64) / h_max as i64;
        hidden_q[j] = scaled.max(-128).min(127) as i32;
    }

    // ── Step 5: Layer 2 forward pass ──────────────────────────────────────────
    // output = sum_j( W2[j] * hidden_q[j] ) + B2[0]
    //
    // W2 shape: [1 × N_HIDDEN] = [1 × 8] = 8 elements
    let mut output_acc: i32 = 0;
    for j in 0..model_weights::N_HIDDEN {
        let w: i32 = model_weights::W2[j] as i32;
        output_acc += w * hidden_q[j];
    }
    output_acc += model_weights::B2[0];

    // ── Step 6: Binary decision ───────────────────────────────────────────────
    // Positive output → class 1 (Abnormal) [sigmoid > 0.5 equivalent]
    // Negative or zero → class 0 (Normal)
    if output_acc > 0 { 1 } else { 0 }
}
