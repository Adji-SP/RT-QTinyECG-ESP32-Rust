//! model_weights.rs
//! ==================
//! Quantized int8 weights and int32 biases for the ECG MLP classifier.
//!
//! Architecture:
//!   5 inputs → 8 hidden ReLU → 1 output threshold
//!
//! Layer shapes:
//!   W1 shape = [8, 5]
//!   B1 shape = [8]
//!   W2 shape = [1, 8]
//!   B2 shape = [1]
//!
//! Storage layout:
//!   W1 is stored row-major as [hidden_neuron][input_feature].
//!   Index formula:
//!     W1[j * N_FEATURES + i]
//!
//!   W2 is stored row-major as [output_neuron][hidden_neuron].
//!   Since there is only 1 output neuron, W2[j] is valid.
//!
//! To replace these weights with trained values:
//!   1. Run: python python/generate_dummy_ecg.py
//!   2. Run: python python/train_simple_model.py
//!   3. Run: python python/quantize_weights.py
//!   4. Run: python python/export_rust_weights.py
//!
//! Used by:
//!   src/inference.rs → mlp_infer()
//!
//! DISCLAIMER: These are educational placeholder/trained weights.
//! NOT for clinical medical use.

#![allow(dead_code)]

// ── Model Architecture Constants ──────────────────────────────────────────────

/// Number of input features:
/// [mean, maximum, minimum, peak_to_peak, energy_scaled]
pub const N_FEATURES: usize = 5;

/// Number of hidden neurons in Layer 1.
pub const N_HIDDEN: usize = 8;

/// Number of output neurons.
pub const N_OUTPUT: usize = 1;

// ── Trained Quantized Weights ─────────────────────────────────────────────────

/// Layer 1 weights W1.
///
/// Shape:
///   [N_HIDDEN x N_FEATURES] = [8 x 5]
///
/// Layout:
///   row-major, W1[j * N_FEATURES + i]
pub const W1: [i8; 40] = [
    -56,   61,   83,   51,  -20,
    -20,  -67,  105,  -11,   16,
    -67,  120,  113,  -10,  -44,
    -29,  -70,  -21,   29,   -6,
     67,  -22,  -23,   11,  -34,
     33,  -51,   14,   71,  -44,
     37,  -36, -115,   60,  127,
     91,    8,  -31,   52,   24,
];

/// Layer 1 biases B1.
///
/// Shape:
///   [N_HIDDEN] = [8]
pub const B1: [i32; 8] = [
    -39, -28, -57,  93,
    -14, -15, -53,   8,
];

/// Layer 2 weights W2.
///
/// Shape:
///   [N_OUTPUT x N_HIDDEN] = [1 x 8]
pub const W2: [i8; 8] = [
    -36, -39, 127, 75,
    126,  40,   8, 78,
];

/// Layer 2 bias B2.
///
/// Shape:
///   [N_OUTPUT] = [1]
pub const B2: [i32; 1] = [
    -128,
];

// ── Quantization Metadata ─────────────────────────────────────────────────────
//
// These scales are retained for documentation and future dequantized debugging.
// The current embedded inference path uses integer-only arithmetic and does not
// use floating point scales in the hot path.

/// Layer 1 weight dequantization scale.
pub const W1_SCALE: f32 = 0.00712235;

/// Layer 1 bias dequantization scale.
pub const B1_SCALE: f32 = 0.00712235;

/// Layer 2 weight dequantization scale.
pub const W2_SCALE: f32 = 0.00924780;

/// Layer 2 bias dequantization scale.
pub const B2_SCALE: f32 = 0.00924780;