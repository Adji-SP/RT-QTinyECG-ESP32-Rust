"""
export_rust_weights.py
======================
Exports quantized int8 weights from data/quantized_weights.npz
into a Rust source file:

    firmware/esp32-rust/src/model_weights.rs

The generated Rust file contains:
  - `const` arrays of int8 weights and int32 biases
  - Scale factors as float constants
  - Comments explaining how to use them in inference.rs

This replaces the placeholder weights in model_weights.rs with
real trained + quantized values.

Usage:
    python python/export_rust_weights.py
    (requires data/quantized_weights.npz from quantize_weights.py)

After running:
    Rebuild firmware with: cargo build --release
    (from firmware/esp32-rust/)
"""

import os
import textwrap
import numpy as np

INPUT_NPZ   = os.path.join(os.path.dirname(__file__), "..", "data", "quantized_weights.npz")
OUTPUT_RS   = os.path.join(os.path.dirname(__file__), "..", "firmware", "esp32-rust", "src", "model_weights.rs")


# ─── Rust Array Formatter ─────────────────────────────────────────────────────

def format_i8_array(name: str, arr: np.ndarray, comment: str = "") -> str:
    """Format a numpy int8 array as a Rust const array declaration."""
    flat    = arr.flatten().tolist()
    n       = len(flat)
    # Format as rows of 8 values each
    chunks  = [flat[i:i+8] for i in range(0, n, 8)]
    rows    = []
    for chunk in chunks:
        rows.append("    " + ", ".join(f"{v:4d}" for v in chunk) + ",")
    inner   = "\n".join(rows)
    cmt     = f"    // {comment}\n" if comment else ""
    return (
        f"/// {comment}\n"
        f"pub const {name}: [i8; {n}] = [\n"
        f"{cmt}"
        f"{inner}\n"
        f"];\n"
    )


def format_i32_array(name: str, arr: np.ndarray, comment: str = "") -> str:
    """Format a numpy int32 array as a Rust const array declaration."""
    flat    = arr.flatten().tolist()
    n       = len(flat)
    chunks  = [flat[i:i+4] for i in range(0, n, 4)]
    rows    = []
    for chunk in chunks:
        rows.append("    " + ", ".join(f"{int(v):8d}" for v in chunk) + ",")
    inner   = "\n".join(rows)
    cmt     = f"    // {comment}\n" if comment else ""
    return (
        f"/// {comment}\n"
        f"pub const {name}: [i32; {n}] = [\n"
        f"{cmt}"
        f"{inner}\n"
        f"];\n"
    )


def format_f32_const(name: str, value: float, comment: str = "") -> str:
    """Format a float constant as a Rust pub const."""
    return f"/// {comment}\npub const {name}: f32 = {value:.8f};\n"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[export_rust_weights] Loading quantized weights...")

    if not os.path.exists(INPUT_NPZ):
        print(f"ERROR: {INPUT_NPZ} not found. Run quantize_weights.py first.")
        print("       Using placeholder weights instead.")
        use_placeholder = True
    else:
        use_placeholder = False
        data = np.load(INPUT_NPZ)
        print(f"  Found keys: {list(data.keys())}")

    # Build the Rust file content
    lines = []

    lines.append(textwrap.dedent("""\
        //! model_weights.rs
        //! ==================
        //! Quantized int8 weights and int32 biases for the ECG MLP classifier.
        //!
        //! Architecture: 5 inputs → 8 hidden (ReLU) → 1 output (threshold)
        //!   - Layer 1: W1 shape [8, 5] (8 neurons, 5 inputs each)
        //!              b1 shape [8]
        //!   - Layer 2: W2 shape [1, 8] (1 output, 8 inputs)
        //!              b2 shape [1]
        //!
        //! Quantization:
        //!   Symmetric per-layer: scale_i = max(|W_i|) / 127.0
        //!   W_q = clip(round(W / scale), -128, 127)  → stored as i8
        //!   b_q = clip(round(b / (scale * input_scale)), ...)  → stored as i32
        //!
        //! To replace these weights with trained values:
        //!   1. Run: python python/generate_dummy_ecg.py
        //!   2. Run: python python/train_simple_model.py
        //!   3. Run: python python/quantize_weights.py
        //!   4. Run: python python/export_rust_weights.py
        //!   This file will be automatically regenerated.
        //!
        //! Used by: src/inference.rs  (mlp_infer() function)
        //!
        //! DISCLAIMER: These are educational placeholder/trained weights.
        //! NOT for clinical medical use.

        #![allow(dead_code)]

        // ── Model Architecture Constants ──────────────────────────────────────────────

        /// Number of input features
        pub const N_FEATURES: usize = 5;

        /// Number of hidden neurons in Layer 1
        pub const N_HIDDEN: usize = 8;

        /// Number of output neurons
        pub const N_OUTPUT: usize = 1;

    """))

    if use_placeholder:
        # Provide clearly labeled placeholder weights
        lines.append(textwrap.dedent("""\
            // ── PLACEHOLDER WEIGHTS ───────────────────────────────────────────────────────
            // These are placeholder int8 weights for compilation and testing.
            // Replace by running: python python/export_rust_weights.py
            //
            // Layer 1 weight matrix: shape [N_HIDDEN, N_FEATURES] = [8, 5]
            // Stored in row-major order: W1[neuron_idx][feature_idx]

            /// Layer 1 weights W1 [8 x 5] = 40 values, stored row-major
            pub const W1: [i8; 40] = [
                // Neuron 0: responds to mean, max, min, p2p, energy
                  10,  20, -10,  30,   5,
                // Neuron 1
                  15,  -5,  25,  10,   8,
                // Neuron 2
                  -8,  12,  18, -15,  20,
                // Neuron 3
                   5,  -3,  10,  40,  -5,
                // Neuron 4
                  22,   7, -12,   8,  15,
                // Neuron 5
                  -5,  30,   5, -10,  12,
                // Neuron 6
                  18,  -8,  20,  15,  -6,
                // Neuron 7
                  -3,  10, -20,  25,  18,
            ];

            /// Layer 1 biases b1 [8 values]
            pub const B1: [i32; 8] = [
                5, -3, 8, -10, 2, 15, -7, 4,
            ];

            /// Layer 2 weights W2 [1 x 8] = 8 values
            pub const W2: [i8; 8] = [
                20, -15, 18, 25, -10, 12, -8, 22,
            ];

            /// Layer 2 bias b2 [1 value]
            pub const B2: [i32; 1] = [
                -30,
            ];

            /// Layer 1 weight dequantization scale
            pub const W1_SCALE: f32 = 0.00787402;  // 1.0 / 127.0

            /// Layer 1 bias dequantization scale
            pub const B1_SCALE: f32 = 0.00787402;

            /// Layer 2 weight dequantization scale
            pub const W2_SCALE: f32 = 0.00787402;

            /// Layer 2 bias dequantization scale
            pub const B2_SCALE: f32 = 0.00787402;

        """))
    else:
        # Export real quantized weights from NPZ
        keys = list(data.keys())

        # Try to find Layer 1 and Layer 2 weights
        # Expected keys: W_q_0, b_q_0, w_scale_0, b_scale_0, W_q_1, b_q_1, ...
        try:
            W1_q = data["W_q_0"]
            b1_q = data["b_q_0"]
            w1_scale = float(data["w_scale_0"][0])
            b1_scale = float(data["b_scale_0"][0])
        except KeyError:
            print("  WARNING: Could not find Layer 1 weights in NPZ. Using zeros.")
            W1_q = np.zeros((8, 5), dtype=np.int8)
            b1_q = np.zeros(8, dtype=np.int32)
            w1_scale = 1.0 / 127.0
            b1_scale = 1.0 / 127.0

        try:
            W2_q = data["W_q_1"]
            b2_q = data["b_q_1"]
            w2_scale = float(data["w_scale_1"][0])
            b2_scale = float(data["b_scale_1"][0])
        except KeyError:
            print("  WARNING: Could not find Layer 2 weights in NPZ. Using zeros.")
            W2_q = np.zeros((1, 8), dtype=np.int8)
            b2_q = np.zeros(1, dtype=np.int32)
            w2_scale = 1.0 / 127.0
            b2_scale = 1.0 / 127.0

        n_w1 = W1_q.size
        n_b1 = b1_q.size
        n_w2 = W2_q.size
        n_b2 = b2_q.size

        lines.append(f"// ── TRAINED QUANTIZED WEIGHTS (generated by export_rust_weights.py) ─────────\n\n")
        lines.append(format_i8_array( f"W1", W1_q,   f"Layer 1 weights W1 [{W1_q.shape[0]} x {W1_q.shape[1]}], row-major"))
        lines.append("\n")
        lines.append(format_i32_array(f"B1", b1_q,   f"Layer 1 biases b1 [{b1_q.size}]"))
        lines.append("\n")
        lines.append(format_i8_array( f"W2", W2_q,   f"Layer 2 weights W2 [{W2_q.shape[0]} x {W2_q.shape[1]}], row-major"))
        lines.append("\n")
        lines.append(format_i32_array(f"B2", b2_q,   f"Layer 2 bias b2 [{b2_q.size}]"))
        lines.append("\n")
        lines.append(format_f32_const("W1_SCALE", w1_scale, "Layer 1 weight dequantization scale"))
        lines.append(format_f32_const("B1_SCALE", b1_scale, "Layer 1 bias dequantization scale"))
        lines.append(format_f32_const("W2_SCALE", w2_scale, "Layer 2 weight dequantization scale"))
        lines.append(format_f32_const("B2_SCALE", b2_scale, "Layer 2 bias dequantization scale"))

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_RS), exist_ok=True)
    content = "".join(lines)
    with open(OUTPUT_RS, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[export_rust_weights] Written to: {OUTPUT_RS}")
    print("[export_rust_weights] Done. Rebuild firmware with: cargo build --release")


if __name__ == "__main__":
    main()
