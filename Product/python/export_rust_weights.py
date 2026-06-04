"""
export_rust_weights.py
======================
Exports quantized int8 weights from data/quantized_weights.npz
into a Rust source file:

    firmware/esp32-rust/src/model_weights.rs

The generated Rust file contains:
  - `const` arrays of int8 weights and int32 biases
  - Architecture constants (N_FEATURES, N_HIDDEN, N_OUTPUT)

NOTE: Scale constants (W1_SCALE etc.) are intentionally NOT exported.
  The firmware uses pure integer-only inference and never dereferences
  those floats. Keeping them would mislead readers into thinking the
  firmware dequantizes weights at runtime (it does not).
  If you need them for off-device validation, add them manually.

This replaces the placeholder weights in model_weights.rs with
real trained + quantized values.

Usage:
    python python/export_rust_weights.py
    (requires data/quantized_weights.npz from quantize_weights.py)

    If quantized_weights.npz is missing, this script ABORTS with an error.
    Do NOT use placeholder weights in production — they produce wrong predictions.

After running:
    Rebuild firmware with: cargo build --release
    (from firmware/esp32-rust/)
"""

import os
import sys
import textwrap
import numpy as np

INPUT_NPZ     = os.path.join(os.path.dirname(__file__), "..", "data", "quantized_weights.npz")
FEAT_MEAN_NPY = os.path.join(os.path.dirname(__file__), "..", "data", "feat_mean.npy")
FEAT_STD_NPY  = os.path.join(os.path.dirname(__file__), "..", "data", "feat_std.npy")
OUTPUT_RS     = os.path.join(os.path.dirname(__file__), "..", "firmware", "esp32-rust", "src", "model_weights.rs")


# ─── Rust Array Formatter ─────────────────────────────────────────────────────

def format_i8_array(name: str, arr: np.ndarray, comment: str = "",
                   row_width: int = 8) -> str:
    """Format a numpy int8 array as a Rust const array declaration.

    Args:
        name      : Rust constant name
        arr       : numpy int8 array
        comment   : doc-comment string
        row_width : number of values per visual row (use N_FEATURES for W1,
                    8 for W2, etc., to align with the logical matrix layout)
    """
    flat    = arr.flatten().tolist()
    n       = len(flat)
    # Format as rows of row_width values each
    chunks  = [flat[i:i+row_width] for i in range(0, n, row_width)]
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


def format_f32_array(name: str, arr: np.ndarray, comment: str = "") -> str:
    """Format a float array as a Rust pub const array."""
    vals = ", ".join(f"{float(v):.6f}" for v in arr.flat)
    return (
        f"/// {comment}\n"
        f"pub const {name}: [f32; {arr.size}] = [{vals}];\n"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[export_rust_weights] Loading quantized weights...")

    if not os.path.exists(INPUT_NPZ):
        print(f"ERROR: {INPUT_NPZ} not found.")
        print("       Run quantize_weights.py first before exporting to Rust.")
        print("       Aborting — do NOT use placeholder weights in firmware.")
        sys.exit(1)

    data = np.load(INPUT_NPZ)
    print(f"  Found keys: {list(data.keys())}")

    # Build the Rust file content
    lines = []

    lines.append(textwrap.dedent("""\
        //! model_weights.rs
        //! ==================
        //! Quantized int8 weights, int32 biases, and feature normalization constants
        //! for the ECG MLP classifier.
        //!
        //! Architecture: 5 inputs -> 8 hidden (ReLU) -> 1 output (threshold)
        //!   - Layer 1: W1 shape [N_HIDDEN x N_FEATURES] = [8 x 5], row-major
        //!              B1 shape [N_HIDDEN] = [8]
        //!   - Layer 2: W2 shape [N_OUTPUT x N_HIDDEN] = [1 x 8], row-major
        //!              B2 shape [N_OUTPUT] = [1]
        //!
        //! Feature Normalization (CRITICAL for correctness):
        //!   Raw features = [mean, max, min, peak_to_peak, centered_energy/4096]
        //!   Normalized   = (raw - FEAT_MEAN) / FEAT_STD   (z-score)
        //!   FEAT_MEAN and FEAT_STD are computed on the training set by StandardScaler
        //!   in train_simple_model.py and exported here.
        //!   The firmware MUST apply this normalization before MLP inference.
        //!   This preserves amplitude information across windows (unlike per-window
        //!   normalization which destroys it).
        //!
        //! Quantization:
        //!   Symmetric per-layer: scale_i = max(|W_i|) / 127.0
        //!   W_q = clip(round(W / scale), -128, 127)  -> stored as i8
        //!   b_q = clip(round(b / (scale * 127.0)), ...)  -> stored as i32
        //!         Note: input_scale = 127.0 because firmware inputs are clipped to [-128, 127]
        //!         after z-score normalization (see mlp_infer() in inference.rs).
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

        // -- Model Architecture Constants --

        /// Number of input features
        pub const N_FEATURES: usize = 5;

        /// Number of hidden neurons in Layer 1
        pub const N_HIDDEN: usize = 8;

        /// Number of output neurons
        pub const N_OUTPUT: usize = 1;

    """))

    # Build the real quantized weight arrays.
    # (No placeholder fallback — this script aborts if NPZ is missing.)

    # Try to find Layer 1 and Layer 2 weights.
    # Expected keys: W_q_0, b_q_0, W_q_1, b_q_1
    try:
        W1_q = data["W_q_0"]
        b1_q = data["b_q_0"]
        # sklearn stores first-layer weights as [features, hidden].
        # Firmware reads row-major [hidden, features].
        if W1_q.ndim == 2 and W1_q.shape[0] == 5:
            W1_q = W1_q.T
    except KeyError:
        print("  ERROR: Could not find Layer 1 weights (W_q_0 / b_q_0) in NPZ.")
        sys.exit(1)

    try:
        W2_q = data["W_q_1"]
        b2_q = data["b_q_1"]
        # sklearn stores second-layer weights as [hidden, output].
        # Firmware stores one output row as [1, hidden].
        if W2_q.ndim == 2 and W2_q.shape[1] == 1:
            W2_q = W2_q.T
    except KeyError:
        print("  ERROR: Could not find Layer 2 weights (W_q_1 / b_q_1) in NPZ.")
        sys.exit(1)

    lines.append(f"// -- TRAINED QUANTIZED WEIGHTS (generated by export_rust_weights.py) --\n\n")
    # W1: format in rows of N_FEATURES (5) so each visual row = one neuron
    lines.append(format_i8_array(
        "W1", W1_q,
        f"Layer 1 weights W1 [{W1_q.shape[0]} x {W1_q.shape[1]}], row-major (each row = one neuron)",
        row_width=W1_q.shape[1] if W1_q.ndim == 2 else 5,
    ))
    lines.append("\n")
    lines.append(format_i32_array("B1", b1_q, f"Layer 1 biases B1 [{b1_q.size}]"))
    lines.append("\n")
    lines.append(format_i8_array(
        "W2", W2_q,
        f"Layer 2 weights W2 [{W2_q.shape[0]} x {W2_q.shape[1]}], row-major",
        row_width=W2_q.size,
    ))
    lines.append("\n")
    lines.append(format_i32_array("B2", b2_q, f"Layer 2 bias B2 [{b2_q.size}]"))

    # Load and export feature normalization constants.
    # These are required for the firmware to apply the same z-score normalization
    # that StandardScaler applied during training.
    if os.path.exists(FEAT_MEAN_NPY) and os.path.exists(FEAT_STD_NPY):
        feat_mean = np.load(FEAT_MEAN_NPY).astype(np.float32)
        feat_std  = np.load(FEAT_STD_NPY).astype(np.float32)
        lines.append("\n// -- Feature Normalization Constants (from StandardScaler in train_simple_model.py) --\n")
        lines.append(format_f32_array(
            "FEAT_MEAN", feat_mean,
            "Per-feature training mean for z-score normalization. "
            "Apply: norm = (raw - FEAT_MEAN) / FEAT_STD before MLP inference."
        ))
        lines.append("\n")
        lines.append(format_f32_array(
            "FEAT_STD", feat_std,
            "Per-feature training std for z-score normalization."
        ))
        print(f"  FEAT_MEAN: {feat_mean.round(2)}")
        print(f"  FEAT_STD : {feat_std.round(2)}")
    else:
        print("  WARNING: feat_mean.npy / feat_std.npy not found. FEAT_MEAN/FEAT_STD not exported.")
        print("  Run: python python/train_simple_model.py to generate them.")

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_RS), exist_ok=True)
    content = "".join(lines)
    with open(OUTPUT_RS, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[export_rust_weights] Written to: {OUTPUT_RS}")
    print("[export_rust_weights] Done. Rebuild firmware with: cargo build --release")


if __name__ == "__main__":
    main()
