"""
quantize_weights.py
====================
Quantizes float32 model weights to int8 for firmware deployment.

Loads the trained model from data/model.pkl and quantizes:
  - Weight matrices (float32 → int8, per-layer symmetric quantization)
  - Bias vectors (float32 → int32, using same scale)

Shows:
  - Original size (float32)
  - Quantized size (int8)
  - Compression ratio
  - Max quantization error per layer

Quantization method:
  Symmetric per-layer linear quantization.
  scale_i = max(|W_i|) / 127.0
  W_q_i   = round(W_i / scale_i)  clamped to [-128, 127]

This is the simplest quantization scheme.
More advanced schemes (per-channel, asymmetric) can be added later.

Usage:
    python python/quantize_weights.py
    (requires data/model.pkl from train_simple_model.py)

Saves:
    data/quantized_weights.npz

Then run:
    python python/export_rust_weights.py
"""

import os
import pickle
import numpy as np

INPUT_MODEL  = os.path.join(os.path.dirname(__file__), "..", "data", "model.pkl")
OUTPUT_NPZ   = os.path.join(os.path.dirname(__file__), "..", "data", "quantized_weights.npz")


# ─── Quantization Functions ───────────────────────────────────────────────────

def quantize_layer_symmetric(weights_f32: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Symmetric per-layer quantization of a weight matrix.

    Maps float32 weights to int8 using:
        scale = max(|W|) / 127.0
        W_q   = clip(round(W / scale), -128, 127)

    Args:
        weights_f32: float32 weight matrix of any shape

    Returns:
        (weights_int8, scale):
            weights_int8 : int8 array (same shape as input)
            scale        : float32 scale factor for dequantization
    """
    w_max = np.max(np.abs(weights_f32))
    if w_max == 0.0:
        # All-zero weights: scale doesn't matter
        return np.zeros_like(weights_f32, dtype=np.int8), 1.0

    scale     = w_max / 127.0
    weights_q = np.clip(np.round(weights_f32 / scale), -128, 127).astype(np.int8)
    return weights_q, float(scale)


def quantize_bias_int32(bias_f32: np.ndarray, weight_scale: float,
                         input_scale: float = 42.333) -> tuple[np.ndarray, float]:
    """
    Quantize bias to int32.

    Biases are quantized with the combined scale (weight_scale * input_scale).

    The firmware accumulates:  acc += W_q[i] * feat_q[i]
    where feat_q = clip(z_score * 42.333, -128, 127).
    42.333 = 127 / 3  (maps ±3 sigma of z-score → ±127).

    So the accumulator W_q · feat_q runs at scale:  weight_scale * 42.333
    The bias must be quantized at the same scale to maintain the correct
    decision boundary.

    Using 127.0 here (old value for raw [-128,127] int inputs) would make
    the bias ~3× too large, shifting the boundary and breaking inference.

    Args:
        bias_f32     : float32 bias vector
        weight_scale : Scale of the weight matrix for this layer
        input_scale  : Effective scale of feat_q per unit of sklearn input.
                       42.333 for z-score inputs scaled by 42.333 in firmware.

    Returns:
        (bias_int32, bias_scale)
    """
    bias_scale = weight_scale * input_scale
    if bias_scale == 0.0:
        bias_scale = 1.0
    bias_q = np.clip(np.round(bias_f32 / bias_scale), -2147483648, 2147483647).astype(np.int32)
    return bias_q, float(bias_scale)


def quantization_error(original: np.ndarray, quantized: np.ndarray, scale: float) -> dict:
    """
    Compute quantization error statistics.

    Args:
        original : Original float32 weights
        quantized: int8/int32 quantized weights
        scale    : Scale factor used

    Returns:
        dict with max_error, mean_error, relative_error
    """
    dequantized = quantized.astype(np.float64) * scale
    error       = np.abs(original.astype(np.float64) - dequantized)
    return {
        "max_error"     : float(error.max()),
        "mean_error"    : float(error.mean()),
        "relative_error": float(error.mean() / (np.abs(original).mean() + 1e-9)),
    }


# ─── MLP Weight Extraction ────────────────────────────────────────────────────

def extract_mlp_weights(model) -> list[dict]:
    """
    Extract weight matrices and biases from an sklearn MLPClassifier.

    Returns a list of dicts, one per layer:
        [{"W": float32_array, "b": float32_array}, ...]
    """
    layers = []
    for W, b in zip(model.coefs_, model.intercepts_):
        layers.append({
            "W": W.astype(np.float32),
            "b": b.astype(np.float32),
        })
    return layers


def extract_logistic_weights(model) -> list[dict]:
    """
    Extract weight matrix and bias from an sklearn LogisticRegression.

    For binary classification, coef_ shape is (1, n_features).
    Returns a single layer.
    """
    W = model.coef_.astype(np.float32)
    b = model.intercept_.astype(np.float32)
    return [{"W": W, "b": b}]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[quantize_weights] Loading trained model...")

    if not os.path.exists(INPUT_MODEL):
        print(f"ERROR: {INPUT_MODEL} not found. Run train_simple_model.py first.")
        return

    with open(INPUT_MODEL, "rb") as f:
        model = pickle.load(f)

    model_type = type(model).__name__
    print(f"  Model type: {model_type}")

    # Extract weights
    if hasattr(model, "coefs_"):
        # MLP
        layers = extract_mlp_weights(model)
        print(f"  MLP layers : {len(layers)}")
    elif hasattr(model, "coef_"):
        # Logistic Regression
        layers = extract_logistic_weights(model)
        print(f"  Logistic layers: {len(layers)}")
    else:
        print(f"ERROR: Unsupported model type: {model_type}")
        return

    # Quantize each layer
    print("\n[quantize_weights] Quantizing weights (float32 -> int8)...")

    quantized_layers = []
    total_f32_bytes  = 0
    total_q_bytes    = 0

    for i, layer in enumerate(layers):
        W_f32 = layer["W"]
        b_f32 = layer["b"]

        # Quantize weights.
        W_q, w_scale = quantize_layer_symmetric(W_f32)
        b_q, b_scale = quantize_bias_int32(b_f32, w_scale, input_scale=42.333)

        # Compute sizes
        f32_size = W_f32.nbytes + b_f32.nbytes
        q_size   = W_q.nbytes  + b_q.nbytes
        total_f32_bytes += f32_size
        total_q_bytes   += q_size

        # Quantization error
        err = quantization_error(W_f32, W_q, w_scale)

        quantized_layers.append({
            "W_q"    : W_q,
            "b_q"    : b_q,
            "w_scale": w_scale,
            "b_scale": b_scale,
            "f32_size": f32_size,
            "q_size"  : q_size,
            "err"     : err,
        })

    # Save as NPZ FIRST (before any print that might crash on Windows CP1252)
    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    save_dict = {}
    for i, ql in enumerate(quantized_layers):
        save_dict[f"W_q_{i}"]    = ql["W_q"]
        save_dict[f"b_q_{i}"]    = ql["b_q"]
        save_dict[f"w_scale_{i}"] = np.array([ql["w_scale"]])
        save_dict[f"b_scale_{i}"] = np.array([ql["b_scale"]])
    np.savez(OUTPUT_NPZ, **save_dict)
    print(f"\n[quantize_weights] Saved to: {OUTPUT_NPZ}")
    print("[quantize_weights] Done. Run export_rust_weights.py next.")


if __name__ == "__main__":
    main()
