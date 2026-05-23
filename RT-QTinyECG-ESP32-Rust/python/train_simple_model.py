"""
train_simple_model.py
=====================
Trains a small classifier on the synthetic ECG feature dataset.

Two model options (controlled by MODEL_TYPE):
  - "logistic"  : Logistic Regression (simplest, fastest)
  - "mlp"       : Tiny Multi-Layer Perceptron (5 → 8 → 1)

Input features (per 128-sample window):
  [mean, maximum, minimum, peak_to_peak, energy]

Output:
  0 = Normal
  1 = Abnormal

The model is intentionally very small so it can be quantized and
manually deployed on ESP32 without TensorFlow Lite.

No GPU required. Runs on CPU only.

Usage:
    python python/train_simple_model.py

Saves:
    - Trained model (sklearn pickle): data/model.pkl
    - Model metrics printed to console

Then run:
    python python/quantize_weights.py
    python python/export_rust_weights.py
"""

import os
import csv
import pickle
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import moving_average, extract_features_array, sliding_windows

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_TYPE   = "mlp"           # "logistic" or "mlp"
WINDOW_SIZE  = 128             # Samples per window (matches firmware)
WINDOW_STEP  = 64              # Step size (50% overlap)
FILTER_SIZE  = 8               # Moving average window size
RANDOM_SEED  = 42

INPUT_CSV  = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecg.csv")
MODEL_OUT  = os.path.join(os.path.dirname(__file__), "..", "data", "model.pkl")
SCALER_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "scaler.pkl")


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_ecg_csv(path: str):
    """Load sample_ecg.csv, return (timestamps, adc_values, labels) arrays."""
    timestamps = []
    adc_values = []
    labels     = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row["time_ms"]))
            adc_values.append(int(row["adc_value"]))
            labels.append(int(row["label"]))
    return (
        np.array(timestamps, dtype=np.int32),
        np.array(adc_values, dtype=np.int32),
        np.array(labels, dtype=np.int32),
    )


# ─── Feature Dataset Builder ──────────────────────────────────────────────────

def build_feature_dataset(adc_signal: np.ndarray, labels: np.ndarray,
                           window_size: int, step: int, filter_size: int):
    """
    Build (X, y) feature dataset from sliding windows over the ECG signal.

    1. Apply moving average filter to the full signal
    2. Slide a window of window_size samples with given step
    3. For each window, extract 5 features
    4. Assign label = majority label within that window

    Returns:
        X : float64 array of shape (n_windows, 5)
        y : int32 array of shape (n_windows,)
    """
    # Step 1: Filter the signal
    filtered = moving_average(adc_signal, filter_size)

    X_list = []
    y_list = []

    # Step 2: Sliding windows
    for start_idx, window in sliding_windows(filtered, window_size, step):
        # Step 3: Extract features
        features = extract_features_array(window)
        X_list.append(features)

        # Step 4: Window label = majority label in the window
        window_labels = labels[start_idx : start_idx + window_size]
        majority_label = int(np.round(np.mean(window_labels)))
        y_list.append(majority_label)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ─── Model Training ───────────────────────────────────────────────────────────

def train_logistic(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        C=1.0,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    """
    Train a tiny MLP classifier.
    Architecture: 5 → 8 → 1  (binary output)
    Uses sklearn's MLPClassifier with one hidden layer of 8 neurons.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(8,),       # One hidden layer, 8 neurons
        activation="relu",             # ReLU activation (matches Rust firmware)
        solver="adam",
        max_iter=500,
        random_state=RANDOM_SEED,
        learning_rate_init=0.001,
        alpha=0.001,                   # L2 regularization
    )
    model.fit(X_train, y_train)
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[train_simple_model] Loading ECG data...")

    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run generate_dummy_ecg.py first.")
        return

    timestamps, adc_values, labels = load_ecg_csv(INPUT_CSV)
    print(f"  Loaded {len(adc_values)} samples")
    print(f"  Normal   samples: {(labels == 0).sum()}")
    print(f"  Abnormal samples: {(labels == 1).sum()}")

    # Build windowed feature dataset
    print(f"\n[train_simple_model] Building feature dataset...")
    print(f"  Window size : {WINDOW_SIZE} samples ({WINDOW_SIZE * 4} ms @ 250 Hz)")
    print(f"  Window step : {WINDOW_STEP} samples")
    print(f"  Filter size : {FILTER_SIZE} samples")

    X, y = build_feature_dataset(adc_values, labels, WINDOW_SIZE, WINDOW_STEP, FILTER_SIZE)
    print(f"  Feature dataset shape: {X.shape}")
    print(f"  Label distribution  : Normal={( y==0).sum()}, Abnormal={(y==1).sum()}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Feature scaling (important for MLP and logistic regression)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train the model
    print(f"\n[train_simple_model] Training {MODEL_TYPE} model...")
    if MODEL_TYPE == "logistic":
        model = train_logistic(X_train_sc, y_train)
        print("  Model type: Logistic Regression")
    else:
        model = train_mlp(X_train_sc, y_train)
        print(f"  Model type: MLP ({model.hidden_layer_sizes} hidden units)")

    # Evaluate
    y_pred = model.predict(X_test_sc)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n── Training Results ────────────────────────────")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print weights for MLP (for quantization)
    if MODEL_TYPE == "mlp":
        print(f"\n── MLP Weights ─────────────────────────────────")
        for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
            print(f"  Layer {i+1}: W shape={W.shape}, b shape={b.shape}")
            print(f"    W range: [{W.min():.4f}, {W.max():.4f}]")
            print(f"    b range: [{b.min():.4f}, {b.max():.4f}]")

    # Save model and scaler
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[train_simple_model] Model saved to : {MODEL_OUT}")
    print(f"[train_simple_model] Scaler saved to: {SCALER_OUT}")
    print("[train_simple_model] Done. Run quantize_weights.py next.")


if __name__ == "__main__":
    main()
