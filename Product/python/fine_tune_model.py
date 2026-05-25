# python/fine_tune_model.py
# ==========================
# Fine-Tuning Loop: Retrain PC model on boundary/disagreement samples
#
# Augments the training dataset with samples where the PC model and ESP32
# disagreed, retrains the sklearn MLP, and re-exports quantized weights.
#
# Usage:
#   py python/fine_tune_model.py                    # basic retrain
#   py python/fine_tune_model.py --epochs 200       # more iterations
#   py python/fine_tune_model.py --extra-data data/disagreements.csv
#
# After running, rebuild firmware:
#   py python/export_rust_weights.py
#   cd firmware/esp32-rust && cargo build --release
#
# DISCLAIMER: Educational prototype only. Not for clinical use.

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.join(SCRIPT_DIR, "..")
DATA_DIR       = os.path.join(ROOT_DIR, "data")
ECG_CSV        = os.path.join(DATA_DIR, "sample_ecg.csv")
MODEL_PKL      = os.path.join(DATA_DIR, "model.pkl")
SCALER_PKL     = os.path.join(DATA_DIR, "scaler.pkl")
REPORT_CSV     = os.path.join(DATA_DIR, "comparison_report.csv")
FINETUNE_LOG   = os.path.join(DATA_DIR, "finetune_history.csv")

sys.path.insert(0, SCRIPT_DIR)
from preprocessing import extract_firmware_mlp_input_array

# ── Signal processing constants ────────────────────────────────────────────────
RING_BUF_SIZE  = 128
FILTER_WINDOW  = 8
SAMPLE_RATE    = 250


def parse_args():
    p = argparse.ArgumentParser(
        description="RT-QTinyECG Fine-Tuning: retrain on disagreement samples"
    )
    p.add_argument("--epochs",      default=200,   type=int,   help="Training max iterations")
    p.add_argument("--extra-data",  default=None,             help="Extra disagreement CSV to augment training")
    p.add_argument("--hidden",      default="8",              help="Hidden layers e.g. '8' or '8,8' or '16'")
    p.add_argument("--augment-factor", default=3, type=int,   help="How many times to repeat boundary samples")
    p.add_argument("--test-split",  default=0.2,  type=float, help="Test set fraction")
    p.add_argument("--seed",        default=42,   type=int,   help="Random seed")
    return p.parse_args()


# ── Moving average filter (mirrors firmware) ──────────────────────────────────
class MovingAverage:
    def __init__(self, window=FILTER_WINDOW):
        self.window = window
        self.buf = [0] * window
        self.idx = 0
        self.s = 0

    def push(self, val):
        self.s -= self.buf[self.idx]
        self.buf[self.idx] = val
        self.s += val
        self.idx = (self.idx + 1) % self.window
        return self.s // self.window


def extract_features_from_window(window):
    """Firmware-normalized MLP inputs, matching inference.rs."""
    return extract_firmware_mlp_input_array(np.array(window, dtype=np.int32)).tolist()


def build_dataset_from_csv(csv_path):
    """Convert ECG sample CSV into windowed feature vectors."""
    df = pd.read_csv(csv_path, comment="#")
    df.columns = df.columns.str.strip()

    adc_col = next((c for c in ["adc_raw", "adc_value", "value"] if c in df.columns), df.columns[1])
    lbl_col = next((c for c in ["label", "gt_label", "ground_truth"] if c in df.columns), None)
    if lbl_col is None:
        lbl_col = df.columns[-1]

    filt = MovingAverage(FILTER_WINDOW)
    ring_buf = []
    features, labels = [], []

    for _, row in df.iterrows():
        adc = int(row[adc_col])
        lbl = int(row[lbl_col])
        filtered = filt.push(adc)

        if len(ring_buf) < RING_BUF_SIZE:
            ring_buf.append(filtered)
        else:
            ring_buf.pop(0)
            ring_buf.append(filtered)

        if len(ring_buf) == RING_BUF_SIZE:
            features.append(extract_features_from_window(ring_buf))
            labels.append(lbl)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=int)


def build_dataset_from_disagreements(csv_path, ecg_csv_path):
    """
    Build augmented dataset from disagreement indices.
    Loads the original ECG CSV and extracts features for disagreed samples.
    """
    disag_df = pd.read_csv(csv_path)
    ecg_df   = pd.read_csv(ecg_csv_path, comment="#")
    ecg_df.columns = ecg_df.columns.str.strip()

    if "index" not in disag_df.columns:
        return np.array([]).reshape(0, 5), np.array([])

    adc_col = next((c for c in ["adc_raw", "adc_value", "value"] if c in ecg_df.columns), ecg_df.columns[1])
    lbl_col = next((c for c in ["label", "gt_label", "ground_truth"] if c in ecg_df.columns), ecg_df.columns[-1])

    indices = set(disag_df["index"].tolist())
    features, labels = [], []
    filt = MovingAverage(FILTER_WINDOW)
    ring_buf = []

    for i, row in ecg_df.iterrows():
        adc = int(row[adc_col])
        lbl = int(row[lbl_col])
        filtered = filt.push(adc)

        if len(ring_buf) < RING_BUF_SIZE:
            ring_buf.append(filtered)
        else:
            ring_buf.pop(0)
            ring_buf.append(filtered)

        if len(ring_buf) == RING_BUF_SIZE and i in indices:
            features.append(extract_features_from_window(ring_buf))
            labels.append(lbl)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=int)


def main():
    args = parse_args()

    # ── Parse hidden layer sizes ──────────────────────────────────────────────
    try:
        hidden = tuple(int(x) for x in args.hidden.split(","))
    except ValueError:
        print(f"[fine_tune] ERROR: Invalid --hidden '{args.hidden}'. Use '8' or '8,8'.")
        sys.exit(1)

    print(f"[fine_tune] Fine-Tuning Configuration:")
    print(f"  Architecture    : 5 → {' → '.join(str(h) for h in hidden)} → 1")
    print(f"  Max iterations  : {args.epochs}")
    print(f"  Augment factor  : {args.augment_factor}×")
    print()

    # ── Build base dataset ────────────────────────────────────────────────────
    if not os.path.exists(ECG_CSV):
        print(f"[fine_tune] ERROR: {ECG_CSV} not found. Run generate_dummy_ecg.py first.")
        sys.exit(1)

    print(f"[fine_tune] Building feature dataset from {ECG_CSV}...")
    X, y = build_dataset_from_csv(ECG_CSV)
    print(f"  Base samples    : {len(X)} (Normal: {(y==0).sum()}, Abnormal: {(y==1).sum()})")

    # ── Augment with disagreement samples ─────────────────────────────────────
    extra_data_path = args.extra_data
    if extra_data_path is None and os.path.exists(REPORT_CSV):
        # Auto-generate disagreements from comparison report
        report_df = pd.read_csv(REPORT_CSV)
        disag_mask = (
            (report_df["pc_prediction"] >= 0) &
            (report_df["esp32_prediction"] >= 0) &
            (report_df["pc_prediction"] != report_df["esp32_prediction"])
        )
        disag_df = report_df[disag_mask][["index", "gt_label", "pc_prediction", "esp32_prediction"]]
        if len(disag_df) > 0:
            auto_disag = os.path.join(DATA_DIR, "auto_disagreements.csv")
            disag_df.to_csv(auto_disag, index=False)
            extra_data_path = auto_disag
            print(f"  Auto-extracted {len(disag_df)} disagreement samples → {auto_disag}")

    if extra_data_path and os.path.exists(extra_data_path):
        print(f"[fine_tune] Loading extra/disagreement samples from {extra_data_path}...")
        X_extra, y_extra = build_dataset_from_disagreements(extra_data_path, ECG_CSV)
        if len(X_extra) > 0:
            # Repeat boundary samples augment_factor times
            X_rep = np.tile(X_extra, (args.augment_factor, 1))
            y_rep = np.tile(y_extra,  args.augment_factor)
            X = np.vstack([X, X_rep])
            y = np.concatenate([y, y_rep])
            print(f"  After augment   : {len(X)} samples (added {len(X_rep)} boundary repeats)")
        else:
            print("  (no disagreement features found — using base dataset only)")

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed, stratify=y
    )
    print(f"\n[fine_tune] Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Feature scaling ───────────────────────────────────────────────────────
    # ── Load old model for comparison ─────────────────────────────────────────
    old_acc = None
    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            old_model = pickle.load(f)
        if os.path.exists(SCALER_PKL):
            with open(SCALER_PKL, "rb") as f:
                old_scaler = pickle.load(f)
            old_input = old_scaler.transform(X_test) if old_scaler is not None else X_test
            old_preds = old_model.predict(old_input)
            old_acc = accuracy_score(y_test, old_preds)
            print(f"\n[fine_tune] Previous model test accuracy: {old_acc:.4f}")

    # ── Train new model ───────────────────────────────────────────────────────
    print(f"\n[fine_tune] Training new model...")
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=args.epochs,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    new_acc = accuracy_score(y_test, y_pred)

    print(f"\n[fine_tune] Fine-tuning results:")
    print(f"  New accuracy : {new_acc:.4f}  ({new_acc*100:.1f}%)")
    if old_acc is not None:
        delta = new_acc - old_acc
        print(f"  Improvement  : {delta:+.4f}  ({delta*100:+.2f}%)")
    print(f"  Iterations   : {model.n_iter_}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

    # ── Save updated model ────────────────────────────────────────────────────
    saved_model = False
    if old_acc is None or new_acc >= old_acc:
        with open(MODEL_PKL, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PKL, "wb") as f:
            pickle.dump(None, f)
        saved_model = True
        print(f"[fine_tune] Saved updated model to {MODEL_PKL}")
    else:
        print("[fine_tune] New model is worse than previous model; keeping existing model.pkl unchanged.")

    # ── Log history ───────────────────────────────────────────────────────────
    import datetime
    history_row = {
        "timestamp":   datetime.datetime.now().isoformat(),
        "architecture": str(hidden),
        "epochs":       model.n_iter_,
        "old_accuracy": old_acc if old_acc else "N/A",
        "new_accuracy": new_acc,
        "saved_model":  saved_model,
        "n_train":      len(X_train),
        "n_test":       len(X_test),
    }
    hist_df = pd.DataFrame([history_row])
    if os.path.exists(FINETUNE_LOG):
        existing = pd.read_csv(FINETUNE_LOG)
        hist_df = pd.concat([existing, hist_df], ignore_index=True)
    hist_df.to_csv(FINETUNE_LOG, index=False)
    print(f"[fine_tune] History logged to {FINETUNE_LOG}")

    # ── Next steps ────────────────────────────────────────────────────────────
    print(f"\n[fine_tune] Next steps:")
    print(f"  1. Re-quantize:  py python/quantize_weights.py")
    print(f"  2. Export Rust:  py python/export_rust_weights.py")
    print(f"  3. Rebuild:      cd firmware/esp32-rust && cargo build --release")
    print(f"  4. Reflash:      espflash flash target\\xtensa-esp32-none-elf\\release\\ecg-esp32")
    print(f"  5. Re-evaluate:  py python/uart_feed_evaluator.py --port COM3")
    print(f"  6. Compare:      py python/compare_models.py")


if __name__ == "__main__":
    main()
