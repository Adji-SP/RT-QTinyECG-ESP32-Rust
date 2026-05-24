# python/optimization_report.py
# ==============================
# Optimization Report Generator
#
# Reads the comparison_report.csv from compare_models.py,
# identifies optimization targets, and prints actionable recommendations
# based on the quantization delta, disagreement patterns, and metrics.
#
# Usage:
#   py python/optimization_report.py
#
# DISCLAIMER: Educational prototype only. Not for clinical use.

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.join(SCRIPT_DIR, "..")
DATA_DIR      = os.path.join(ROOT_DIR, "data")
REPORT_CSV    = os.path.join(DATA_DIR, "comparison_report.csv")
ESP32_CSV     = os.path.join(DATA_DIR, "esp32_predictions.csv")
ECG_CSV       = os.path.join(DATA_DIR, "sample_ecg.csv")
MODEL_PKL     = os.path.join(DATA_DIR, "model.pkl")
WEIGHTS_NPZ   = os.path.join(DATA_DIR, "quantized_weights.npz")
OPT_REPORT    = os.path.join(DATA_DIR, "optimization_targets.json")

# ── Thresholds for recommendations ────────────────────────────────────────────
GOOD_ACCURACY    = 0.90
ACCEPT_ACCURACY  = 0.85
MAX_QUANT_DELTA  = 0.05   # 5% max allowed quantization loss
GOOD_AGREE_RATE  = 0.95
GOOD_INF_US      = 100    # µs


def load_data():
    data = {}

    if os.path.exists(REPORT_CSV):
        data["report"] = pd.read_csv(REPORT_CSV)
    else:
        print(f"[optimization_report] WARNING: {REPORT_CSV} not found.")
        print("  → Run uart_feed_evaluator.py then compare_models.py first.")
        return None

    if os.path.exists(ESP32_CSV):
        data["esp32"] = pd.read_csv(ESP32_CSV)

    if os.path.exists(ECG_CSV):
        data["ecg"] = pd.read_csv(ECG_CSV, comment="#")

    return data


def compute_metrics(report_df):
    valid = report_df[
        (report_df["pc_prediction"] >= 0) &
        (report_df["esp32_prediction"] >= 0) &
        (report_df["gt_label"] >= 0)
    ].copy()

    if valid.empty:
        return None

    from sklearn.metrics import accuracy_score
    metrics = {
        "pc_accuracy":    accuracy_score(valid["gt_label"], valid["pc_prediction"]),
        "esp32_accuracy": accuracy_score(valid["gt_label"], valid["esp32_prediction"]),
        "n_samples":      len(valid),
        "n_disagreements": int((valid["pc_prediction"] != valid["esp32_prediction"]).sum()),
    }
    metrics["quant_delta"]    = metrics["pc_accuracy"] - metrics["esp32_accuracy"]
    metrics["agreement_rate"] = 1.0 - metrics["n_disagreements"] / len(valid)
    return metrics


def analyze_disagreements(report_df):
    disag = report_df[
        (report_df["pc_prediction"] >= 0) &
        (report_df["esp32_prediction"] >= 0) &
        (report_df["pc_prediction"] != report_df["esp32_prediction"])
    ].copy()

    if disag.empty:
        return {"n_total": 0}

    analysis = {
        "n_total":           len(disag),
        "n_on_normal":       int((disag["gt_label"] == 0).sum()),
        "n_on_abnormal":     int((disag["gt_label"] == 1).sum()),
        "pc_says_normal_esp32_says_abnormal":  int(
            ((disag["pc_prediction"] == 0) & (disag["esp32_prediction"] == 1)).sum()
        ),
        "pc_says_abnormal_esp32_says_normal":  int(
            ((disag["pc_prediction"] == 1) & (disag["esp32_prediction"] == 0)).sum()
        ),
    }
    return analysis


def get_model_info():
    info = {}
    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)
        info["type"] = type(model).__name__
        if hasattr(model, "hidden_layer_sizes"):
            info["architecture"] = f"5 → {' → '.join(str(s) for s in model.hidden_layer_sizes)} → 1"
        if hasattr(model, "coefs_"):
            total_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
            info["float32_params"] = total_params
            info["float32_bytes"]  = total_params * 4

    if os.path.exists(WEIGHTS_NPZ):
        npz = np.load(WEIGHTS_NPZ)
        int8_bytes = sum(a.nbytes for a in npz.values())
        info["int8_bytes"] = int8_bytes
        if "float32_bytes" in info:
            info["compression_ratio"] = info["float32_bytes"] / int8_bytes

    return info


def print_report(metrics, disag, model_info):
    SEP = "═" * 62
    print(f"\n{SEP}")
    print(f"  RT-QTinyECG — Optimization Report")
    print(SEP)

    # ── Current metrics ───────────────────────────────────────────────────────
    print("\n── 1. Current Performance ──────────────────────────────────────")
    print(f"  PC float32 accuracy  : {metrics['pc_accuracy']:.4f}  ({metrics['pc_accuracy']*100:.1f}%)")
    print(f"  ESP32 int8 accuracy  : {metrics['esp32_accuracy']:.4f}  ({metrics['esp32_accuracy']*100:.1f}%)")
    print(f"  Quantization delta   : {metrics['quant_delta']:+.4f}  ({metrics['quant_delta']*100:+.2f}%)")
    print(f"  Model agreement      : {metrics['agreement_rate']:.4f}  ({metrics['agreement_rate']*100:.1f}%)")

    # ── Model info ────────────────────────────────────────────────────────────
    if model_info:
        print("\n── 2. Model Details ────────────────────────────────────────────")
        if "architecture" in model_info:
            print(f"  Architecture         : {model_info['architecture']}")
        if "float32_params" in model_info:
            print(f"  float32 params       : {model_info['float32_params']}")
            print(f"  float32 size         : {model_info.get('float32_bytes', 0)} bytes")
        if "int8_bytes" in model_info:
            print(f"  int8 size            : {model_info['int8_bytes']} bytes")
        if "compression_ratio" in model_info:
            print(f"  Compression ratio    : {model_info['compression_ratio']:.1f}×")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    print("\n── 3. Disagreement Analysis ────────────────────────────────────")
    if disag["n_total"] == 0:
        print("  ✅ No disagreements — models are perfectly aligned!")
    else:
        print(f"  Total disagreements  : {disag['n_total']}")
        print(f"  On normal segments   : {disag['n_on_normal']}")
        print(f"  On abnormal segments : {disag['n_on_abnormal']}")
        print(f"  PC=Normal, ESP32=Abn : {disag['pc_says_normal_esp32_says_abnormal']} (ESP32 over-detecting)")
        print(f"  PC=Abnorm, ESP32=Nrm : {disag['pc_says_abnormal_esp32_says_normal']} (ESP32 under-detecting)")

    # ── Actionable recommendations ────────────────────────────────────────────
    recs = []
    priority = []

    print("\n── 4. Recommended Optimizations ───────────────────────────────")

    # Quant delta
    if metrics["quant_delta"] > MAX_QUANT_DELTA:
        r = {
            "priority": 1, "type": "FINE_TUNE",
            "description": f"Quantization delta {metrics['quant_delta']*100:.1f}% > 5% threshold",
            "action": "Run fine_tune_model.py with disagreement samples, re-quantize, reflash",
            "file": "python/fine_tune_model.py",
        }
        priority.append(r)
        print(f"\n  ⚠ [P1] FINE-TUNE REQUIRED")
        print(f"     Quantization delta {metrics['quant_delta']*100:.1f}% exceeds 5% threshold.")
        print(f"     Action: py python/fine_tune_model.py --epochs 100")

    # ESP32 accuracy
    if metrics["esp32_accuracy"] < ACCEPT_ACCURACY:
        r = {
            "priority": 2, "type": "THRESHOLD_ADJUST",
            "description": f"ESP32 accuracy {metrics['esp32_accuracy']*100:.1f}% < 85% target",
            "action": "Adjust THRESHOLD in firmware/esp32-rust/src/inference.rs",
            "file": "firmware/esp32-rust/src/inference.rs",
        }
        priority.append(r)
        print(f"\n  ⚠ [P2] THRESHOLD ADJUSTMENT")
        print(f"     ESP32 accuracy {metrics['esp32_accuracy']*100:.1f}% below 85% target.")
        print(f"     Action: Adjust THRESHOLD constant in firmware/esp32-rust/src/inference.rs")
        if disag.get("pc_says_normal_esp32_says_abnormal", 0) > disag.get("pc_says_abnormal_esp32_says_normal", 0):
            print(f"     Suggestion: RAISE threshold (ESP32 is over-detecting abnormal)")
        else:
            print(f"     Suggestion: LOWER threshold (ESP32 is missing abnormal patterns)")

    # Agreement rate
    if metrics["agreement_rate"] < GOOD_AGREE_RATE:
        r = {
            "priority": 3, "type": "ARCHITECTURE",
            "description": f"Agreement rate {metrics['agreement_rate']*100:.1f}% < 95% target",
            "action": "Try larger architecture (5→16→1) or retrain with more data",
            "file": "python/train_simple_model.py",
        }
        priority.append(r)
        print(f"\n  ⚠ [P3] ARCHITECTURE UPGRADE")
        print(f"     Agreement rate {metrics['agreement_rate']*100:.1f}% < 95% — models diverge too often.")
        print(f"     Action: Increase hidden layer size in train_simple_model.py")
        print(f"     Try:  hidden_layer_sizes=(16,) or (8, 8)")

    # PC accuracy also low
    if metrics["pc_accuracy"] < ACCEPT_ACCURACY:
        print(f"\n  ⚠ [P0] PC MODEL ALSO LOW ACCURACY ({metrics['pc_accuracy']*100:.1f}%)")
        print(f"     Both models struggle — this is a data/feature quality issue, not quantization.")
        print(f"     Action: Improve generate_dummy_ecg.py (better abnormal patterns)")
        print(f"            OR use real ECG dataset (MIT-BIH, PhysioNet)")
        priority.append({
            "priority": 0, "type": "DATA_QUALITY",
            "description": "PC model accuracy too low — data quality issue",
            "action": "Improve synthetic data generation or use real ECG dataset",
            "file": "python/generate_dummy_ecg.py",
        })

    # All good
    if not priority:
        print(f"\n  ✅ All metrics within targets!")
        print(f"     Consider:")
        print(f"     - Testing int4 quantization (run quantize_weights.py with QUANT_BITS=4)")
        print(f"     - Trying a smaller architecture (5→4→1) for lower flash usage")
        print(f"     - Measuring real inference time on ESP32 with CCOUNT register")

    print(f"\n── 5. Next Experiment Suggestions ─────────────────────────────")
    print(f"  Experiment A: Architecture sweep")
    print(f"    Test: 5→4→1, 5→8→1, 5→16→1, 5→8→8→1")
    print(f"    Measure: accuracy, inference_µs, flash_bytes")
    print(f"    Command: edit train_simple_model.py → hidden_layer_sizes=(...)")
    print()
    print(f"  Experiment B: Quantization precision sweep")
    print(f"    Test: float32, int8, int4")
    print(f"    Measure: accuracy delta, model bytes")
    print(f"    Command: edit quantize_weights.py → QUANT_BITS=4 or 8")
    print()
    print(f"  Experiment C: Window size sweep")
    print(f"    Test: RING_BUF_SIZE = 64, 128, 256")
    print(f"    Measure: alert latency, accuracy")
    print(f"    Command: edit firmware/esp32-rust/src/main.rs → RING_BUF_SIZE")

    print(f"\n{SEP}")
    print(f"  Optimization analysis complete.")
    print(f"  See: docs/optimization_guide.md for detailed guidance.")
    print(SEP)

    # Save JSON report
    report = {
        "metrics": metrics,
        "disagreements": disag,
        "model_info": {k: (int(v) if isinstance(v, np.integer) else v)
                       for k, v in model_info.items()},
        "recommendations": priority,
    }
    with open(OPT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[optimization_report] JSON saved to: {OPT_REPORT}")


def main():
    data = load_data()
    if data is None:
        sys.exit(1)

    report_df  = data["report"]
    metrics    = compute_metrics(report_df)
    if metrics is None:
        print("[optimization_report] ERROR: Could not compute metrics from report.")
        sys.exit(1)

    disag      = analyze_disagreements(report_df)
    model_info = get_model_info()

    print_report(metrics, disag, model_info)


if __name__ == "__main__":
    main()
