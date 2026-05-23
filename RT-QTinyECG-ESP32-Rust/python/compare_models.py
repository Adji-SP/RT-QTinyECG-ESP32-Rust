# python/compare_models.py
# =========================
# Cross-Model Comparison: PC float32 vs ESP32 int8
#
# Loads both prediction sets (from uart_feed_evaluator.py) and
# computes accuracy, quantization delta, and disagreement analysis.
#
# Usage:
#   py python/compare_models.py
#   py python/compare_models.py --save-disagreements data/disagreements.csv
#
# DISCLAIMER: Educational prototype only. Not for clinical use.

import os
import sys
import argparse
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(SCRIPT_DIR, "..")
DATA_DIR     = os.path.join(ROOT_DIR, "data")
ESP32_CSV    = os.path.join(DATA_DIR, "esp32_predictions.csv")
SIM_LOG_CSV  = os.path.join(DATA_DIR, "simulated_realtime_log.csv")
ECG_CSV      = os.path.join(DATA_DIR, "sample_ecg.csv")
REPORT_CSV   = os.path.join(DATA_DIR, "comparison_report.csv")


def parse_args():
    p = argparse.ArgumentParser(
        description="RT-QTinyECG Cross-Model Comparison (PC float32 vs ESP32 int8)"
    )
    p.add_argument("--esp32-csv",  default=ESP32_CSV,   help="ESP32 predictions CSV")
    p.add_argument("--sim-csv",    default=SIM_LOG_CSV, help="PC simulation log CSV")
    p.add_argument("--ecg-csv",    default=ECG_CSV,     help="Ground truth ECG CSV")
    p.add_argument("--save-disagreements", default=None,
                   help="Optional: save disagreement samples to this CSV")
    return p.parse_args()


def print_metrics_block(name, y_true, y_pred, prefix=""):
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    if not valid:
        print(f"{prefix}  No valid predictions for {name}")
        return {}

    yt = [v[0] for v in valid]
    yp = [v[1] for v in valid]

    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    f1   = f1_score(yt, yp, zero_division=0)
    cm   = confusion_matrix(yt, yp, labels=[0, 1])

    print(f"{prefix}  Samples evaluated : {len(yt)}")
    print(f"{prefix}  Accuracy          : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"{prefix}  Precision         : {prec:.4f}")
    print(f"{prefix}  Recall            : {rec:.4f}")
    print(f"{prefix}  F1-Score          : {f1:.4f}")
    print(f"{prefix}  Confusion Matrix:")
    print(f"{prefix}               Pred Normal  Pred Abnormal")
    if len(cm) == 2:
        print(f"{prefix}    GT Normal      {cm[0][0]:6d}        {cm[0][1]:6d}")
        print(f"{prefix}    GT Abnormal    {cm[1][0]:6d}        {cm[1][1]:6d}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_comparison(args):
    # ── Load ground truth ─────────────────────────────────────────────────────
    gt_df = None
    if os.path.exists(args.ecg_csv):
        gt_df = pd.read_csv(args.ecg_csv, comment="#")
        gt_df.columns = gt_df.columns.str.strip()
        label_col = next(
            (c for c in ["label", "gt_label", "ground_truth"] if c in gt_df.columns),
            gt_df.columns[-1]
        )
        gt_labels = list(gt_df[label_col].astype(int))
        print(f"[compare_models] Ground truth loaded: {len(gt_labels)} samples")
    else:
        gt_labels = None
        print(f"[compare_models] WARNING: {args.ecg_csv} not found — no GT labels available")

    # ── Load ESP32 predictions ────────────────────────────────────────────────
    esp32_preds = None
    esp32_pc_preds = None
    round_trips = []

    if os.path.exists(args.esp32_csv):
        esp_df = pd.read_csv(args.esp32_csv)
        esp32_preds   = list(esp_df["esp32_prediction"].astype(int))
        esp32_pc_preds = list(esp_df["pc_prediction"].astype(int))
        if "round_trip_ms" in esp_df.columns:
            round_trips = [r for r in esp_df["round_trip_ms"] if r > 0]
        print(f"[compare_models] ESP32 predictions loaded: {len(esp32_preds)} samples")
    else:
        print(f"[compare_models] WARNING: {args.esp32_csv} not found.")
        print(f"  → Run uart_feed_evaluator.py first (or with --dry-run)")

    # ── Load PC simulation predictions ────────────────────────────────────────
    pc_sim_preds = None
    if os.path.exists(args.sim_csv):
        sim_df = pd.read_csv(args.sim_csv, comment="#")
        sim_df.columns = sim_df.columns.str.strip()
        pred_col = next(
            (c for c in ["prediction", "pred", "y_pred"] if c in sim_df.columns),
            None
        )
        if pred_col:
            pc_sim_preds = list(sim_df[pred_col].astype(int))
            print(f"[compare_models] PC simulation predictions loaded: {len(pc_sim_preds)} samples")
    else:
        print(f"[compare_models] WARNING: {args.sim_csv} not found.")
        print(f"  → Run realtime_ecg_simulator.py first")

    # ── Align lengths ─────────────────────────────────────────────────────────
    n = min(
        len(gt_labels)      if gt_labels      else 99999,
        len(esp32_preds)    if esp32_preds    else 99999,
        len(esp32_pc_preds) if esp32_pc_preds else 99999,
    )

    if n == 0 or n == 99999:
        print("[compare_models] ERROR: No data to compare.")
        sys.exit(1)

    gt    = gt_labels[:n]      if gt_labels      else [0] * n
    esp32 = esp32_preds[:n]    if esp32_preds    else [-1] * n
    pc_ua = esp32_pc_preds[:n] if esp32_pc_preds else [-1] * n  # PC model from uart_feed run
    pc_sim = pc_sim_preds[:n]  if pc_sim_preds   else [-1] * n  # PC sim model

    # ── Print report ──────────────────────────────────────────────────────────
    SEP = "═" * 62
    print(f"\n{SEP}")
    print(f"  RT-QTinyECG — Cross-Model Comparison Report")
    print(f"  Total samples aligned: {n}")
    print(SEP)

    print("\n── A. PC float32 Model (from uart_feed_evaluator.py) ─────────")
    metrics_pc = print_metrics_block("PC float32", gt, pc_ua, prefix="  ")

    print("\n── B. ESP32 int8 Model (via UART feed) ────────────────────────")
    metrics_esp32 = print_metrics_block("ESP32 int8", gt, esp32, prefix="  ")

    if pc_sim_preds:
        print("\n── C. PC float32 Simulation (realtime_ecg_simulator.py) ───────")
        metrics_sim = print_metrics_block("PC simulation", gt, pc_sim, prefix="  ")
    else:
        metrics_sim = {}

    # ── Quantization delta ────────────────────────────────────────────────────
    if metrics_pc and metrics_esp32:
        delta = metrics_pc.get("accuracy", 0) - metrics_esp32.get("accuracy", 0)
        print(f"\n── D. Quantization Analysis ────────────────────────────────────")
        print(f"  PC float32 accuracy   : {metrics_pc.get('accuracy', 0):.4f}")
        print(f"  ESP32 int8  accuracy  : {metrics_esp32.get('accuracy', 0):.4f}")
        print(f"  Quantization delta    : {delta:+.4f}  ({delta*100:+.2f}%)")
        if abs(delta) < 0.03:
            print(f"  → ✅ Excellent: < 3% accuracy loss from quantization")
        elif abs(delta) < 0.05:
            print(f"  → ⚠ Acceptable: 3–5% accuracy loss")
        else:
            print(f"  → ❌ High degradation: > 5% — consider fine-tuning (Section 3 of optimization_guide.md)")

    # ── Disagreement analysis ─────────────────────────────────────────────────
    print(f"\n── E. Model Disagreement Analysis ─────────────────────────────")
    disagreements = []
    for i, (g, e, p) in enumerate(zip(gt, esp32, pc_ua)):
        if e >= 0 and p >= 0 and e != p:
            disagreements.append({
                "index": i, "gt_label": g,
                "pc_prediction": p, "esp32_prediction": e,
            })

    n_valid = sum(1 for e, p in zip(esp32, pc_ua) if e >= 0 and p >= 0)
    n_agree = n_valid - len(disagreements)
    agree_rate = n_agree / n_valid if n_valid > 0 else 0

    print(f"  Valid paired samples  : {n_valid}")
    print(f"  Agreements            : {n_agree}")
    print(f"  Disagreements         : {len(disagreements)}")
    print(f"  Agreement rate        : {agree_rate:.4f}  ({agree_rate*100:.1f}%)")

    if disagreements:
        disag_normal  = sum(1 for d in disagreements if d["gt_label"] == 0)
        disag_abnorm  = sum(1 for d in disagreements if d["gt_label"] == 1)
        print(f"    On normal segments  : {disag_normal}")
        print(f"    On abnormal segments: {disag_abnorm}")
        if disag_abnorm > disag_normal:
            print(f"  → Disagreements mainly on ABNORMAL samples.")
            print(f"    Likely: threshold too high or missing pattern in training data.")
            print(f"    Action: lower threshold OR fine-tune model (see optimization_guide.md)")

    # ── UART latency ──────────────────────────────────────────────────────────
    if round_trips:
        print(f"\n── F. UART Communication Latency ───────────────────────────────")
        print(f"  Mean round-trip       : {sum(round_trips)/len(round_trips):.1f} ms")
        print(f"  Max  round-trip       : {max(round_trips):.1f} ms")
        print(f"  Min  round-trip       : {min(round_trips):.1f} ms")
        print(f"  Sampling budget       : 4.0 ms (250 Hz)")
        if max(round_trips) > 4.0:
            print(f"  → ⚠ Max latency > 4 ms budget — reduce baud or lower sample rate")
        else:
            print(f"  → ✅ All round-trips within 4 ms budget")

    print(f"\n{SEP}")
    print(f"  Comparison complete.")
    print(SEP)

    # ── Save comparison report ────────────────────────────────────────────────
    report_rows = []
    for i in range(n):
        report_rows.append({
            "index":          i,
            "gt_label":       gt[i],
            "pc_prediction":  pc_ua[i],
            "esp32_prediction": esp32[i],
            "pc_sim_prediction": pc_sim[i] if i < len(pc_sim) else -1,
            "agreement":      int(pc_ua[i] == esp32[i]) if (pc_ua[i] >= 0 and esp32[i] >= 0) else -1,
        })

    pd.DataFrame(report_rows).to_csv(REPORT_CSV, index=False)
    print(f"\n[compare_models] Report saved to: {REPORT_CSV}")

    # ── Save disagreements if requested ──────────────────────────────────────
    if args.save_disagreements and disagreements:
        pd.DataFrame(disagreements).to_csv(args.save_disagreements, index=False)
        print(f"[compare_models] Disagreements saved to: {args.save_disagreements}")

    return metrics_pc, metrics_esp32


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
