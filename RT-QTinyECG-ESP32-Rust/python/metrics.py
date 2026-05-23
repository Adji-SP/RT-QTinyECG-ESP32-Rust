# -*- coding: utf-8 -*-
"""
metrics.py
===========
Computes and prints evaluation metrics from the simulated real-time log.

Loads: data/simulated_realtime_log.csv
       data/sample_ecg.csv  (for ground truth labels)

Metrics computed:
  Classification metrics:
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix

  Real-time performance metrics:
    - Average inference time (µs)
    - Max inference time (µs)
    - P95 inference time (µs)
    - Average alert latency (ms)
    - Max alert latency (ms)
    - Sampling interval mean (ms) and standard deviation (ms)
    - Sampling interval stability (coefficient of variation)

  Model / memory metrics:
    - Approximate model size in bytes (from quantized_weights.npz if available)
    - Static RAM usage estimate for ring buffer and filter state

Usage:
    python python/metrics.py

Requires:
    data/simulated_realtime_log.csv  (from realtime_ecg_simulator.py)
    data/sample_ecg.csv              (from generate_dummy_ecg.py)
"""

import os
import csv
import numpy as np

# Optional sklearn metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. Classification metrics will be computed manually.")

LOG_CSV    = os.path.join(os.path.dirname(__file__), "..", "data", "simulated_realtime_log.csv")
ECG_CSV    = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecg.csv")
WEIGHTS_NPZ = os.path.join(os.path.dirname(__file__), "..", "data", "quantized_weights.npz")

SAMPLE_INTERVAL_MS = 4.0   # Expected: 1000 / 250 Hz
WINDOW_SIZE        = 128


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_log(path: str) -> dict:
    """Load simulated_realtime_log.csv into arrays."""
    rows = {
        "time_ms":          [],
        "adc_value":        [],
        "filtered_value":   [],
        "inference_us":     [],
        "prediction":       [],
        "alert":            [],
        "alert_latency_ms": [],
    }
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows["time_ms"].append(int(row["time_ms"]))
            rows["adc_value"].append(int(row["adc_value"]))
            rows["filtered_value"].append(float(row["filtered_value"]))
            rows["inference_us"].append(float(row["inference_us"]))
            rows["prediction"].append(int(row["prediction"]))
            rows["alert"].append(int(row["alert"]))
            rows["alert_latency_ms"].append(float(row["alert_latency_ms"]))
    return {k: np.array(v) for k, v in rows.items()}


def load_ecg_labels(path: str) -> np.ndarray:
    """Load ground truth labels from sample_ecg.csv."""
    labels = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
    return np.array(labels)


# ─── Classification Metrics ───────────────────────────────────────────────────

def window_labels_from_sample_labels(sample_labels: np.ndarray,
                                      window_size: int,
                                      n_predictions: int) -> np.ndarray:
    """
    Map per-sample ground truth labels to per-window labels.
    Each window label = majority of sample labels in that window.
    Windows are generated with step = window_size // 2.
    """
    step = window_size // 2
    win_labels = []
    for i in range(n_predictions):
        start = i * step
        end   = start + window_size
        if end > len(sample_labels):
            end = len(sample_labels)
        if start >= len(sample_labels):
            win_labels.append(0)
        else:
            chunk = sample_labels[start:end]
            majority = int(np.round(np.mean(chunk)))
            win_labels.append(majority)
    return np.array(win_labels, dtype=np.int32)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute basic binary classification metrics."""
    if SKLEARN_AVAILABLE:
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        cm   = confusion_matrix(y_true, y_pred)
    else:
        # Manual computation
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        acc  = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cm   = np.array([[tn, fp], [fn, tp]])

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


# ─── Real-Time Metrics ────────────────────────────────────────────────────────

def compute_sampling_stability(timestamps_ms: np.ndarray) -> dict:
    """
    Compute sampling interval statistics.

    In ideal 250 Hz sampling, intervals should be exactly 4 ms.
    Jitter > 0.5 ms can indicate timer instability.

    Returns:
        dict with mean, std, cv (coefficient of variation), min, max
    """
    intervals = np.diff(timestamps_ms.astype(np.float64))
    if len(intervals) == 0:
        return {}
    mean_ms = float(np.mean(intervals))
    std_ms  = float(np.std(intervals))
    cv      = std_ms / mean_ms if mean_ms > 0 else 0.0
    return {
        "mean_interval_ms"  : mean_ms,
        "std_interval_ms"   : std_ms,
        "cv"                : cv,
        "min_interval_ms"   : float(np.min(intervals)),
        "max_interval_ms"   : float(np.max(intervals)),
    }


def compute_inference_metrics(inference_us: np.ndarray) -> dict:
    """Compute inference time statistics (filtering out zeros)."""
    nonzero = inference_us[inference_us > 0]
    if len(nonzero) == 0:
        return {"note": "No inference measurements found (buffer never full?)"}
    return {
        "mean_us"  : float(np.mean(nonzero)),
        "max_us"   : float(np.max(nonzero)),
        "min_us"   : float(np.min(nonzero)),
        "p95_us"   : float(np.percentile(nonzero, 95)),
        "n_inferences": int(len(nonzero)),
    }


def compute_alert_latency_metrics(alert_latency_ms: np.ndarray) -> dict:
    """Compute alert latency statistics (filtering out zeros)."""
    nonzero = alert_latency_ms[alert_latency_ms > 0]
    if len(nonzero) == 0:
        return {"note": "No alert latency events (no alerts were triggered?)"}
    return {
        "mean_latency_ms": float(np.mean(nonzero)),
        "max_latency_ms" : float(np.max(nonzero)),
        "min_latency_ms" : float(np.min(nonzero)),
        "n_alert_events" : int(len(nonzero)),
    }


# ─── Model Size Estimation ────────────────────────────────────────────────────

def estimate_model_size() -> dict:
    """
    Estimate model memory footprint from quantized weights NPZ file.
    Also estimates static RAM usage for ring buffer and filter state.
    """
    result = {}

    if os.path.exists(WEIGHTS_NPZ):
        data = np.load(WEIGHTS_NPZ)
        total_bytes = sum(arr.nbytes for arr in data.values())
        result["quantized_weights_bytes"] = total_bytes
        result["quantized_weights_kb"]    = total_bytes / 1024.0
    else:
        # Estimate from placeholder weights
        # W1: [8,5] i8 = 40 bytes, b1: [8] i32 = 32 bytes
        # W2: [1,8] i8 = 8 bytes,  b2: [1] i32 = 4 bytes
        # Scales: 4 × f32 = 16 bytes
        placeholder_bytes = 40 + 32 + 8 + 4 + 16
        result["quantized_weights_bytes"] = placeholder_bytes
        result["quantized_weights_kb"]    = placeholder_bytes / 1024.0
        result["note"] = "Estimated from placeholder. Run export_rust_weights.py for actual."

    # Static RAM usage estimate
    # Ring buffer: WINDOW_SIZE × sizeof(i32) = 128 × 4 = 512 bytes
    ring_buffer_bytes = WINDOW_SIZE * 4
    # Filter state (moving avg, 8 samples): 8 × 4 = 32 bytes
    filter_state_bytes = 8 * 4
    # Log buffer / string: ~64 bytes
    log_buffer_bytes = 64

    result["ring_buffer_ram_bytes"] = ring_buffer_bytes
    result["filter_state_ram_bytes"] = filter_state_bytes
    result["log_buffer_ram_bytes"]   = log_buffer_bytes
    result["total_static_ram_estimate_bytes"] = (
        result["quantized_weights_bytes"]
        + ring_buffer_bytes
        + filter_state_bytes
        + log_buffer_bytes
    )

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RT-QTinyECG-ESP32-Rust — Evaluation Metrics")
    print("=" * 60)

    # Check files
    if not os.path.exists(LOG_CSV):
        print(f"\nERROR: {LOG_CSV} not found.")
        print("  Run: python python/realtime_ecg_simulator.py")
        return

    if not os.path.exists(ECG_CSV):
        print(f"\nERROR: {ECG_CSV} not found.")
        print("  Run: python python/generate_dummy_ecg.py")
        return

    # Load data
    print(f"\n[metrics] Loading log data...")
    log  = load_log(LOG_CSV)
    gt   = load_ecg_labels(ECG_CSV)
    print(f"  Log samples   : {len(log['time_ms'])}")
    print(f"  GT labels     : {len(gt)}")

    # ── 1. Sampling Interval Stability ──
    print(f"\n── 1. Sampling Interval Stability ─────────────────")
    samp = compute_sampling_stability(log["time_ms"])
    print(f"  Expected interval : {SAMPLE_INTERVAL_MS:.1f} ms (250 Hz)")
    print(f"  Mean interval     : {samp.get('mean_interval_ms', 0):.3f} ms")
    print(f"  Std deviation     : {samp.get('std_interval_ms', 0):.3f} ms")
    print(f"  Min interval      : {samp.get('min_interval_ms', 0):.3f} ms")
    print(f"  Max interval      : {samp.get('max_interval_ms', 0):.3f} ms")
    print(f"  Coeff of Variation: {samp.get('cv', 0)*100:.2f}%")
    cv = samp.get("cv", 0)
    if cv < 0.01:
        print(f"  → Excellent stability (CV < 1%)")
    elif cv < 0.05:
        print(f"  → Good stability (CV < 5%)")
    else:
        print(f"  → Poor stability (CV ≥ 5%) — timer jitter concern")

    # ── 2. Inference Time ──
    print(f"\n── 2. Inference Time ──────────────────────────────")
    inf = compute_inference_metrics(log["inference_us"])
    if "note" in inf:
        print(f"  {inf['note']}")
    else:
        print(f"  Inferences run    : {inf['n_inferences']}")
        print(f"  Mean time         : {inf['mean_us']:.2f} µs")
        print(f"  Max time          : {inf['max_us']:.2f} µs")
        print(f"  Min time          : {inf['min_us']:.2f} µs")
        print(f"  P95 time          : {inf['p95_us']:.2f} µs")
        budget_us = SAMPLE_INTERVAL_MS * 1000
        print(f"  Sampling budget   : {budget_us:.0f} µs (4 ms @ 250 Hz)")
        if inf["max_us"] < budget_us:
            print(f"  → OK: Max inference fits within 4 ms sampling interval")
        else:
            print(f"  → WARNING: Inference exceeds sampling budget!")

    # ── 3. Alert Latency ──
    print(f"\n── 3. Alert Latency ───────────────────────────────")
    alat = compute_alert_latency_metrics(log["alert_latency_ms"])
    if "note" in alat:
        print(f"  {alat['note']}")
    else:
        print(f"  Alert events      : {alat['n_alert_events']}")
        print(f"  Mean latency      : {alat['mean_latency_ms']:.2f} ms")
        print(f"  Max latency       : {alat['max_latency_ms']:.2f} ms")
        print(f"  Min latency       : {alat['min_latency_ms']:.2f} ms")

    # ── 4. Classification Metrics ──
    print(f"\n── 4. Classification Metrics ──────────────────────")
    # Predictions are per-window (not per-sample). Map ground truth to window level.
    predictions = log["prediction"]
    n_pred = len(predictions[predictions != 0]) + len(predictions[predictions == 0])
    # Use only non-zero predictions (window was full)
    valid_mask = log["inference_us"] > 0
    pred_valid = log["prediction"][valid_mask]
    n_valid    = len(pred_valid)

    if n_valid > 0:
        # Map sample labels to window labels
        gt_windowed = window_labels_from_sample_labels(gt, WINDOW_SIZE, n_valid)
        gt_windowed = gt_windowed[:n_valid]

        clf = compute_classification_metrics(gt_windowed, pred_valid)
        print(f"  Windows evaluated : {n_valid}")
        print(f"  Accuracy          : {clf['accuracy']:.4f}")
        print(f"  Precision         : {clf['precision']:.4f}")
        print(f"  Recall            : {clf['recall']:.4f}")
        print(f"  F1-Score          : {clf['f1']:.4f}")
        print(f"  Confusion Matrix:")
        cm = clf["cm"]
        if cm.shape == (2, 2):
            print(f"             Pred Normal  Pred Abnormal")
            print(f"    GT Normal     {cm[0,0]:6d}       {cm[0,1]:6d}")
            print(f"    GT Abnormal   {cm[1,0]:6d}       {cm[1,1]:6d}")
        if SKLEARN_AVAILABLE:
            print(f"\n  Detailed Report:")
            print(classification_report(
                gt_windowed, pred_valid,
                target_names=["Normal", "Abnormal"],
                zero_division=0
            ))
    else:
        print(f"  No complete window inferences found.")

    # ── 5. Model / Memory Metrics ──
    print(f"\n── 5. Model & Memory Size Estimation ─────────────")
    mem = estimate_model_size()
    print(f"  Quantized weights size : {mem.get('quantized_weights_bytes', 'N/A')} bytes "
          f"({mem.get('quantized_weights_kb', 0):.2f} KB)")
    print(f"  Ring buffer RAM        : {mem.get('ring_buffer_ram_bytes', 'N/A')} bytes")
    print(f"  Filter state RAM       : {mem.get('filter_state_ram_bytes', 'N/A')} bytes")
    print(f"  Log buffer RAM         : {mem.get('log_buffer_ram_bytes', 'N/A')} bytes")
    print(f"  Total static RAM est.  : {mem.get('total_static_ram_estimate_bytes', 'N/A')} bytes "
          f"({mem.get('total_static_ram_estimate_bytes',0)/1024:.2f} KB)")
    print(f"  ESP32 total RAM        : 520 KB SRAM (plenty of headroom)")
    if "note" in mem:
        print(f"  Note: {mem['note']}")

    print(f"\n{'='*60}")
    print(f"  Metrics evaluation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
