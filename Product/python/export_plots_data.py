# python/export_plots_data.py
# ============================
# Exports all data required for GNUPlot visualization scripts.
#
# Run this before any gnuplot/*.gp script.
# Usage:  py python/export_plots_data.py
#
# Outputs to data/plots/:
#   accuracy_comparison.csv   - PC float32 vs ESP32 int8 vs simulation
#   model_weights_w1.csv      - Layer 1 weights float32 vs dequantized int8
#   model_weights_w2.csv      - Layer 2 weights float32 vs dequantized int8
#   quantization_error.csv    - Per-weight absolute quantization error
#   confusion_matrix_pc.csv   - PC model confusion matrix (2x2)
#   confusion_matrix_esp32.csv- ESP32 model confusion matrix (2x2)
#   training_history.csv      - Fine-tune iteration history
#   weight_distribution.csv   - Histogram of weight values (float32 vs int8)

import os
import sys
import pickle
import numpy as np
import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(SCRIPT_DIR, "..")
DATA_DIR     = os.path.join(ROOT_DIR, "data")
PLOTS_DIR    = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_PKL    = os.path.join(DATA_DIR, "model.pkl")
SCALER_PKL   = os.path.join(DATA_DIR, "scaler.pkl")
WEIGHTS_NPZ  = os.path.join(DATA_DIR, "quantized_weights.npz")
REPORT_CSV   = os.path.join(DATA_DIR, "comparison_report.csv")
SIM_LOG_CSV  = os.path.join(DATA_DIR, "simulated_realtime_log.csv")
ECG_CSV      = os.path.join(DATA_DIR, "sample_ecg.csv")
FINETUNE_LOG = os.path.join(DATA_DIR, "finetune_history.csv")


def load_model():
    if not os.path.exists(MODEL_PKL):
        print(f"  WARNING: {MODEL_PKL} not found — run train_simple_model.py first")
        return None, None
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    return model, None


def load_quantized():
    if not os.path.exists(WEIGHTS_NPZ):
        print(f"  WARNING: {WEIGHTS_NPZ} not found — run quantize_weights.py first")
        return None
    return np.load(WEIGHTS_NPZ)


def export_model_weights(model, npz):
    """Export float32 vs dequantized int8 weights per layer."""
    if model is None or npz is None:
        print("  SKIP: model weights (model or npz not available)")
        return

    # Float32 weights from sklearn model
    W1_f = model.coefs_[0]      # shape (5, hidden)
    b1_f = model.intercepts_[0] # shape (hidden,)
    W2_f = model.coefs_[1]      # shape (hidden, 1)
    b2_f = model.intercepts_[1] # shape (1,)

    # Int8 quantized weights + scales from npz
    keys = list(npz.files)

    def get_npz(key_contains):
        for k in keys:
            if key_contains in k:
                return npz[k]
        return None

    W1_q = get_npz("W1")
    W2_q = get_npz("W2")
    s1   = get_npz("scale_W1")
    s2   = get_npz("scale_W2")

    # Dequantize: float_approx = int8 * scale / 127
    def dequantize(q, scale):
        if q is None or scale is None:
            return None
        return q.astype(np.float32) * float(scale) / 127.0

    W1_dq = dequantize(W1_q, s1)
    W2_dq = dequantize(W2_q, s2)

    # Export W1 (flatten by row: input_neuron × hidden_neuron)
    rows = []
    flat_f32 = W1_f.flatten()
    flat_dq  = W1_dq.flatten() if W1_dq is not None else [None] * len(flat_f32)
    for i, (f, d) in enumerate(zip(flat_f32, flat_dq)):
        rows.append({"param_idx": i, "float32": f,
                     "int8_dequant": d if d is not None else f,
                     "layer": 1})
    pd.DataFrame(rows).to_csv(os.path.join(PLOTS_DIR, "model_weights_w1.csv"), index=False)
    print(f"  Exported: model_weights_w1.csv ({len(rows)} params)")

    # Export W2
    rows = []
    flat_f32 = W2_f.flatten()
    flat_dq  = W2_dq.flatten() if W2_dq is not None else [None] * len(flat_f32)
    for i, (f, d) in enumerate(zip(flat_f32, flat_dq)):
        rows.append({"param_idx": i, "float32": f,
                     "int8_dequant": d if d is not None else f,
                     "layer": 2})
    pd.DataFrame(rows).to_csv(os.path.join(PLOTS_DIR, "model_weights_w2.csv"), index=False)
    print(f"  Exported: model_weights_w2.csv ({len(rows)} params)")

    # Export combined quantization error (all params)
    all_f32 = np.concatenate([W1_f.flatten(), W2_f.flatten(), b1_f.flatten(), b2_f.flatten()])
    if W1_dq is not None and W2_dq is not None:
        all_dq  = np.concatenate([W1_dq.flatten(), W2_dq.flatten(),
                                   np.zeros_like(b1_f), np.zeros_like(b2_f)])
    else:
        all_dq = all_f32.copy()

    err_rows = []
    for i, (f, d) in enumerate(zip(all_f32, all_dq)):
        err_rows.append({
            "param_idx":  i,
            "float32":    f,
            "int8_dequant": d,
            "abs_error":  abs(f - d),
            "rel_error":  abs(f - d) / (abs(f) + 1e-9),
        })
    pd.DataFrame(err_rows).to_csv(os.path.join(PLOTS_DIR, "quantization_error.csv"), index=False)
    print(f"  Exported: quantization_error.csv ({len(err_rows)} params)")

    # Export weight distribution histogram data (20 bins)
    all_bins = np.linspace(all_f32.min(), all_f32.max(), 21)
    f32_hist, edges = np.histogram(all_f32, bins=all_bins)
    dq_hist,  _     = np.histogram(all_dq,  bins=all_bins)
    hist_rows = []
    for i in range(len(f32_hist)):
        hist_rows.append({
            "bin_center":  (edges[i] + edges[i+1]) / 2.0,
            "float32_count": int(f32_hist[i]),
            "int8_count":    int(dq_hist[i]),
        })
    pd.DataFrame(hist_rows).to_csv(os.path.join(PLOTS_DIR, "weight_distribution.csv"), index=False)
    print(f"  Exported: weight_distribution.csv (20 bins)")


def export_accuracy_comparison():
    """Export accuracy/F1 metrics for bar chart."""
    rows = []

    # From comparison_report.csv (PC uart_feed + ESP32)
    if os.path.exists(REPORT_CSV):
        rep = pd.read_csv(REPORT_CSV)
        valid = rep[(rep["pc_prediction"] >= 0) & (rep["gt_label"] >= 0)]
        gt = valid["gt_label"].values
        pc = valid["pc_prediction"].values
        e  = valid["esp32_prediction"].values

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        def safe_metrics(yt, yp):
            ev = [(t, p) for t, p in zip(yt, yp) if p >= 0]
            if not ev:
                return 0, 0, 0, 0
            yt2 = [v[0] for v in ev]
            yp2 = [v[1] for v in ev]
            return (accuracy_score(yt2, yp2),
                    precision_score(yt2, yp2, zero_division=0),
                    recall_score(yt2, yp2, zero_division=0),
                    f1_score(yt2, yp2, zero_division=0))

        acc_pc, prec_pc, rec_pc, f1_pc = safe_metrics(gt, pc)
        acc_e,  prec_e,  rec_e,  f1_e  = safe_metrics(gt, e)

        rows.append({"model": "PC_float32_feed", "accuracy": acc_pc,
                     "precision": prec_pc, "recall": rec_pc, "f1": f1_pc})
        rows.append({"model": "ESP32_int8", "accuracy": acc_e,
                     "precision": prec_e, "recall": rec_e, "f1": f1_e})

    # From simulated_realtime_log.csv (full simulation model)
    if os.path.exists(SIM_LOG_CSV) and os.path.exists(ECG_CSV):
        sim = pd.read_csv(SIM_LOG_CSV, comment="#")
        sim.columns = sim.columns.str.strip()
        ecg = pd.read_csv(ECG_CSV, comment="#")
        ecg.columns = ecg.columns.str.strip()

        pred_col = next((c for c in ["prediction", "pred"] if c in sim.columns), None)
        lbl_col  = next((c for c in ["label", "gt_label"] if c in ecg.columns), ecg.columns[-1])

        if pred_col:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            n = min(len(sim), len(ecg))
            yt = ecg[lbl_col].values[:n].astype(int)
            yp = sim[pred_col].values[:n].astype(int)
            rows.append({
                "model": "PC_sim_float32",
                "accuracy":  accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall":    recall_score(yt, yp, zero_division=0),
                "f1":        f1_score(yt, yp, zero_division=0),
            })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(PLOTS_DIR, "accuracy_comparison.csv"), index=False)
        print(f"  Exported: accuracy_comparison.csv ({len(rows)} models)")

        # Also export transposed version (rows=metrics, cols=models) for GNUPlot histogram
        df = pd.DataFrame(rows)
        metrics = [("Accuracy",  "accuracy"),
                   ("Precision", "precision"),
                   ("Recall",    "recall"),
                   ("F1-Score",  "f1")]
        chart_rows = []
        for mname, mcol in metrics:
            row = {"metric": mname}
            for _, r in df.iterrows():
                row[r["model"]] = r[mcol]
            chart_rows.append(row)
        pd.DataFrame(chart_rows).to_csv(
            os.path.join(PLOTS_DIR, "accuracy_chart.csv"), index=False)
        print(f"  Exported: accuracy_chart.csv (transposed, GNUPlot-ready)")
    else:
        print("  SKIP: accuracy_comparison (no report data)")


def export_confusion_matrices():
    """Export confusion matrix data for PC and ESP32 models."""
    if not os.path.exists(REPORT_CSV):
        print("  SKIP: confusion matrices (comparison_report.csv not found)")
        return

    rep = pd.read_csv(REPORT_CSV)
    from sklearn.metrics import confusion_matrix

    def export_cm(y_true, y_pred, name):
        valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
        if not valid:
            return
        yt = [v[0] for v in valid]
        yp = [v[1] for v in valid]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        # GNUPlot matrix format: row col value
        rows = []
        labels = ["Normal", "Abnormal"]
        for i in range(2):
            for j in range(2):
                rows.append({"row": i, "col": j,
                             "row_label": labels[i], "col_label": labels[j],
                             "count": int(cm[i][j])})
        out = os.path.join(PLOTS_DIR, f"confusion_matrix_{name}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Exported: confusion_matrix_{name}.csv")

    gt = rep["gt_label"].values
    export_cm(gt, rep["pc_prediction"].values,    "pc")
    export_cm(gt, rep["esp32_prediction"].values, "esp32")


def export_training_history():
    """Export fine-tune training history."""
    if not os.path.exists(FINETUNE_LOG):
        # Generate synthetic placeholder history for demo
        rows = []
        for i in range(1, 11):
            rows.append({
                "iteration":    i,
                "old_accuracy": 0.578,
                "new_accuracy": min(0.578 + i * 0.03, 0.92),
                "note":         "placeholder — run fine_tune_model.py to get real data",
            })
        pd.DataFrame(rows).to_csv(os.path.join(PLOTS_DIR, "training_history.csv"), index=False)
        print("  Exported: training_history.csv (placeholder — run fine_tune_model.py for real data)")
    else:
        hist = pd.read_csv(FINETUNE_LOG)
        hist["iteration"] = range(1, len(hist) + 1)
        hist.to_csv(os.path.join(PLOTS_DIR, "training_history.csv"), index=False)
        print(f"  Exported: training_history.csv ({len(hist)} fine-tune runs)")


def main():
    print("[export_plots_data] Exporting GNUPlot data to data/plots/...")
    print()

    model, _ = load_model()
    npz      = load_quantized()

    print("  --- Model Weights ---")
    export_model_weights(model, npz)

    print()
    print("  --- Accuracy Metrics ---")
    export_accuracy_comparison()

    print()
    print("  --- Confusion Matrices ---")
    export_confusion_matrices()

    print()
    print("  --- Training History ---")
    export_training_history()

    print()
    print(f"[export_plots_data] Done. All files in: {PLOTS_DIR}")
    print()
    print("  Now run GNUPlot scripts:")
    print("    gnuplot gnuplot/plot_accuracy_comparison.gp")
    print("    gnuplot gnuplot/plot_model_weights.gp")
    print("    gnuplot gnuplot/plot_quantization_error.gp")
    print("    gnuplot gnuplot/plot_confusion_matrix.gp")
    print("    gnuplot gnuplot/plot_training_history.gp")


if __name__ == "__main__":
    main()
