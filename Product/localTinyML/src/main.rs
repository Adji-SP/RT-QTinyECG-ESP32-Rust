//! main.rs — RT-QTinyECG Local Test Harness
//! ==========================================
//!
//! Runs the firmware's quantized MLP inference pipeline entirely on the host
//! machine using the same model weights and the same sample_ecg.csv dataset
//! that is used for UART evaluation on the ESP32.
//!
//! ## How it replicates the firmware
//!
//! The pipeline here mirrors firmware/esp32-rust/src/main.rs exactly:
//!
//!   CSV row → moving-avg filter (window=8) → ring buffer (128 samples)
//!             → inference::infer() → debounce → summary stats
//!
//! The only differences from the firmware are:
//!   - Input comes from a CSV file instead of the ADC or UART.
//!   - Output goes to stdout instead of UART/LED/buzzer.
//!   - Timing uses std::time::Instant instead of the Xtensa CCOUNT register.
//!   - No LED/buzzer hardware.
//!
//! ## Usage
//!
//! ```powershell
//! cargo run                        # uses default dataset path
//! cargo run -- path/to/custom.csv  # specify a different CSV
//! ```
//!
//! ## CSV format
//!
//! The CSV must have a header row with at least the columns:
//!   time_ms, adc_value, label
//!
//! `label` is the ground-truth (0 = Normal, 1 = Abnormal). It is used to
//! compute accuracy metrics at the end of the run.
//!
//! DISCLAIMER: Educational prototype only. Not for clinical use.

mod model_weights;
mod inference;
mod filter;
mod ring_buffer;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

// ── Pipeline constants (must match firmware/esp32-rust/src/main.rs) ───────────

/// Ring buffer size in samples. 128 samples @ 250 Hz = 512 ms window.
const RING_BUF_SIZE: usize = 128;

/// Moving average filter window (same as firmware FILTER_WINDOW = 8).
const FILTER_WINDOW: usize = 8;

/// Number of consecutive ABNORMAL windows required before triggering alert.
const DEBOUNCE_ABNORMAL: u8 = 3;

/// Number of consecutive NORMAL windows required before clearing alert.
const DEBOUNCE_NORMAL: u8 = 5;

// ── ANSI colour helpers ────────────────────────────────────────────────────────

fn green(s: &str) -> String { format!("\x1b[32m{}\x1b[0m", s) }
fn red(s: &str)   -> String { format!("\x1b[31m{}\x1b[0m", s) }
fn bold(s: &str)  -> String { format!("\x1b[1m{}\x1b[0m",  s) }
fn cyan(s: &str)  -> String { format!("\x1b[36m{}\x1b[0m", s) }
fn yellow(s: &str)-> String { format!("\x1b[33m{}\x1b[0m", s) }
fn dim(s: &str)   -> String { format!("\x1b[2m{}\x1b[0m",  s) }

// ── Data record from CSV ──────────────────────────────────────────────────────

struct Record {
    time_ms:   u32,
    adc_value: i32,
    label:     u8,   // ground truth
}

// ── CSV loader ────────────────────────────────────────────────────────────────

/// Parse sample_ecg.csv.  Returns (records, col_indices).
fn load_csv(path: &str) -> Result<Vec<Record>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open '{}': {}", path, e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header to find column indices (robust against column reordering)
    let header_line = lines.next()
        .ok_or("CSV is empty")?
        .map_err(|e| format!("Read error: {}", e))?;

    let headers: Vec<&str> = header_line.split(',').map(str::trim).collect();

    let col = |name: &str| -> Result<usize, String> {
        headers.iter().position(|h| *h == name)
            .ok_or_else(|| format!("Column '{}' not found in CSV header", name))
    };

    let col_time  = col("time_ms")?;
    let col_adc   = col("adc_value")?;
    let col_label = col("label")?;

    let mut records = Vec::new();

    for (line_no, line) in lines.enumerate() {
        let line = line.map_err(|e| format!("Line {}: read error: {}", line_no + 2, e))?;
        let line = line.trim();
        if line.is_empty() { continue; }

        let fields: Vec<&str> = line.split(',').collect();

        let parse_field = |col: usize, name: &str| -> Result<&str, String> {
            fields.get(col)
                .map(|s| s.trim())
                .ok_or_else(|| format!("Line {}: missing column '{}'", line_no + 2, name))
        };

        let time_ms: u32 = parse_field(col_time, "time_ms")?
            .parse().map_err(|_| format!("Line {}: bad time_ms", line_no + 2))?;
        let adc_value: i32 = parse_field(col_adc, "adc_value")?
            .parse().map_err(|_| format!("Line {}: bad adc_value", line_no + 2))?;
        let label: u8 = parse_field(col_label, "label")?
            .parse().map_err(|_| format!("Line {}: bad label", line_no + 2))?;

        records.push(Record { time_ms, adc_value, label });
    }

    Ok(records)
}

// ── Confusion matrix ──────────────────────────────────────────────────────────

struct ConfusionMatrix {
    tp: u32, // predicted abnormal, actually abnormal
    tn: u32, // predicted normal,   actually normal
    fp: u32, // predicted abnormal, actually normal
    fn_: u32,// predicted normal,   actually abnormal
}

impl ConfusionMatrix {
    fn new() -> Self { Self { tp: 0, tn: 0, fp: 0, fn_: 0 } }

    fn update(&mut self, predicted: u8, actual: u8) {
        match (predicted, actual) {
            (1, 1) => self.tp  += 1,
            (0, 0) => self.tn  += 1,
            (1, 0) => self.fp  += 1,
            (0, 1) => self.fn_ += 1,
            _      => {}
        }
    }

    fn total(&self) -> u32 { self.tp + self.tn + self.fp + self.fn_ }

    fn accuracy(&self) -> f64 {
        let t = self.total();
        if t == 0 { return 0.0; }
        (self.tp + self.tn) as f64 / t as f64
    }

    fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { return 0.0; }
        self.tp as f64 / denom as f64
    }

    fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 { return 0.0; }
        self.tp as f64 / denom as f64
    }

    fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { return 0.0; }
        2.0 * p * r / (p + r)
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    // ── Banner ────────────────────────────────────────────────────────────────
    println!();
    println!("{}", bold("╔══════════════════════════════════════════════════════╗"));
    println!("{}", bold("║     RT-QTinyECG  ·  Local TinyML Test Harness       ║"));
    println!("{}", bold("║     Quantized MLP  5→8→1  ·  Integer arithmetic     ║"));
    println!("{}", bold("╚══════════════════════════════════════════════════════╝"));
    println!();

    // ── Resolve dataset path ──────────────────────────────────────────────────
    let args: Vec<String> = env::args().collect();
    let csv_path: PathBuf = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Default: look for ../../data/sample_ecg.csv relative to project root.
        // When running `cargo run` from localTinyML/, the working dir is that folder.
        let candidates = [
            "../../data/sample_ecg.csv",
            "../data/sample_ecg.csv",
            "data/sample_ecg.csv",
            "sample_ecg.csv",
        ];
        candidates.iter()
            .map(PathBuf::from)
            .find(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from(candidates[0]))
    };

    println!("{} {}", cyan("Dataset:"), csv_path.display());

    // ── Load CSV ──────────────────────────────────────────────────────────────
    let records = match load_csv(csv_path.to_str().unwrap_or("")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {}", red("ERROR"), e);
            eprintln!();
            eprintln!("Usage: cargo run [-- path/to/ecg.csv]");
            std::process::exit(1);
        }
    };

    println!("{} {} samples loaded", cyan("Records:"), bold(&records.len().to_string()));
    println!();

    // ── Print model config ────────────────────────────────────────────────────
    println!("{}", bold("Model Architecture"));
    println!("  {:12} {} → {} → {} (Quantized MLP, integer arithmetic)",
        "Topology:", model_weights::N_FEATURES, model_weights::N_HIDDEN, model_weights::N_OUTPUT);
    println!("  {:12} Window={}, Filter={}", "Pipeline:", RING_BUF_SIZE, FILTER_WINDOW);
    println!("  {:12} Abnormal≥{} windows, Clear≥{} windows",
        "Debounce:", DEBOUNCE_ABNORMAL, DEBOUNCE_NORMAL);
    println!();

    // ── Show first window debug info ──────────────────────────────────────────
    println!("{}", bold("First-window debug (after ring buffer fills):"));

    // ── Inference pipeline state ──────────────────────────────────────────────
    let mut ring_buf = ring_buffer::RingBuffer::<i32, RING_BUF_SIZE>::new();
    let mut filt     = filter::MovingAverageState::<FILTER_WINDOW>::new();

    let mut abnormal_streak: u8 = 0;
    let mut normal_streak:   u8 = 0;
    let mut alert_active: bool  = false;

    // Metrics
    let mut cm_raw     = ConfusionMatrix::new(); // raw prediction vs label (per window)
    let mut cm_debounce= ConfusionMatrix::new(); // debounced alert vs label (per sample)
    let mut windows_run: u32 = 0;
    let mut total_inference_ns: u128 = 0;
    let mut first_window_shown = false;
    let mut alert_events: Vec<(u32, u32)> = Vec::new(); // (start_ms, end_ms)
    let mut alert_start_ms: u32 = 0;

    // Per-sample output: only show first N windows to avoid flooding
    const SHOW_DETAIL_WINDOWS: u32 = 8;
    let mut detail_shown: u32 = 0;

    println!();
    println!("{}", bold("── Live inference trace (first 8 windows shown) ─────────────────────────"));
    println!("{:>10}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>7}  {}",
        "time_ms", "adc", "mean", "max", "p2p", "energy", "out_acc", "pred", "alert");
    println!("{}", dim(&"─".repeat(90)));

    // ── Main sample loop ──────────────────────────────────────────────────────
    for record in &records {
        // 1. Moving-average filter (identical to firmware)
        let filtered: i32 = filt.push_and_average(record.adc_value);

        // 2. Push into ring buffer
        ring_buf.push(filtered);

        // 3. Inference once buffer is full
        if ring_buf.is_full() {
            let window: &[i32] = ring_buf.as_slice();

            // Time the inference
            let t0 = Instant::now();
            let prediction = inference::infer(window);
            let elapsed_ns = t0.elapsed().as_nanos();

            total_inference_ns += elapsed_ns;
            windows_run += 1;

            // Debug: show internals for the very first window
            if !first_window_shown {
                first_window_shown = true;
                let features = inference::extract_features(window);
                let (pred, feat_q, hidden, out_acc) =
                    inference::mlp_infer_debug(&features);

                println!();
                println!("  feat_raw   [mean={}, max={}, min={}, p2p={}, energy/4096={}]",
                    features.mean, features.maximum, features.minimum,
                    features.peak_to_peak, features.energy / 4096);
                println!("  feat_q     {:?}", feat_q);
                println!("  hidden     {:?}", hidden);
                println!("  output_acc {:>10}  →  prediction: {}",
                    out_acc, if pred == 1 { red("ABNORMAL") } else { green("normal") });
                println!();
            }

            // Debounce state machine (identical to firmware)
            if prediction == 1 {
                abnormal_streak = abnormal_streak.saturating_add(1);
                normal_streak   = 0;
                if abnormal_streak >= DEBOUNCE_ABNORMAL && !alert_active {
                    alert_active   = true;
                    alert_start_ms = record.time_ms;
                }
            } else {
                normal_streak   = normal_streak.saturating_add(1);
                abnormal_streak = 0;
                if normal_streak >= DEBOUNCE_NORMAL && alert_active {
                    alert_events.push((alert_start_ms, record.time_ms));
                    alert_active = false;
                }
            }

            // ── Metrics: raw per-window prediction vs ground truth ───────────
            cm_raw.update(prediction, record.label);

            // ── Metrics: debounced alert vs ground truth (per sample) ─────────
            cm_debounce.update(alert_active as u8, record.label);

            // ── Print detail trace (first SHOW_DETAIL_WINDOWS windows) ────────
            if detail_shown < SHOW_DETAIL_WINDOWS {
                let features = inference::extract_features(window);
                let (_, _, _, out_acc) = inference::mlp_infer_debug(&features);
                let pred_str = if prediction == 1 { red("ABN") } else { green("NRM") };
                let alert_str = if alert_active { red("ALERT") } else { dim("    -") };

                println!("{:>10}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}  {:>7}  {}",
                    record.time_ms, record.adc_value,
                    features.mean, features.maximum, features.peak_to_peak,
                    features.energy / 4096, out_acc, pred_str, alert_str);

                detail_shown += 1;
            }
        }
    }

    // Close any open alert
    if alert_active {
        if let Some(last) = records.last() {
            alert_events.push((alert_start_ms, last.time_ms));
        }
    }

    // ── Final report ─────────────────────────────────────────────────────────
    println!("{}", dim(&"─".repeat(90)));
    println!();
    println!("{}", bold("══════════════════════════════════════════════════════"));
    println!("{}", bold("  INFERENCE RESULTS"));
    println!("{}", bold("══════════════════════════════════════════════════════"));
    println!();

    // Timing
    let avg_ns = if windows_run > 0 { total_inference_ns / windows_run as u128 } else { 0 };
    let avg_us = avg_ns as f64 / 1000.0;
    println!("{}", bold("Timing"));
    println!("  Windows processed    : {}", bold(&windows_run.to_string()));
    println!("  Total inference time : {:.3} ms", total_inference_ns as f64 / 1_000_000.0);
    println!("  Avg per window       : {:.2} µs  ({} ns)", avg_us, avg_ns);
    println!("  Throughput           : {:.0} windows/sec",
        if avg_ns > 0 { 1_000_000_000.0 / avg_ns as f64 } else { 0.0 });
    println!();

    // Raw per-window metrics
    println!("{}", bold("Raw Window Predictions (MLP output, no debounce)"));
    println!("  Accuracy   : {:.2}%", cm_raw.accuracy()  * 100.0);
    println!("  Precision  : {:.2}%", cm_raw.precision() * 100.0);
    println!("  Recall     : {:.2}%", cm_raw.recall()    * 100.0);
    println!("  F1 score   : {:.4}",  cm_raw.f1());
    println!();
    println!("  Confusion matrix:");
    println!("              Predicted");
    println!("              Normal  Abnormal");
    println!("  Actual  NRM  {:>6}  {:>8}", cm_raw.tn, cm_raw.fp);
    println!("          ABN  {:>6}  {:>8}", cm_raw.fn_, cm_raw.tp);
    println!();

    // Debounced metrics
    println!("{}", bold("Debounced Alert Metrics (per sample)"));
    println!("  Accuracy   : {:.2}%", cm_debounce.accuracy()  * 100.0);
    println!("  Precision  : {:.2}%", cm_debounce.precision() * 100.0);
    println!("  Recall     : {:.2}%", cm_debounce.recall()    * 100.0);
    println!("  F1 score   : {:.4}",  cm_debounce.f1());
    println!();

    // Alert events
    println!("{}", bold("Alert Events"));
    if alert_events.is_empty() {
        println!("  {} (no abnormal segments detected)", green("No alerts fired"));
    } else {
        println!("  {} alert event(s) detected:", alert_events.len());
        for (i, (start, end)) in alert_events.iter().enumerate() {
            let duration_ms = end.saturating_sub(*start);
            println!("    [{}]  {}ms → {}ms  ({}ms duration)",
                i + 1, start, end,
                yellow(&duration_ms.to_string()));
        }
    }
    println!();

    // Pass/fail gate
    let accuracy = cm_raw.accuracy() * 100.0;
    println!("{}", bold("── Summary ──────────────────────────────────────────"));
    if accuracy >= 80.0 {
        println!("  {} Accuracy {:.1}% ≥ 80% — model is working correctly",
            green("PASS"), accuracy);
    } else {
        println!("  {} Accuracy {:.1}% < 80% — check weights or feature pipeline",
            red("WARN"), accuracy);
    }
    println!();
    println!("{}", dim("Weights source : firmware/esp32-rust/src/model_weights.rs"));
    println!("{}", dim("Dataset source : data/sample_ecg.csv"));
    println!("{}", dim("Pipeline       : CSV → MovingAvg(8) → RingBuf(128) → MLP(5→8→1)"));
    println!("{}", dim("DISCLAIMER     : Educational prototype. Not for clinical use."));
    println!();
}
