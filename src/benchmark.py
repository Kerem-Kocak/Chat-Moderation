"""
(D) Benchmark - task metric, model size and latency for every model (R3 + R4).

All models are evaluated on the SAME held-out test set (official TweetEval
test split, never used for training or model selection).

  * Quality : accuracy, macro-F1 (main metric, classes are imbalanced),
              precision / recall of the offensive class
  * Size    : size of the saved weight files on disk, in MB
  * Latency : end-to-end time for ONE message (raw text -> normalization ->
              model -> probabilities), batch size 1, on the CPU, averaged over
              ``BENCHMARK_RUNS`` (>= 100) runs after ``BENCHMARK_WARMUP``
              warm-up runs. Mean, median (p50) and p95 are reported.

Run (on the machine whose hardware you report, i.e. your laptop):

    python -m src.benchmark                 # all models
    python -m src.benchmark --runs 500      # more latency runs
    NUM_THREADS=4 python -m src.benchmark   # (bash) other CPU-thread setting

Outputs in results/:
  results.csv / results.md - the results table for the slides
  predictions.csv          - every test message with every model's prediction
                             (use it to find failure examples)
  hardware.txt             - CPU / threads / library versions to state on the slide
"""
from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from src import config
from src.data import load_splits
from src.predictors import BasePredictor, load_predictors
from src.utils import classification_metrics


# --------------------------------------------------------------------------- #
# Individual measurements
# --------------------------------------------------------------------------- #
def evaluate_quality(predictor: BasePredictor, texts: list[str], labels: np.ndarray):
    """Predict the whole test set (batched) and compute the task metrics."""
    probs = predictor.predict_proba(texts)
    preds = probs.argmax(1)
    return classification_metrics(labels, preds), preds, probs[:, 1]


def measure_latency(predictor: BasePredictor, texts: list[str],
                    runs: int = config.BENCHMARK_RUNS,
                    warmup: int = config.BENCHMARK_WARMUP) -> dict[str, float]:
    """
    Single-message latency in milliseconds. Messages are taken in order from
    the test set (cycling if needed), so every model sees the same inputs.
    """
    for i in range(warmup):                      # warm-up: caches, lazy init, ...
        predictor.predict_proba([texts[i % len(texts)]])
    times = []
    for i in range(runs):
        msg = texts[i % len(texts)]
        t0 = time.perf_counter()
        predictor.predict_proba([msg])
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"latency_mean_ms": float(t.mean()), "latency_p50_ms": float(np.median(t)),
            "latency_p95_ms": float(np.percentile(t, 95))}


def _cpu_name() -> str:
    """Readable CPU model name on Windows / Linux / macOS."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def hardware_info() -> dict[str, str]:
    import sklearn
    import transformers
    return {
        "cpu": _cpu_name(),
        "logical_cores": str(os.cpu_count()),
        "torch_threads": str(torch.get_num_threads()),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scikit-learn": sklearn.__version__,
        "quantized_engine": torch.backends.quantized.engine,
    }


# --------------------------------------------------------------------------- #
# Full benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(keys: list[str] | None = None, runs: int = config.BENCHMARK_RUNS) -> pd.DataFrame:
    test = load_splits()["test"]
    texts, labels = test["text"].tolist(), test["label"].to_numpy()
    predictors = load_predictors(keys)            # also sets torch threads
    hw = hardware_info()
    print("Hardware:", hw)
    print(f"Test set: {len(texts)} messages, latency: {runs} runs, batch size 1, CPU\n")

    rows = []
    pred_table = test.copy()
    for key, p in predictors.items():
        metrics, preds, p_off = evaluate_quality(p, texts, labels)
        lat = measure_latency(p, texts, runs=runs)
        rows.append({
            "model": p.name, "ladder": p.family, "precision": p.precision,
            **{k: round(v, 4) for k, v in metrics.items()},
            "size_mb": round(p.size_mb(), 2),
            **{k: round(v, 2) for k, v in lat.items()},
            "meets_budget": "yes" if lat["latency_mean_ms"] <= config.LATENCY_BUDGET_MS else "no",
        })
        pred_table[f"{key}_pred"] = preds
        pred_table[f"{key}_p_off"] = np.round(p_off, 4)
        print(f"{p.name:<26} macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"size={p.size_mb():8.2f} MB  latency={lat['latency_mean_ms']:7.2f} ms")

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "results.csv", index=False)
    pred_table.to_csv(config.RESULTS_DIR / "predictions.csv", index=False)
    with open(config.RESULTS_DIR / "hardware.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in hw.items()))
        f.write(f"\nlatency_runs: {runs}\nlatency_budget_ms: {config.LATENCY_BUDGET_MS}\n")
    with open(config.RESULTS_DIR / "results.md", "w", encoding="utf-8") as f:
        cols = ["model", "precision", "accuracy", "macro_f1", "size_mb",
                "latency_mean_ms", "latency_p95_ms", "meets_budget"]
        f.write(_to_markdown(df[cols]))
        f.write(f"\n\nHardware: {hw['cpu']}, {hw['torch_threads']} thread(s), "
                f"batch size 1, mean of {runs} runs. Test set: {len(texts)} tweets.\n")
    print(f"\nSaved results to {config.RESULTS_DIR}")
    return df


def _to_markdown(df: pd.DataFrame) -> str:
    """Tiny markdown table writer (avoids the optional `tabulate` dependency)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark all models on the test set.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of: lexicon tfidf w2v_ffn bert_fp32 bert_int8")
    ap.add_argument("--runs", type=int, default=config.BENCHMARK_RUNS)
    args = ap.parse_args()
    run_benchmark(args.models, args.runs)

