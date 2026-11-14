"""
Train the whole model ladder in one command.

    python -m src.train_all                         # everything
    python -m src.train_all --models tfidf w2v_ffn  # only some steps

Steps (in order):
  lexicon  - (B) rule baseline: mined lexicon + edit distance
  tfidf    - (B) classical baseline: TF-IDF + logistic regression
  w2v_ffn  - (A) own word2vec + feed-forward NN (SGD + backprop)
  bert     - (C) fine-tune BERT (use a GPU: Colab or local CUDA)
  quantize - (D) dynamic INT8 quantization of the fine-tuned BERT

Validation metrics and training times are written to results/training_log.json
(useful for the "how each model was trained" slide).
"""
from __future__ import annotations

import argparse
import time

from src import config
from src.data import describe_splits, load_splits
from src.utils import save_json

STEP_ORDER = ["lexicon", "tfidf", "w2v_ffn", "bert", "quantize"]


def _get_step(name: str):
    # imported lazily so that e.g. the lexicon step does not import transformers
    if name == "lexicon":
        from src.models.lexicon_matcher import train_and_save
    elif name == "tfidf":
        from src.models.tfidf_baseline import train_and_save
    elif name == "w2v_ffn":
        from src.models.w2v_ffn import train_and_save
    elif name == "bert":
        from src.models.bert_classifier import train_and_save
    elif name == "quantize":
        from src.models.quantize import quantize_and_save as train_and_save
    else:
        raise ValueError(name)
    return train_and_save


def main(steps: list[str]) -> None:
    if config.SMOKE_TEST:
        print("*** SMOKE TEST MODE: tiny data + 2-layer BERT. Do NOT report these numbers. ***")
    splits = load_splits()
    print(describe_splits(splits).to_string(index=False), "\n")

    log_path = config.RESULTS_DIR / "training_log.json"
    log = {}
    if log_path.exists():  # keep results of steps that are not re-run
        from src.utils import load_json
        log = load_json(log_path)

    for name in STEP_ORDER:
        if name not in steps:
            continue
        print(f"=== {name} ===")
        t0 = time.perf_counter()
        result = _get_step(name)(splits)
        result["train_seconds"] = round(time.perf_counter() - t0, 1)
        log[name] = result
        val = result.get("val")
        if val:
            print(f"=== {name} done in {result['train_seconds']}s - "
                  f"val macro-F1={val['macro_f1']:.4f} acc={val['accuracy']:.4f} ===\n")
        save_json(log, log_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the model ladder.")
    ap.add_argument("--models", nargs="*", default=STEP_ORDER, choices=STEP_ORDER)
    main(ap.parse_args().models)

