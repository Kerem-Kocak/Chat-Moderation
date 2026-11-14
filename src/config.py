"""
Central configuration for the Real-Time Chat Moderation project.

Every path, hyper-parameter and constant lives here so that the rest of the
code base (training, benchmarking, app) stays consistent. Changing a value in
this file changes it everywhere - this is also what keeps the architecture
diagram in the slides in sync with the code.

Smoke-test mode
---------------
Set the environment variable ``SMOKE_TEST=1`` to run the whole pipeline on a
tiny subset of the data with a tiny 2-layer BERT. This is
only meant to check that the code runs end-to-end in a few minutes; its
numbers must NOT be reported. Smoke artifacts are written to separate folders
so they never overwrite the real models.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Mode
# --------------------------------------------------------------------------- #
SMOKE_TEST: bool = os.environ.get("SMOKE_TEST", "0") == "1"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
_suffix = "_smoke" if SMOKE_TEST else ""
DATA_DIR = ROOT_DIR / f"data{_suffix}"            # cached dataset splits (parquet)
ARTIFACTS_DIR = ROOT_DIR / f"artifacts{_suffix}"  # trained model files (NOT in git)
RESULTS_DIR = ROOT_DIR / f"results{_suffix}"      # benchmark tables (small, in git)

LEXICON_DIR = ARTIFACTS_DIR / "lexicon"      # rule baseline (edit-distance lexicon)
TFIDF_DIR = ARTIFACTS_DIR / "tfidf_lr"       # classical baseline
W2V_DIR = ARTIFACTS_DIR / "w2v_ffn"          # own word2vec + feed-forward NN
BERT_DIR = ARTIFACTS_DIR / "bert_fp32"       # fine-tuned BERT (FP32)
BERT_INT8_DIR = ARTIFACTS_DIR / "bert_int8"  # dynamically quantized BERT (INT8)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# TweetEval "offensive" = OffensEval 2019 (SemEval-2019 Task 6, sub-task A).
# Official splits: train 11,916 / validation 1,324 / test 860 tweets.
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_CONFIG = "offensive"
LABELS = ["clean", "offensive"]  # index 0 = NOT offensive, 1 = OFFensive
SEED = 42

# Sub-sample sizes used only in smoke-test mode.
SMOKE_SIZES = {"train": 600, "validation": 200, "test": 200}

# --------------------------------------------------------------------------- #
# Model hyper-parameters
# --------------------------------------------------------------------------- #
# (B) Rule baseline: lexicon mined from the training set + edit distance.
LEXICON_TOP_K_GRID = [50, 80, 120, 200]   # lexicon sizes tried on validation
LEXICON_MAX_EDIT_GRID = [1]               # 1 = Levenshtein edit-distance matching

# (B) Classical baseline: TF-IDF (word + char n-grams) + logistic regression.
TFIDF_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# (A) Own word2vec (skip-gram with negative sampling) + feed-forward network.
W2V = {
    "dim": 100,          # embedding size
    "window": 5,         # max context window (dynamic, as in the original paper)
    "negatives": 5,      # negative samples per positive pair
    "min_count": 2,      # ignore words seen fewer times
    "epochs": 5 if not SMOKE_TEST else 1,
    "batch_size": 1024,
    "lr": 2e-3,
    "subsample_t": 1e-3,  # frequent-word sub-sampling threshold
}
FFN = {
    "hidden": 128,
    "dropout": 0.3,
    "epochs": 40 if not SMOKE_TEST else 3,
    "batch_size": 64,
    "lr": 0.05,          # plain SGD (+ momentum) as required by the use case
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "patience": 6,       # early stopping on validation macro-F1
    "max_len": 64,       # max tokens per message
}

# (C) Transformer: BERT fine-tuned for sequence classification.
BERT_MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2" if SMOKE_TEST else "bert-base-uncased"
# The tiny smoke-test BERT uses exactly the same uncased WordPiece vocabulary
# as bert-base-uncased, so we always take the tokenizer from bert-base-uncased.
BERT_TOKENIZER_NAME = "bert-base-uncased"
BERT = {
    "epochs": 3 if not SMOKE_TEST else 1,
    "batch_size": 32,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_len": 64,       # tweets are short; 64 sub-word tokens covers ~99 %
}

# --------------------------------------------------------------------------- #
# Efficiency / benchmarking
# --------------------------------------------------------------------------- #
# Latency budget per message on a CPU-only server (Use Case 4, area D).
LATENCY_BUDGET_MS = 50.0
# Number of single-message runs used to average latency (homework: >= 100).
BENCHMARK_RUNS = 200 if not SMOKE_TEST else 20
BENCHMARK_WARMUP = 10
# CPU threads used for inference. 1 thread simulates one worker of a
# CPU-only chat server that handles many conversations in parallel.
NUM_THREADS = int(os.environ.get("NUM_THREADS", "1"))

