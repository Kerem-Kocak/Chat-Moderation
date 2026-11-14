"""
Small helpers shared by all modules: seeding, device selection, file sizes,
metrics and JSON I/O.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def set_seed(seed: int) -> None:
    """Make Python, NumPy and PyTorch random generators reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU for *training* if there is one (Colab / local CUDA)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_size_mb(paths: Iterable[Path | str]) -> float:
    """Total size on disk of the given files in MB (1 MB = 1024 * 1024 bytes)."""
    return sum(Path(p).stat().st_size for p in paths) / (1024 ** 2)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Task metrics for Use Case 4.

    * accuracy  - share of correct predictions
    * macro_f1  - mean F1 of both classes; the main metric because the
                  classes are imbalanced (~1/3 offensive)
    * offensive precision / recall - how reliable / complete the flags are
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "off_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "off_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def save_json(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

