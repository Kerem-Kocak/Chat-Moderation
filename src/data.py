"""
Dataset loading.

We use TweetEval / offensive (= OffensEval 2019, SemEval-2019 Task 6A):
English tweets labelled NOT offensive (0) or OFFensive (1).

The official train / validation / test splits are used unchanged:
  * train      -> fit every model
  * validation -> choose hyper-parameters / best epoch
  * test       -> the ONE held-out set on which all models are compared (R3)

Citations (put them on the Data slide):
  Zampieri et al. (2019) "SemEval-2019 Task 6: Identifying and Categorizing
  Offensive Language in Social Media (OffensEval)".
  Barbieri et al. (2020) "TweetEval: Unified Benchmark and Comparative
  Evaluation for Tweet Classification".

Run ``python -m src.data`` to download the data and print split statistics.
"""
from __future__ import annotations

import pandas as pd

from src import config

SPLITS = ("train", "validation", "test")


def _download() -> dict[str, pd.DataFrame]:
    """Download the dataset from the Hugging Face Hub (needs internet once)."""
    # In smoke-test mode, reuse the full splits if they were already downloaded.
    full_dir = config.ROOT_DIR / "data"
    if config.SMOKE_TEST and all((full_dir / f"{s}.parquet").exists() for s in SPLITS):
        return {s: pd.read_parquet(full_dir / f"{s}.parquet") for s in SPLITS}

    # We download the official parquet files directly with huggingface_hub
    # instead of `datasets.load_dataset`: same data, fewer dependencies, and
    # it avoids Windows' 260-character path limit hit by `datasets` lock files.
    from huggingface_hub import hf_hub_download  # imported lazily: the app does not need it

    out = {}
    for s in SPLITS:
        path = hf_hub_download(config.DATASET_NAME, repo_type="dataset",
                               filename=f"{config.DATASET_CONFIG}/{s}-00000-of-00001.parquet")
        out[s] = pd.read_parquet(path)[["text", "label"]]
    return out


def load_splits() -> dict[str, pd.DataFrame]:
    """
    Return {"train": df, "validation": df, "test": df}, each with the columns
    ``text`` (str) and ``label`` (int: 0 clean, 1 offensive).

    The first call downloads and caches the splits as parquet files in
    ``config.DATA_DIR``; later calls read the cache (fast and offline).
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {s: config.DATA_DIR / f"{s}.parquet" for s in SPLITS}

    if all(p.exists() for p in paths.values()):
        return {s: pd.read_parquet(p) for s, p in paths.items()}

    splits = _download()
    if config.SMOKE_TEST:  # tiny random subsets, only to test the code
        splits = {
            s: df.sample(n=min(len(df), config.SMOKE_SIZES[s]), random_state=config.SEED)
            .reset_index(drop=True)
            for s, df in splits.items()
        }
    for s, df in splits.items():
        df.to_parquet(paths[s], index=False)
    return splits


def describe_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Size and class balance per split - copy this table to the Data slide."""
    rows = []
    for name, df in splits.items():
        n_off = int((df["label"] == 1).sum())
        rows.append({
            "split": name,
            "messages": len(df),
            "clean": len(df) - n_off,
            "offensive": n_off,
            "offensive_%": round(100 * n_off / len(df), 1),
            "avg_words": round(df["text"].str.split().str.len().mean(), 1),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    s = load_splits()
    print(describe_splits(s).to_string(index=False))
    print("\nExample tweets:")
    print(s["train"].sample(5, random_state=1).to_string(index=False))

