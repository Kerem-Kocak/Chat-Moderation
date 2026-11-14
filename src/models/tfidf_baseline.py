"""
(B) Classical NLP - classical baseline: TF-IDF features + logistic regression.

Features (concatenated with a FeatureUnion):
  * word uni- and bi-grams of the normalized, tokenized text
  * character 3-5-grams inside word boundaries (``char_wb``) - very robust to
    misspellings and creative spelling, which are common in chat

TF-IDF weight:  tfidf(t, d) = (1 + log tf(t, d)) * log((1 + N) / (1 + df(t))) + 1
(``sublinear_tf=True`` gives the 1 + log tf part), then L2-normalized.

The classifier is a logistic regression with ``class_weight="balanced"`` to
compensate for the 2:1 class imbalance. The regularization strength C is
chosen on the validation set.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config
from src.preprocessing import normalize, tokenize
from src.utils import classification_metrics

MODEL_FILE = "tfidf_lr.joblib"


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Raw text in -> class probabilities out (normalization is inside)."""
    word_tfidf = TfidfVectorizer(
        preprocessor=normalize,      # our regex normalization (B)
        tokenizer=tokenize,          # our regex tokenizer (B)
        token_pattern=None,          # silence sklearn warning (we pass a tokenizer)
        lowercase=False,             # already lower-cased by normalize
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        preprocessor=normalize,
        analyzer="char_wb",
        lowercase=False,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )
    return Pipeline([
        ("features", FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])),
        ("clf", LogisticRegression(C=C, class_weight="balanced",
                                   max_iter=2000, solver="liblinear")),
    ])


def train_and_save(splits: dict[str, pd.DataFrame]) -> dict:
    """Fit on train, pick C on validation, save the best pipeline."""
    train, val = splits["train"], splits["validation"]
    best = None
    for C in config.TFIDF_C_GRID:
        pipe = build_pipeline(C).fit(train["text"], train["label"])
        m = classification_metrics(val["label"], pipe.predict(val["text"]))
        print(f"  [tfidf] C={C:<5} val macro-F1={m['macro_f1']:.4f}")
        if best is None or m["macro_f1"] > best[0]["macro_f1"]:
            best = (m, pipe, C)

    metrics, pipe, C = best
    config.TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.TFIDF_DIR / MODEL_FILE)
    n_feat = len(pipe.named_steps["features"].get_feature_names_out())
    print(f"  [tfidf] best C={C}; {n_feat:,} features")
    return {"val": metrics, "params": {"C": C, "n_features": n_feat}}


def load(directory=config.TFIDF_DIR) -> Pipeline:
    return joblib.load(directory / MODEL_FILE)

