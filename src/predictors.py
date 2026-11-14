"""
One common interface for all five models, used by the benchmark AND the app
(so the numbers in the results table come from exactly the code the app runs).

    predictor.predict_proba(["msg 1", "msg 2"])  -> np.ndarray of shape (n, 2)
    predictor.predict("one message")             -> ("offensive", 0.93)
    predictor.size_mb()                          -> size of the weight files

Every predictor takes RAW text; normalization happens inside. All inference
runs on the CPU (the deployment target of Use Case 4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src import config
from src.utils import file_size_mb


class BasePredictor:
    key: str = ""         # short id, used in the app and CSV files
    name: str = ""        # name shown in tables / UI
    family: str = ""      # rung of the model ladder
    precision: str = ""   # numeric precision of the weights
    files: list[Path] = []

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, text: str) -> tuple[str, float]:
        probs = self.predict_proba([text])[0]
        idx = int(np.argmax(probs))
        return config.LABELS[idx], float(probs[idx])

    def size_mb(self) -> float:
        return file_size_mb(self.files)


class LexiconPredictor(BasePredictor):
    key, name, family, precision = "lexicon", "Lexicon + Edit Distance", "Rule baseline", "n/a"

    def __init__(self, directory=config.LEXICON_DIR):
        from src.models.lexicon_matcher import LEXICON_FILE, LexiconMatcher
        self.matcher = LexiconMatcher.load(directory)
        self.files = [directory / LEXICON_FILE]

    def predict_proba(self, texts):
        return self.matcher.predict_proba(texts)


class TfidfPredictor(BasePredictor):
    key, name, family, precision = "tfidf", "TF-IDF + LogReg", "Classical baseline", "FP64"

    def __init__(self, directory=config.TFIDF_DIR):
        from src.models import tfidf_baseline
        self.pipe = tfidf_baseline.load(directory)
        self.files = [directory / tfidf_baseline.MODEL_FILE]

    def predict_proba(self, texts):
        return self.pipe.predict_proba(list(texts))


class W2VFFNPredictor(BasePredictor):
    key, name, family, precision = "w2v_ffn", "Word2Vec + FFN", "Neural network", "FP32"

    def __init__(self, directory=config.W2V_DIR):
        from src.models import w2v_ffn
        self._m = w2v_ffn
        self.model, self.vocab, self.cfg = w2v_ffn.load(directory)
        self.files = [directory / w2v_ffn.WEIGHTS_FILE, directory / w2v_ffn.VOCAB_FILE]

    def predict_proba(self, texts):
        ids = self._m.texts_to_ids(texts, self.vocab, self.cfg["max_len"])
        return self._m.predict_proba_ffn(self.model, ids, "cpu")


class BertPredictor(BasePredictor):
    key, name, family, precision = "bert_fp32", "BERT-base (FP32)", "Transformer", "FP32"

    def __init__(self, directory=config.BERT_DIR):
        from src.models import bert_classifier
        self._predict = bert_classifier.predict_proba_bert
        self.model, self.tokenizer = bert_classifier.load(directory, device="cpu")
        self.files = [directory / "model.safetensors"]

    def predict_proba(self, texts):
        return self._predict(self.model, self.tokenizer, list(texts), "cpu")


class QuantizedBertPredictor(BertPredictor):
    key, name, family, precision = "bert_int8", "BERT-base (INT8)", "Quantized Transformer", "INT8 (dynamic)"

    def __init__(self, directory=config.BERT_INT8_DIR):
        from src.models import bert_classifier, quantize
        self._predict = bert_classifier.predict_proba_bert
        self.model, self.tokenizer = quantize.load_quantized(directory)
        self.files = [directory / quantize.INT8_FILE]


# The ladder in display order. The app's default model is the quantized one.
PREDICTOR_CLASSES = [LexiconPredictor, TfidfPredictor, W2VFFNPredictor,
                     BertPredictor, QuantizedBertPredictor]


def load_predictors(keys: list[str] | None = None) -> dict[str, BasePredictor]:
    """Load the requested (default: all) predictors; skip ones not trained yet."""
    torch.set_num_threads(config.NUM_THREADS)
    out = {}
    for cls in PREDICTOR_CLASSES:
        if keys is not None and cls.key not in keys:
            continue
        try:
            out[cls.key] = cls()
        except (FileNotFoundError, OSError) as e:
            print(f"[warn] {cls.name} not available ({type(e).__name__}: {e}). Train it first.")
    return out

