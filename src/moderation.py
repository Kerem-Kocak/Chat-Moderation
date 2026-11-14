"""
UI-independent moderation logic used by the app (and testable without Gradio).

    service = ModerationService(load_predictors())
    service.moderate("you are an 1d10t", "bert_int8")
    -> {"label": "offensive", "confidence": 0.97, "p_offensive": 0.97,
        "latency_ms": 21.3, "within_budget": True,
        "display_text": "you are an i****", "masked_words": 1, ...}

Pipeline per message (matches the architecture diagram):
  raw text -> (B) regex normalization (inside every predictor)
           -> selected model -> P(clean), P(offensive)    [timed, CPU]
           -> post-processing: if offensive, mask lexicon words
              (exact or within edit distance 1); if no word can be
              located, hide the whole message.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from src import config
from src.predictors import BasePredictor
from src.preprocessing import explain_normalization, normalize, tokenize

_RE_CHUNK = re.compile(r"\S+")
HIDDEN_TEXT = "█████ (message hidden by the moderator)"


class ModerationService:
    def __init__(self, predictors: dict[str, BasePredictor]):
        self.predictors = predictors
        self.matcher = predictors["lexicon"].matcher if "lexicon" in predictors else None

    # ------------------------------------------------------------------ #
    def warm_up(self, n: int = 5) -> None:
        """Warm up caches and tensor shapes across realistic message lengths."""
        samples = [
            "warm up message",
            "you are such an idiot, uninstall the game right now before ruining the match",
            "nice shot bro, you crushed that last round, let us queue again together!",
        ]
        for p in self.predictors.values():
            for i in range(n):
                p.predict_proba([samples[i % len(samples)]])

    def classify(self, text: str, key: str) -> tuple[str, float, float, float]:
        """(label, confidence, p_offensive, latency_ms), end-to-end on the CPU."""
        t0 = time.perf_counter()
        probs = self.predictors[key].predict_proba([text])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        idx = int(np.argmax(probs))
        return config.LABELS[idx], float(probs[idx]), float(probs[1]), latency_ms

    # ------------------------------------------------------------------ #
    def mask(self, text: str) -> tuple[str, int]:
        """
        Replace every word matching the offensive lexicon (after regex
        normalization, exact or within edit distance 1) by its first letter
        followed by stars. Returns (masked text, number of masked words).
        """
        if self.matcher is None:
            return text, 0
        n = 0

        def repl(m: re.Match) -> str:
            nonlocal n
            chunk = m.group(0)
            if any(self.matcher.match_token(t) for t in tokenize(normalize(chunk))):
                n += 1
                alnum = [i for i, c in enumerate(chunk) if c.isalnum()]
                keep = alnum[0] if alnum else 0
                return "".join(c if i == keep or not c.isalnum() else "*"
                               for i, c in enumerate(chunk))
            return chunk

        return _RE_CHUNK.sub(repl, text), n

    def moderate(self, text: str, key: str) -> dict:
        """Classify one chat message and build what the chat should display."""
        label, conf, p_off, ms = self.classify(text, key)
        if label == "offensive":
            masked, n = self.mask(text)
            display = masked if n > 0 else HIDDEN_TEXT
        else:
            display, n = text, 0
        return {"label": label, "confidence": conf, "p_offensive": p_off,
                "latency_ms": ms, "within_budget": ms <= config.LATENCY_BUDGET_MS,
                "display_text": display, "masked_words": n,
                "model": self.predictors[key].name}

    # ------------------------------------------------------------------ #
    def inspect(self, text: str, latency_runs: int = 5):
        """One message through the whole pipeline: steps, matches, all models."""
        steps = pd.DataFrame(explain_normalization(text), columns=["normalization step", "result"])
        steps.loc[len(steps)] = ["tokens", " | ".join(tokenize(normalize(text)))]

        found = self.matcher.find_matches(text) if self.matcher else []
        matches = pd.DataFrame(found, columns=["token", "match", "distance"])

        rows = []
        for key, p in self.predictors.items():
            lat = [self.classify(text, key)[3] for _ in range(latency_runs)]
            label, conf, p_off, _ = self.classify(text, key)
            rows.append({"model": p.name, "ladder": p.family, "precision": p.precision,
                         "prediction": label, "P(offensive)": round(p_off, 3),
                         f"latency_ms (median of {latency_runs})": round(float(np.median(lat)), 2),
                         "size_mb": round(p.size_mb(), 1)})
        return steps, matches, pd.DataFrame(rows)

