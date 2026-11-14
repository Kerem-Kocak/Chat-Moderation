"""
(B) Classical NLP - rule baseline: offensive-word lexicon + minimum edit distance.

Idea
----
1. Mine a lexicon of offensive words *from the training data* (no hand-made
   swear list): for every word we compare how often it appears in offensive
   vs. clean tweets with a smoothed log-odds ratio and keep the top-K words.
2. A message is flagged if one of its tokens is within edit distance <= d of a
   lexicon word. Edit distance catches spelling variants that survive regex
   normalization ("stuupid" -> "stupid", "idoit" -> "idiot", "fuuck" -> "fuck").
   Tokens shorter than 4 characters must match exactly (otherwise "as" would
   match "ass").

The lexicon size K and max distance d are chosen on the validation set.
This model is also used by the app to *mask* the offensive words in flagged
messages.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import normalize, tokenize
from src.utils import classification_metrics, load_json, save_json

LEXICON_FILE = "lexicon.json"


# --------------------------------------------------------------------------- #
# Minimum edit distance (Levenshtein, Wagner-Fischer dynamic programming)
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """
    Minimum number of insertions, deletions and substitutions (each cost 1)
    that turn ``a`` into ``b``.

    D[i][j] = min( D[i-1][j]   + 1,             # delete a[i]
                   D[i][j-1]   + 1,             # insert b[j]
                   D[i-1][j-1] + (a[i] != b[j]) )  # substitute / match

    Only two rows of the DP table are kept (O(len(b)) memory). If
    ``max_dist`` is given we stop early as soon as every cell of a row is
    larger than it and return ``max_dist + 1``.
    """
    if a == b:
        return 0
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# Lexicon mining
# --------------------------------------------------------------------------- #
def mine_lexicon(texts, labels, top_k: int = 100, min_count: int = 5,
                 min_len: int = 3, min_offensive_ratio: float = 0.75) -> list[str]:
    """
    Rank words by the smoothed log-odds of appearing in an offensive tweet:

        score(w) = log P(w | offensive) - log P(w | clean)
        P(w | c) = (df_c(w) + 1) / (N_c + 2)     (add-one smoothing)

    df_c(w) = number of tweets of class c that contain w.
    Words must occur in >= ``min_count`` tweets and >= ``min_offensive_ratio``
    of those tweets must be offensive.
    """
    df_off, df_clean = Counter(), Counter()
    for text, y in zip(texts, labels):
        toks = set(tokenize(normalize(text)))  # document frequency, not raw counts
        (df_off if y == 1 else df_clean).update(toks)

    n_off = int(np.sum(labels))
    n_clean = len(labels) - n_off
    scored = []
    for w in set(df_off) | set(df_clean):
        c_off, c_clean = df_off[w], df_clean[w]
        if w.startswith("<") or len(w) < min_len or (c_off + c_clean) < min_count:
            continue
        if c_off / (c_off + c_clean) < min_offensive_ratio:
            continue
        score = math.log((c_off + 1) / (n_off + 2)) - math.log((c_clean + 1) / (n_clean + 2))
        scored.append((score, w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:top_k]]


# --------------------------------------------------------------------------- #
# Matcher (the actual rule-based classifier)
# --------------------------------------------------------------------------- #
class LexiconMatcher:
    """Flags a message if any token is within ``max_edit`` of a lexicon word."""

    def __init__(self, lexicon: list[str], max_edit: int = 1, min_fuzzy_len: int = 5):
        self.lexicon = sorted(set(lexicon))
        self.lexicon_set = set(self.lexicon)
        self.max_edit = max_edit
        self.min_fuzzy_len = min_fuzzy_len  # short words: exact match only ("ass" vs "as")
        self.by_len: dict[int, list[str]] = defaultdict(list)  # length buckets
        for w in self.lexicon:
            self.by_len[len(w)].append(w)
        self._cache: dict[str, tuple[str, int] | None] = {}

    def match_token(self, tok: str) -> tuple[str, int] | None:
        """Return (closest lexicon word, distance) or None if nothing is close."""
        if tok in self._cache:
            return self._cache[tok]
        best = None
        if tok in self.lexicon_set:
            best = (tok, 0)
        elif self.max_edit > 0 and len(tok) >= self.min_fuzzy_len:
            # only words whose length differs by <= max_edit can be close enough
            for length in range(len(tok) - self.max_edit, len(tok) + self.max_edit + 1):
                for w in self.by_len.get(length, []):
                    if len(w) < self.min_fuzzy_len:
                        continue
                    d = levenshtein(tok, w, self.max_edit)
                    if d <= self.max_edit and (best is None or d < best[1]):
                        best = (w, d)
        self._cache[tok] = best
        return best

    def find_matches(self, text: str) -> list[dict]:
        """All offensive tokens in a raw message, e.g. for masking in the app."""
        out = []
        for tok in tokenize(normalize(text)):
            m = self.match_token(tok)
            if m is not None:
                out.append({"token": tok, "match": m[0], "distance": m[1]})
        return out

    def predict_proba(self, texts) -> np.ndarray:
        """
        Rule "probabilities": 0.95 for an exact lexicon hit, 0.75 for a fuzzy
        hit, 0.05 otherwise. They are confidence scores of a rule, not
        calibrated probabilities.
        """
        p_off = []
        for text in texts:
            matches = self.find_matches(text)
            if any(m["distance"] == 0 for m in matches):
                p_off.append(0.95)
            elif matches:
                p_off.append(0.75)
            else:
                p_off.append(0.05)
        p_off = np.array(p_off)
        return np.stack([1 - p_off, p_off], axis=1)

    # ---- persistence ---- #
    def save(self, directory=config.LEXICON_DIR) -> None:
        save_json({"lexicon": self.lexicon, "max_edit": self.max_edit,
                   "min_fuzzy_len": self.min_fuzzy_len}, directory / LEXICON_FILE)

    @classmethod
    def load(cls, directory=config.LEXICON_DIR) -> "LexiconMatcher":
        d = load_json(directory / LEXICON_FILE)
        return cls(d["lexicon"], d["max_edit"], d["min_fuzzy_len"])


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def train_and_save(splits: dict[str, pd.DataFrame]) -> dict:
    """Grid-search lexicon size and max edit distance on validation macro-F1."""
    train, val = splits["train"], splits["validation"]
    best = None
    for k in config.LEXICON_TOP_K_GRID:
        lexicon = mine_lexicon(train["text"], train["label"].to_numpy(), top_k=k)
        for d in config.LEXICON_MAX_EDIT_GRID:
            matcher = LexiconMatcher(lexicon, max_edit=d)
            pred = matcher.predict_proba(val["text"]).argmax(1)
            m = classification_metrics(val["label"], pred)
            print(f"  [lexicon] top_k={k:<4} max_edit={d}  val macro-F1={m['macro_f1']:.4f}")
            if best is None or m["macro_f1"] > best[0]["macro_f1"]:
                best = (m, matcher, k, d)

    metrics, matcher, k, d = best
    matcher.save()
    print(f"  [lexicon] best: top_k={k}, max_edit={d}; first words: {matcher.lexicon[:15]}")
    return {"val": metrics, "params": {"top_k": k, "max_edit": d}}
