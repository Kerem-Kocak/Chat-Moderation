"""
(A) Neural networks - our OWN word2vec + a feed-forward network.

Part 1: word2vec (skip-gram with negative sampling, Mikolov et al. 2013)
------------------------------------------------------------------------
Trained from scratch, in PyTorch, on the training tweets only (no pretrained
vectors). For a centre word c and a context word o inside a window:

    loss = -log s(u_o . v_c) - sum_{k=1..K} log s(-u_k . v_c),   u_k ~ P_n(w)

  v = "input" embedding (the word vector we keep), u = "output" embedding,
  s = sigmoid, P_n(w) ~ count(w)^0.75 (noise distribution).
Tricks from the paper: dynamic window size and sub-sampling of frequent words.

Part 2: feed-forward classifier
-------------------------------
    token ids -> frozen word2vec vectors -> [mean-pool ; max-pool] (2 x 100)
              -> LayerNorm -> Linear(200,128) -> ReLU -> Dropout(0.3)
              -> Linear(128, 2) -> softmax
Trained with plain mini-batch SGD (+ momentum) and back-propagation, with
class-weighted cross-entropy; the best epoch is chosen by validation macro-F1
(early stopping).
"""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.preprocessing import normalize, tokenize
from src.utils import classification_metrics, get_device, load_json, save_json, set_seed

WEIGHTS_FILE = "w2v_ffn.pt"
VOCAB_FILE = "vocab.json"
CONFIG_FILE = "config.json"
PAD_ID, UNK_ID = 0, 1


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    """Maps tokens <-> integer ids. id 0 = <pad>, id 1 = <unk>."""

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, min_count: int) -> "Vocab":
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common() if c >= min_count]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def __len__(self) -> int:
        return len(self.itos)


def texts_to_ids(texts, vocab: Vocab, max_len: int) -> list[list[int]]:
    """Raw messages -> normalized -> tokenized -> truncated id lists."""
    return [vocab.encode(tokenize(normalize(t)))[:max_len] for t in texts]


def pad_batch(id_lists: list[list[int]]) -> torch.Tensor:
    """Right-pad a list of id lists with PAD_ID into a (B, T) tensor."""
    T = max(1, max(len(x) for x in id_lists))
    out = torch.full((len(id_lists), T), PAD_ID, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        if ids:
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


# --------------------------------------------------------------------------- #
# Part 1 - word2vec (skip-gram + negative sampling)
# --------------------------------------------------------------------------- #
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, dim)   # v: the vectors we keep
        self.out_emb = nn.Embedding(vocab_size, dim)  # u: context vectors
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)

    def forward(self, center, context, negatives):
        v = self.in_emb(center)                      # (B, D)
        u_pos = self.out_emb(context)                # (B, D)
        u_neg = self.out_emb(negatives)              # (B, K, D)
        pos = F.logsigmoid((v * u_pos).sum(-1))                          # (B,)
        neg = F.logsigmoid(-torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)).sum(-1)  # (B,)
        return -(pos + neg).mean()


def _skipgram_pairs(sentences: list[np.ndarray], keep_prob: np.ndarray,
                    window: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Build all (centre, context) id pairs for one epoch, vectorized with NumPy.
    Frequent words are randomly dropped (sub-sampling) and each centre word
    gets a random window size b in [1, window] (dynamic window).
    """
    ids = np.concatenate(sentences)
    sent = np.concatenate([np.full(len(s), i) for i, s in enumerate(sentences)])
    keep = rng.random(len(ids)) < keep_prob[ids]          # sub-sampling
    ids, sent = ids[keep], sent[keep]
    n = len(ids)
    win = rng.integers(1, window + 1, size=n)             # dynamic window per centre
    centres, contexts = [], []
    for off in range(1, window + 1):
        c = np.arange(n - off)
        same = sent[c] == sent[c + off]                   # never cross tweet borders
        fwd = same & (win[c] >= off)                      # context to the right
        bwd = same & (win[c + off] >= off)                # context to the left
        centres += [c[fwd], c[bwd] + off]
        contexts += [c[fwd] + off, c[bwd]]
    centres, contexts = np.concatenate(centres), np.concatenate(contexts)
    return ids[centres], ids[contexts]


def train_word2vec(token_lists, vocab: Vocab, device: torch.device) -> np.ndarray:
    """Train skip-gram word2vec on the training tweets; returns a (V, D) matrix."""
    p = config.W2V
    rng = np.random.default_rng(config.SEED)
    V = len(vocab)

    # sentences without <unk>; word counts for sub-sampling and noise distribution
    sentences = [np.array([i for i in vocab.encode(t) if i > UNK_ID], dtype=np.int64)
                 for t in token_lists]
    sentences = [s for s in sentences if len(s) > 1]
    counts = np.bincount(np.concatenate(sentences), minlength=V).astype(np.float64)
    freqs = counts / counts.sum()
    with np.errstate(divide="ignore"):
        ratio = np.where(freqs > 0, p["subsample_t"] / freqs, 1.0)
    keep_prob = np.minimum(1.0, np.sqrt(ratio) + ratio)  # Mikolov's formula
    noise = torch.tensor(counts ** 0.75, dtype=torch.float, device=device)
    noise /= noise.sum()

    model = SkipGramNS(V, p["dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    for epoch in range(p["epochs"]):
        centres, contexts = _skipgram_pairs(sentences, keep_prob, p["window"], rng)
        perm = rng.permutation(len(centres))
        centres = torch.tensor(centres[perm], device=device)
        contexts = torch.tensor(contexts[perm], device=device)
        total, steps = 0.0, 0
        for s in range(0, len(centres), p["batch_size"]):
            c, o = centres[s:s + p["batch_size"]], contexts[s:s + p["batch_size"]]
            neg = torch.multinomial(noise, len(c) * p["negatives"], replacement=True)
            loss = model(c, o, neg.view(len(c), p["negatives"]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        print(f"  [word2vec] epoch {epoch + 1}/{p['epochs']}  pairs={len(centres):,}  "
              f"loss={total / max(steps, 1):.4f}")

    emb = model.in_emb.weight.detach().cpu().numpy()
    emb[PAD_ID] = 0.0
    emb[UNK_ID] = 0.0
    return emb


def most_similar(word: str, emb: np.ndarray, vocab: Vocab, k: int = 8) -> list[tuple[str, float]]:
    """Nearest neighbours by cosine similarity - a sanity check for the slides."""
    if word not in vocab.stoi:
        return []
    unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = unit @ unit[vocab.stoi[word]]
    best = np.argsort(-sims)[1:k + 1]
    return [(vocab.itos[i], round(float(sims[i]), 3)) for i in best]


# --------------------------------------------------------------------------- #
# Part 2 - feed-forward classifier on top of the word2vec vectors
# --------------------------------------------------------------------------- #
class W2VFFN(nn.Module):
    def __init__(self, embeddings: torch.Tensor, hidden: int, dropout: float, n_classes: int = 2):
        super().__init__()
        # frozen = the word2vec vectors are features, only the FFN is trained
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=PAD_ID)
        dim = embeddings.size(1)
        self.norm = nn.LayerNorm(2 * dim)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # ids: (B, T)
        mask = (ids > UNK_ID).unsqueeze(-1).float()          # ignore <pad> and <unk>
        e = self.emb(ids)                                    # (B, T, D)
        n_tok = mask.sum(1)                                  # (B, 1)
        mean = (e * mask).sum(1) / n_tok.clamp(min=1)        # average word vector
        mx = e.masked_fill(mask == 0, -1e4).max(1).values    # max over tokens
        mx = torch.where(n_tok > 0, mx, torch.zeros_like(mx))  # empty message -> zeros
        return self.ffn(self.norm(torch.cat([mean, mx], dim=-1)))  # logits (B, 2)


@torch.no_grad()
def predict_proba_ffn(model: W2VFFN, id_lists, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(id_lists), batch_size):
        x = pad_batch(id_lists[s:s + batch_size]).to(device)
        out.append(F.softmax(model(x), dim=-1).cpu().numpy())
    return np.concatenate(out)


def train_ffn(model: W2VFFN, train_ids, y_train, val_ids, y_val, device) -> tuple[W2VFFN, dict]:
    """Mini-batch SGD + back-propagation with early stopping on val macro-F1."""
    p = config.FFN
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    # class weights = N / (2 * N_c): mistakes on the minority class cost more
    class_w = torch.tensor(len(y_train) / (2 * np.bincount(y_train, minlength=2)),
                           dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.SGD([q for q in model.parameters() if q.requires_grad],
                          lr=p["lr"], momentum=p["momentum"], weight_decay=p["weight_decay"])
    best_f1, best_state, best_metrics, bad_epochs = -1.0, None, None, 0
    g = torch.Generator().manual_seed(config.SEED)

    for epoch in range(p["epochs"]):
        model.train()
        perm = torch.randperm(len(train_ids), generator=g).tolist()
        total = 0.0
        for s in range(0, len(perm), p["batch_size"]):
            idx = perm[s:s + p["batch_size"]]
            x = pad_batch([train_ids[i] for i in idx]).to(device)
            y = y_train_t[idx].to(device)
            loss = criterion(model(x), y)   # forward pass
            opt.zero_grad()
            loss.backward()                 # back-propagation: compute gradients
            opt.step()                      # SGD update: w <- w - lr * grad (+ momentum)
            total += loss.item() * len(idx)

        val_pred = predict_proba_ffn(model, val_ids, device).argmax(1)
        m = classification_metrics(y_val, val_pred)
        print(f"  [ffn] epoch {epoch + 1:>2}  train loss={total / len(perm):.4f}  "
              f"val macro-F1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_f1, best_metrics, bad_epochs = m["macro_f1"], m, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= p["patience"]:
                print(f"  [ffn] early stopping after epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, best_metrics


# --------------------------------------------------------------------------- #
# Training entry point + loading
# --------------------------------------------------------------------------- #
def train_and_save(splits: dict[str, pd.DataFrame]) -> dict:
    set_seed(config.SEED)
    device = get_device()
    train, val = splits["train"], splits["validation"]

    # 1) tokens -> vocabulary -> own word2vec
    train_tokens = [tokenize(normalize(t)) for t in train["text"]]
    vocab = Vocab.build(train_tokens, config.W2V["min_count"])
    print(f"  [word2vec] vocabulary: {len(vocab):,} words (min_count={config.W2V['min_count']})")
    emb = train_word2vec(train_tokens, vocab, device)
    for w in ["stupid", "idiot", "love", "trump"]:
        print(f"  [word2vec] most similar to '{w}': {most_similar(w, emb, vocab, 5)}")

    # 2) feed-forward classifier
    model = W2VFFN(torch.tensor(emb), config.FFN["hidden"], config.FFN["dropout"]).to(device)
    train_ids = texts_to_ids(train["text"], vocab, config.FFN["max_len"])
    val_ids = texts_to_ids(val["text"], vocab, config.FFN["max_len"])
    model, metrics = train_ffn(model, train_ids, train["label"].to_numpy(),
                               val_ids, val["label"].to_numpy(), device)

    # 3) save weights (incl. the embedding matrix), vocabulary and config
    config.W2V_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), config.W2V_DIR / WEIGHTS_FILE)
    save_json(vocab.itos, config.W2V_DIR / VOCAB_FILE)
    save_json({"dim": int(emb.shape[1]), "hidden": config.FFN["hidden"],
               "dropout": config.FFN["dropout"], "max_len": config.FFN["max_len"]},
              config.W2V_DIR / CONFIG_FILE)
    return {"val": metrics, "params": {"vocab_size": len(vocab), **config.W2V, **config.FFN}}


def load(directory=config.W2V_DIR) -> tuple[W2VFFN, Vocab, dict]:
    """Rebuild the network on CPU from the saved files."""
    cfg = load_json(directory / CONFIG_FILE)
    vocab = Vocab(load_json(directory / VOCAB_FILE))
    model = W2VFFN(torch.zeros(len(vocab), cfg["dim"]), cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(directory / WEIGHTS_FILE, map_location="cpu",
                                     weights_only=True))
    return model.eval(), vocab, cfg

