"""
(C) Transformers - BERT fine-tuned for offensive-message classification.

Model: ``bert-base-uncased`` (12 layers, 768 hidden, 12 attention heads,
~110 M parameters, WordPiece sub-word tokenizer). Hugging Face's
``AutoModelForSequenceClassification`` puts a task head on top:

    [CLS] token embedding -> dropout -> Linear(768, 2) -> softmax

All weights (BERT + head) are fine-tuned end-to-end with AdamW, a linear
warm-up / decay learning-rate schedule and gradient clipping. On a GPU we use
mixed precision (FP16) for speed; the saved weights are always FP32.

We use a plain PyTorch training loop instead of the ``Trainer`` class so that
every step is visible (and explainable in the presentation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.preprocessing import normalize
from src.utils import classification_metrics, get_device, set_seed

ID2LABEL = {i: l for i, l in enumerate(config.LABELS)}
LABEL2ID = {l: i for i, l in ID2LABEL.items()}


@torch.no_grad()
def predict_proba_bert(model, tokenizer, texts, device="cpu",
                       batch_size: int = 64, max_len: int = config.BERT["max_len"]) -> np.ndarray:
    """Raw messages -> normalize -> WordPiece -> BERT -> softmax probabilities."""
    model.eval()
    out = []
    texts = [normalize(t) for t in texts]
    for s in range(0, len(texts), batch_size):
        enc = tokenizer(texts[s:s + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        logits = model(**enc).logits
        out.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def _linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    """LR rises linearly 0 -> lr during warm-up, then decays linearly to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_and_save(splits: dict[str, pd.DataFrame],
                   model_name: str = config.BERT_MODEL_NAME,
                   out_dir=config.BERT_DIR) -> dict:
    set_seed(config.SEED)
    p = config.BERT
    device = get_device()
    print(f"  [bert] fine-tuning {model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_TOKENIZER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    ).float().to(device)

    train, val = splits["train"], splits["validation"]
    train_data = list(zip([normalize(t) for t in train["text"]], train["label"].tolist()))

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=p["max_len"], return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    loader = DataLoader(train_data, batch_size=p["batch_size"], shuffle=True,
                        collate_fn=collate, generator=torch.Generator().manual_seed(config.SEED))

    # AdamW; no weight decay on biases and LayerNorm weights (standard for BERT)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [w for n, w in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": p["weight_decay"]},
        {"params": [w for n, w in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=p["lr"])
    total_steps = len(loader) * p["epochs"]
    scheduler = _linear_warmup_decay(optimizer, int(p["warmup_ratio"] * total_steps), total_steps)

    # class-weighted loss, same idea as for the FFN (offensive is the minority)
    y = train["label"].to_numpy()
    class_w = torch.tensor(len(y) / (2 * np.bincount(y, minlength=2)), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    use_amp = device.type == "cuda"          # mixed precision only on GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_metrics = -1.0, None
    for epoch in range(p["epochs"]):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = criterion(logits.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()                       # back-propagation
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  [bert] epoch {epoch + 1} step {step}/{len(loader)} loss={running / step:.4f}")

        val_pred = predict_proba_bert(model, tokenizer, val["text"].tolist(), device).argmax(1)
        m = classification_metrics(val["label"], val_pred)
        print(f"  [bert] epoch {epoch + 1}: train loss={running / len(loader):.4f}  "
              f"val macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}")
        if m["macro_f1"] > best_f1:  # keep the best epoch only
            best_f1, best_metrics = m["macro_f1"], m
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)       # -> model.safetensors + config.json (FP32)
            tokenizer.save_pretrained(out_dir)
            print(f"  [bert] saved best model to {out_dir}")

    return {"val": best_metrics, "params": {"model_name": model_name, **p}}


def load(directory=config.BERT_DIR, device: str = "cpu"):
    """Load the fine-tuned FP32 model + tokenizer (CPU by default, as in production)."""
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory).float().to(device)
    return model.eval(), tokenizer

