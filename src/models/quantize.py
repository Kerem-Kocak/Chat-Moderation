"""
(D) Efficient models - post-training dynamic INT8 quantization of BERT.

How dynamic quantization works
------------------------------
* Every ``nn.Linear`` layer (attention Q/K/V/output projections, the two
  feed-forward layers in each of the 12 blocks, the pooler and the classifier
  head - ~85 % of BERT's parameters) is converted to ``DynamicQuantizedLinear``.
* Weights are stored as 8-bit integers with a scale and zero point:
      w_fp32  ~  scale * (w_int8 - zero_point)          (4 bytes -> 1 byte)
* Activations are quantized "dynamically": at inference time the scale of
  each input tensor is computed on the fly, the matrix multiplication runs in
  INT8 on the CPU and the result is converted back to FP32.
* No retraining and no calibration data are needed, and it runs on any CPU.
* Embedding and LayerNorm layers stay FP32, which is why the model shrinks
  by ~2.4x and not by the full 4x.

Run this step on the machine that will serve the model (your laptop),
because the INT8 kernels depend on the CPU backend:

    python -m src.models.quantize
"""
from __future__ import annotations

import warnings

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.models import bert_classifier
from src.utils import classification_metrics, file_size_mb

# torch.ao.quantization is marked "deprecated" in recent PyTorch versions in
# favour of the separate `torchao` package, but it still works and is the
# simplest, dependency-free way to do dynamic INT8 quantization.
warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*")

INT8_FILE = "model_int8.pt"


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Replace all nn.Linear layers with INT8 dynamically quantized versions."""
    model = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def save_quantized(qmodel: nn.Module, tokenizer, out_dir=config.BERT_INT8_DIR) -> None:
    """Save the INT8 state_dict + model config + tokenizer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out_dir / INT8_FILE)
    qmodel.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def load_quantized(directory=config.BERT_INT8_DIR):
    """
    Rebuild the INT8 model: create an empty BERT from its config, apply the
    same quantization so the module structure matches, then load the INT8
    weights. (The FP32 weights are NOT needed at this point.)
    """
    cfg = AutoConfig.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_config(cfg)
    qmodel = quantize_dynamic_int8(model)
    # weights_only=False: the INT8 state_dict holds packed-parameter objects,
    # not only plain tensors. Only load files that you created yourself.
    state = torch.load(directory / INT8_FILE, map_location="cpu", weights_only=False)
    qmodel.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    return qmodel.eval(), tokenizer


def quantize_and_save(splits: dict[str, pd.DataFrame] | None = None) -> dict:
    """Load the FP32 BERT, quantize it, save it, and compare size + val accuracy."""
    model, tokenizer = bert_classifier.load(config.BERT_DIR, device="cpu")
    qmodel = quantize_dynamic_int8(model)
    save_quantized(qmodel, tokenizer)

    fp32_mb = file_size_mb([config.BERT_DIR / "model.safetensors"])
    int8_mb = file_size_mb([config.BERT_INT8_DIR / INT8_FILE])
    n_lin = sum(isinstance(m, nn.Linear) for m in model.modules())
    print(f"  [quantize] {n_lin} nn.Linear layers -> INT8 "
          f"(backend: {torch.backends.quantized.engine})")
    print(f"  [quantize] size FP32 {fp32_mb:.1f} MB -> INT8 {int8_mb:.1f} MB "
          f"({fp32_mb / int8_mb:.2f}x smaller)")

    result = {"params": {"method": "dynamic INT8 (torch.ao.quantization.quantize_dynamic)",
                         "backend": torch.backends.quantized.engine,
                         "fp32_mb": round(fp32_mb, 1), "int8_mb": round(int8_mb, 1)}}
    if splits is not None:  # quick sanity check that quantization kept the accuracy
        val = splits["validation"]
        qmodel, tok = load_quantized()  # reload from disk = exactly what the app uses
        pred = bert_classifier.predict_proba_bert(qmodel, tok, val["text"].tolist()).argmax(1)
        result["val"] = classification_metrics(val["label"], pred)
        print(f"  [quantize] INT8 val macro-F1={result['val']['macro_f1']:.4f}")
    return result


if __name__ == "__main__":
    from src.data import load_splits
    quantize_and_save(load_splits())
