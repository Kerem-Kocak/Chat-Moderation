# 🛡️ Real-Time Chat Moderation (SWE015 – Use Case 4)

**Student Name:** Kerem KOÇAK  
**Student ID:** 2309111056  
**Course:** SWE015 – Introduction to Large Language Models (*Building an Intelligent Application*)  
**2-Min Demo Video:** I shared the link in blackboard only for my professor. 
**Pretrained Model Weights (`artifacts.zip`):** https://drive.google.com/file/d/1YKd2OniAwFz9xI_ahvVesYFvjs6q1NET/view?usp=sharing  

Flags offensive chat messages **live on a CPU-only server**, masks them, and logs how long each model took to respond.
The same task is solved by a ladder of models of increasing sophistication, and all of them are compared on **the same held-out test set**:

| # | Rung | Model | Area |
|---|---|---|---|
| 0 | Rule baseline (extra) | Lexicon mined from training data + minimum edit distance | B |
| 1 | Classical baseline | TF-IDF (word 1–2-grams + char 3–5-grams) + logistic regression | B |
| 2 | Neural network | **Own word2vec** (skip-gram + negative sampling, PyTorch) → feed-forward NN trained with SGD + backprop | A |
| 3 | Transformer | `bert-base-uncased` fine-tuned for classification | C |
| 4 | Quantized Transformer | BERT with **dynamic INT8 quantization**: what the app runs | D |

All models share the same **regex normalization** (B): URLs, @mentions, hashtags, leetspeak (`$h1t` → `shit`), censoring stars (`f*ck`), repeated letters (`stuuupid` → `stuupid`), emojis.

## Data & References
TweetEval / *offensive* = OffensEval 2019 (SemEval-2019 Task 6A), English tweets, official splits (train 11,916 / validation 1,324 / test 860), downloaded automatically from the Hugging Face Hub (`cardiffnlp/tweet_eval`).

* **OffensEval 2019 Dataset:** Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019). *SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval).* In *Proceedings of SemEval-2019*.
* **TweetEval Benchmark:** Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* In *Findings of EMNLP 2020*.
* **BERT Architecture (`bert-base-uncased`):** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* In *Proceedings of NAACL-HLT 2019*.
* **Skip-gram Word2Vec:** Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* In *Advances in NIPS 2013*.

## Project structure
```
chat-moderation/
├── app.py                   # Gradio app (UI only) - runs the INT8 BERT by default
├── requirements.txt
├── src/
│   ├── config.py            # all paths + hyper-parameters
│   ├── data.py              # download / cache / describe the dataset
│   ├── preprocessing.py     # (B) regex normalization + tokenizer
│   ├── models/
│   │   ├── lexicon_matcher.py  # (B) lexicon + Levenshtein rule baseline
│   │   ├── tfidf_baseline.py   # (B) TF-IDF + logistic regression
│   │   ├── w2v_ffn.py          # (A) own word2vec + feed-forward NN
│   │   ├── bert_classifier.py  # (C) BERT fine-tuning
│   │   └── quantize.py         # (D) dynamic INT8 quantization
│   ├── predictors.py        # one predict() interface for all models (benchmark + app)
│   ├── moderation.py        # app logic: classify + time + mask (UI-independent)
│   ├── benchmark.py         # (D) accuracy / macro-F1, size (MB), latency (ms)
│   └── train_all.py         # trains the whole ladder
└── results/                 # results table, predictions, hardware info
```

## Setup
Python 3.10–3.13.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app (single command)
Download the trained weights (see below) into `artifacts/`, then:
```bash
python app.py
```
Open http://127.0.0.1:7860. Use `python app.py --share` for a temporary public link.

## Reproduce training and results
```bash
# 1) Train everything: data download, lexicon, TF-IDF, word2vec+FFN, BERT, INT8 quantization
#    (BERT takes ~1 h on a laptop CPU, a few minutes on a GPU / Colab)
python -m src.train_all

# 2) Benchmark all models on the test set (CPU, batch size 1, 200 runs)
python -m src.benchmark
```
If BERT was trained on another machine (e.g. Colab), re-run the quantization on the CPU that serves the model:
`python -m src.models.quantize`.

Quick end-to-end check of the code in ~1 minute (tiny data + a 2-layer BERT; numbers are meaningless):
```bash
# Windows PowerShell: $env:SMOKE_TEST="1"; python -m src.train_all; python -m src.benchmark
SMOKE_TEST=1 python -m src.train_all && SMOKE_TEST=1 python -m src.benchmark
```

## Pretrained weights
Weights are not stored in this repository. Download `artifacts.zip` from **[Google Drive (Pretrained Weights)](https://drive.google.com/file/d/1YKd2OniAwFz9xI_ahvVesYFvjs6q1NET/view?usp=sharing)** and unzip it into `chat-moderation/` so the trained models are placed inside `chat-moderation/artifacts/`.

## Results (Held-Out Test Set: 860 Tweets)

Measured by `python -m src.benchmark` on a **13th Gen Intel Core i7-13650HX** (CPU-only, 1 PyTorch thread, batch size 1, mean of 200 single-message runs, latency budget = **50 ms**):

| Model | Rung | Precision | Accuracy | Macro-F1 | Size (MB) | Mean Latency (ms) | p95 Latency (ms) | ≤ 50 ms Budget |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| Lexicon + Edit Distance | Rule baseline | n/a | 0.7698 | 0.6851 | 0.00 | 0.05 | 0.08 | ✅ yes |
| TF-IDF + LogReg | Classical baseline | FP64 | 0.7826 | 0.7409 | 2.27 | 1.11 | 1.37 | ✅ yes |
| Word2Vec + FFN | Neural network | FP32 | 0.7128 | 0.6514 | 3.48 | 0.21 | 0.27 | ✅ yes |
| BERT-base (FP32) | Transformer | FP32 | **0.8535** | **0.8160** | 417.67 | 67.21 | 104.86 | ❌ no |
| **BERT-base (INT8)** | **Quantized Transformer** | **INT8 (dynamic)** | **0.8523** | **0.7935** | **173.09** | **30.07** | **44.37** | **✅ yes** |

Dynamic INT8 quantization shrinks BERT-base by **2.41×** (`417.67 MB → 173.09 MB`) and speeds up CPU inference by **2.24×** (`67.21 ms → 30.07 ms`), bringing both mean and p95 latency under the 50 ms real-time budget while preserving 85.23% accuracy.

## AI Usage Statement (Requirement R8)
* **AI coding tools used:** Google Antigravity IDE assistant (Claude Opus & Gemini models).
* **Parts assisted by AI:** Scaffolding modular Python scripts (`src/preprocessing.py`, `src/models/*`, `src/benchmark.py`), assistance with building the Gradio interface (`app.py`) and polishing English text.
* **Student verification & responsibility:** AI tools were used as development assistants throughout the project. I was actively involved in the development process, reviewed and tested the generated code and also verified the experimental results. All models were trained and evaluated on the real OffensEval dataset. Every metric, model size, latency number and failure example comes directly from our measured runs (`results/results.csv`, `results/training_log.json`, `results/predictions.csv`).

