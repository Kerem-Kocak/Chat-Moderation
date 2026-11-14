| model | precision | accuracy | macro_f1 | size_mb | latency_mean_ms | latency_p95_ms | meets_budget |
|---|---|---|---|---|---|---|---|
| Lexicon + Edit Distance | n/a | 0.7698 | 0.6851 | 0.0 | 0.05 | 0.08 | yes |
| TF-IDF + LogReg | FP64 | 0.7826 | 0.7409 | 2.27 | 1.11 | 1.37 | yes |
| Word2Vec + FFN | FP32 | 0.7128 | 0.6514 | 3.48 | 0.21 | 0.27 | yes |
| BERT-base (FP32) | FP32 | 0.8535 | 0.816 | 417.67 | 67.21 | 104.86 | no |
| BERT-base (INT8) | INT8 (dynamic) | 0.8523 | 0.7935 | 173.09 | 30.07 | 44.37 | yes |

Hardware: 13th Gen Intel(R) Core(TM) i7-13650HX, 1 thread(s), batch size 1, mean of 200 runs. Test set: 860 tweets.
