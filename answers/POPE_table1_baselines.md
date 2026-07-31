# POPE Baselines — Table 1 Format (VTI-comparable)

**Benchmark:** POPE (random + popular + adversarial, averaged)  
**Intervention:** none (vanilla / no-intervention)  
**Samples:** 200 per split × 3 splits = 600 per model  
**Elicitation:** free generation, greedy (`do_sample=False`)  
**Metrics:** accuracy and F1; positive class = yes  

Source: `evaluation/results/{model}/pope/no_intervention/metric_summary.json` (`accuracy_overall`, `f1_overall` over all 600 records).  
Run date: 2026-06-16. See `RESEARCH_LOG.md` for full commands and per-split breakdown.

---

## Table 1 — POPE object hallucination (avg. over 3 splits)

| Model | Acc ↑ | F1 ↑ |
|:------|------:|-----:|
| LLaVA-1.5-7B | 84.7 | 85.2 |
| Qwen-VL-Chat | 85.7 | 85.1 |
| Qwen2-VL-7B-Instruct | 88.0 | 87.4 |

*Values are percentages (×100), one decimal place.*
