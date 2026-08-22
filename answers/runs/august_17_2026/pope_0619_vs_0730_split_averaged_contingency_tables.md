# Split-averaged contingency tables: 06-19 vs 07-30 LLaVA POPE

Date: 2026-08-17  
Question: Create contingency tables for the split-averaged 06-19 vs 07-30 matched comparison, from `evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/comparison_tables.md`.

Counts below are the **sum of the three POPE-split confusion matrices** in `cells.json` (random + popular + adversarial, 200 items each). `n_unparsed = 0` in every cell. Gold is balanced: 100 yes + 100 no per split → **300 gold-yes + 300 gold-no = 600**.

Each table is **gold × predicted** for one arm (a confusion matrix). It is **not** the paired 06-19-vs-07-30 table (both correct / one-only / both wrong). That joint is not determined by `comparison_tables.md` or by these marginals.

Accuracy, recall, gold-yes accuracy, and gold-no accuracy computed from these 2×2 tables match the split-averaged rates in `comparison_tables.md` (equal denominators per split, so mean-of-rates = pooled rate). **Precision does not:** the published precision column is the mean of three split precisions; the 2×2 precision is pooled `TP / (TP+FP)`. Both numbers are given.

---

## Baseline (`no_intervention`; identical on both dates)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 264 | 36 | 300 |
| **Gold no** | 56 | 244 | 300 |
| **total** | 320 | 280 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 508 / 600 | 84.67% |
| recall = gold-yes accuracy | 264 / 300 | 88.00% |
| gold-no accuracy | 244 / 300 | 81.33% |
| precision (pooled) | 264 / 320 | 82.50% |
| precision (split-averaged, published) | — | 83.06% |

---

## β = 0.2

### 06-19 (author nd70 PC1+mean)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 258 | 42 | 300 |
| **Gold no** | 56 | 244 | 300 |
| **total** | 314 | 286 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 502 / 600 | 83.67% |
| recall = gold-yes accuracy | 258 / 300 | 86.00% |
| gold-no accuracy | 244 / 300 | 81.33% |
| precision (pooled) | 258 / 314 | 82.17% |
| precision (split-averaged, published) | — | 82.75% |

### 07-30 (demos850 nd500 meandiff)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 267 | 33 | 300 |
| **Gold no** | 59 | 241 | 300 |
| **total** | 326 | 274 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 508 / 600 | 84.67% |
| recall = gold-yes accuracy | 267 / 300 | 89.00% |
| gold-no accuracy | 241 / 300 | 80.33% |
| precision (pooled) | 267 / 326 | 81.90% |
| precision (split-averaged, published) | — | 82.38% |

---

## β = 0.5

### 06-19 (author nd70 PC1+mean)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 261 | 39 | 300 |
| **Gold no** | 48 | 252 | 300 |
| **total** | 309 | 291 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 513 / 600 | 85.50% |
| recall = gold-yes accuracy | 261 / 300 | 87.00% |
| gold-no accuracy | 252 / 300 | 84.00% |
| precision (pooled) | 261 / 309 | 84.47% |
| precision (split-averaged, published) | — | 85.00% |

### 07-30 (demos850 nd500 meandiff)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 270 | 30 | 300 |
| **Gold no** | 70 | 230 | 300 |
| **total** | 340 | 260 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 500 / 600 | 83.33% |
| recall = gold-yes accuracy | 270 / 300 | 90.00% |
| gold-no accuracy | 230 / 300 | 76.67% |
| precision (pooled) | 270 / 340 | 79.41% |
| precision (split-averaged, published) | — | 79.93% |

---

## β = 0.9

### 06-19 (author nd70 PC1+mean)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 258 | 42 | 300 |
| **Gold no** | 30 | 270 | 300 |
| **total** | 288 | 312 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 528 / 600 | 88.00% |
| recall = gold-yes accuracy | 258 / 300 | 86.00% |
| gold-no accuracy | 270 / 300 | 90.00% |
| precision (pooled) | 258 / 288 | 89.58% |
| precision (split-averaged, published) | — | 89.91% |

### 07-30 (demos850 nd500 meandiff)

| | Predicted yes | Predicted no | total |
|---|---:|---:|---:|
| **Gold yes** | 258 | 42 | 300 |
| **Gold no** | 59 | 241 | 300 |
| **total** | 317 | 283 | 600 |

| | count | rate |
|---|---:|---:|
| accuracy | 499 / 600 | 83.17% |
| recall = gold-yes accuracy | 258 / 300 | 86.00% |
| gold-no accuracy | 241 / 300 | 80.33% |
| precision (pooled) | 258 / 317 | 81.39% |
| precision (split-averaged, published) | — | 82.06% |

---

## Compact count summary

| β | arm | TP | FN | FP | TN | correct | predicted yes |
|---|---|---:|---:|---:|---:|---:|---:|
| — | baseline | 264 | 36 | 56 | 244 | 508 | 320 |
| 0.2 | 06-19 | 258 | 42 | 56 | 244 | 502 | 314 |
| 0.2 | 07-30 | 267 | 33 | 59 | 241 | 508 | 326 |
| 0.5 | 06-19 | 261 | 39 | 48 | 252 | 513 | 309 |
| 0.5 | 07-30 | 270 | 30 | 70 | 230 | 500 | 340 |
| 0.9 | 06-19 | 258 | 42 | 30 | 270 | 528 | 288 |
| 0.9 | 07-30 | 258 | 42 | 59 | 241 | 499 | 317 |

Sources: `comparison_tables.md` (published rates), `cells.json` (per-split `tp/fp/tn/fn`, summed). Frame: LLaVA `vti_textual_additive_mlp`, all layers, POPE n=200/split.
