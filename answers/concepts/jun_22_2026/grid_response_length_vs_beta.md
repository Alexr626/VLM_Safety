# Response length vs β — reproduction grid (additive & uniform_rotation_mlp)

Date: 2026-06-22. Question: does output **character length** change with β for the grid interventions (`additive_mlp`, `additive_layer`, `uniform_rotation_mlp`), analogous to the residual-stream `uniform_rotation` sweep length tables in RESEARCH_LOG.md?

Computed retroactively from `evaluation/results/2026-06-19/{model}/pope_{split}/{iv}__b{beta}/responses.json` as mean of `len(response)` (characters), same unit as the sweep's `mean_len`. n=200/split, greedy decode, `max_new_tokens` per the eval config. `baseline` = `no_intervention` (β-independent).

> Compare against the residual-site `uniform_rotation` sweep (POPE random only) where length ranged, e.g., LLaVA 87.2→3.6 and Qwen2.5 154.3→250.1 over β=0.1→0.6.

---

## Split-averaged (mean of random/popular/adversarial)

### `additive_mlp` — mean response length (chars), averaged over splits

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.4 | 79.9 | 74.7 | 71.0 | 65.0 | 58.8 | 51.0 | 40.8 | 32.8 | 24.1 | 12.4 |
| Qwen-VL-Chat | 64.2 | 64.9 | 65.5 | 65.9 | 65.5 | 65.3 | 64.5 | 64.4 | 63.5 | 63.1 | 62.7 |
| Qwen2-VL-7B | 47.5 | 48.4 | 49.4 | 49.3 | 49.8 | 51.1 | 52.4 | 52.6 | 54.2 | 55.3 | 55.3 |
| Qwen2.5-VL-7B | 154.2 | 155.1 | 155.6 | 154.2 | 157.2 | 155.5 | 158.9 | 157.3 | 159.0 | 159.4 | 158.9 |

### `additive_layer` — mean response length (chars), averaged over splits

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.4 | 79.9 | 74.2 | 71.1 | 65.2 | 58.8 | 51.1 | 40.9 | 32.7 | 24.1 | 12.3 |
| Qwen-VL-Chat | 64.2 | 65.7 | 65.2 | 65.9 | 65.8 | 65.3 | 65.6 | 64.5 | 64.1 | 63.4 | 63.3 |
| Qwen2-VL-7B | 47.5 | 48.0 | 48.8 | 49.5 | 51.0 | 50.4 | 51.1 | 52.8 | 53.8 | 54.0 | 56.5 |
| Qwen2.5-VL-7B | 154.2 | 153.7 | 155.7 | 155.7 | 156.4 | 156.5 | 157.1 | 158.9 | 159.4 | 159.9 | 158.6 |

### `uniform_rotation_mlp` — mean response length (chars), averaged over splits

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.4 | 79.7 | 75.5 | 71.6 | 66.8 | 58.3 | 50.4 | 44.2 | 39.8 | 37.9 | 35.9 |
| Qwen-VL-Chat | 64.2 | 63.6 | 62.2 | 59.9 | 55.6 | 53.8 | 51.7 | 49.0 | 47.5 | 46.3 | 45.8 |
| Qwen2-VL-7B | 47.5 | 49.7 | 51.6 | 54.7 | 59.9 | 63.7 | 69.5 | 75.7 | 81.0 | 84.4 | 90.6 |
| Qwen2.5-VL-7B | 154.2 | 156.0 | 158.8 | 157.2 | 160.3 | 159.4 | 160.9 | 163.2 | 163.7 | 167.7 | 185.5 |

---

## Per-split

### `additive_mlp`

**Split: random**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 87.2 | 83.9 | 80.0 | 76.8 | 70.0 | 63.7 | 54.1 | 41.6 | 33.8 | 25.0 | 13.6 |
| Qwen-VL-Chat | 64.1 | 64.7 | 65.1 | 65.7 | 65.2 | 64.9 | 64.1 | 64.0 | 63.3 | 62.9 | 62.1 |
| Qwen2-VL-7B | 49.9 | 49.8 | 51.0 | 50.8 | 51.2 | 52.7 | 53.7 | 53.5 | 54.3 | 55.3 | 54.1 |
| Qwen2.5-VL-7B | 155.8 | 157.8 | 156.9 | 156.3 | 158.5 | 157.4 | 159.4 | 158.9 | 159.6 | 160.2 | 160.1 |

**Split: popular**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.5 | 78.6 | 72.2 | 68.1 | 62.3 | 56.6 | 49.2 | 39.2 | 31.1 | 21.9 | 10.9 |
| Qwen-VL-Chat | 63.6 | 64.5 | 64.7 | 65.2 | 64.7 | 64.8 | 64.3 | 64.0 | 62.6 | 62.1 | 62.0 |
| Qwen2-VL-7B | 47.3 | 48.8 | 49.6 | 49.4 | 50.1 | 51.4 | 53.4 | 53.7 | 55.3 | 56.9 | 58.0 |
| Qwen2.5-VL-7B | 152.8 | 153.2 | 153.6 | 153.0 | 155.1 | 153.4 | 158.2 | 155.5 | 157.7 | 158.0 | 157.8 |

**Split: adversarial**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 81.4 | 77.3 | 71.8 | 68.2 | 62.5 | 56.1 | 49.7 | 41.5 | 33.4 | 25.2 | 12.6 |
| Qwen-VL-Chat | 65.0 | 65.6 | 66.6 | 66.8 | 66.4 | 66.2 | 65.0 | 65.1 | 64.6 | 64.4 | 63.9 |
| Qwen2-VL-7B | 45.2 | 46.7 | 47.7 | 47.6 | 47.9 | 49.2 | 50.3 | 50.7 | 53.0 | 53.6 | 53.9 |
| Qwen2.5-VL-7B | 154.0 | 154.2 | 156.3 | 153.3 | 158.1 | 155.8 | 159.0 | 157.5 | 159.7 | 159.8 | 159.0 |

### `additive_layer`

**Split: random**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 87.2 | 84.0 | 79.3 | 76.9 | 69.8 | 63.7 | 54.1 | 41.9 | 33.8 | 25.0 | 13.4 |
| Qwen-VL-Chat | 64.1 | 65.4 | 65.0 | 65.4 | 65.5 | 64.9 | 65.3 | 64.2 | 64.0 | 63.1 | 63.1 |
| Qwen2-VL-7B | 49.9 | 50.0 | 50.0 | 51.3 | 51.9 | 52.4 | 51.9 | 53.9 | 54.9 | 54.5 | 55.5 |
| Qwen2.5-VL-7B | 155.8 | 156.0 | 157.6 | 156.6 | 157.2 | 157.0 | 158.5 | 160.8 | 160.5 | 160.8 | 160.1 |

**Split: popular**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.5 | 78.6 | 71.6 | 68.3 | 62.8 | 56.6 | 49.5 | 39.4 | 31.1 | 21.9 | 10.9 |
| Qwen-VL-Chat | 63.6 | 65.1 | 64.5 | 65.5 | 65.3 | 64.9 | 65.2 | 64.2 | 63.4 | 62.7 | 62.2 |
| Qwen2-VL-7B | 47.3 | 48.0 | 49.2 | 49.6 | 51.7 | 51.0 | 52.0 | 53.5 | 54.8 | 55.5 | 59.1 |
| Qwen2.5-VL-7B | 152.8 | 152.3 | 153.5 | 154.6 | 155.0 | 155.0 | 155.7 | 156.3 | 157.9 | 158.6 | 157.5 |

**Split: adversarial**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 81.4 | 77.2 | 71.6 | 68.0 | 63.0 | 56.3 | 49.7 | 41.5 | 33.3 | 25.2 | 12.6 |
| Qwen-VL-Chat | 65.0 | 66.5 | 66.1 | 66.8 | 66.5 | 66.1 | 66.4 | 65.1 | 64.9 | 64.4 | 64.7 |
| Qwen2-VL-7B | 45.2 | 46.0 | 47.1 | 47.5 | 49.4 | 47.9 | 49.4 | 50.8 | 51.8 | 52.1 | 54.8 |
| Qwen2.5-VL-7B | 154.0 | 152.9 | 155.9 | 155.8 | 157.2 | 157.4 | 157.0 | 159.4 | 159.7 | 160.4 | 158.1 |

### `uniform_rotation_mlp`

**Split: random**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 87.2 | 84.2 | 80.3 | 77.0 | 73.1 | 64.5 | 55.8 | 47.3 | 41.7 | 38.4 | 36.3 |
| Qwen-VL-Chat | 64.1 | 63.6 | 61.5 | 59.4 | 55.3 | 53.6 | 51.4 | 49.1 | 47.8 | 46.5 | 45.9 |
| Qwen2-VL-7B | 49.9 | 51.7 | 52.5 | 55.0 | 59.6 | 63.0 | 69.1 | 75.2 | 81.4 | 84.2 | 90.8 |
| Qwen2.5-VL-7B | 155.8 | 159.6 | 162.7 | 159.0 | 163.2 | 163.1 | 166.1 | 167.9 | 168.8 | 170.9 | 188.7 |

**Split: popular**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.5 | 78.1 | 72.7 | 68.4 | 63.4 | 54.8 | 46.6 | 41.5 | 38.9 | 37.3 | 35.2 |
| Qwen-VL-Chat | 63.6 | 62.7 | 61.8 | 59.0 | 55.0 | 53.1 | 51.0 | 48.6 | 46.8 | 45.5 | 45.2 |
| Qwen2-VL-7B | 47.3 | 49.7 | 52.1 | 55.7 | 61.5 | 66.0 | 71.7 | 77.4 | 82.2 | 85.9 | 92.2 |
| Qwen2.5-VL-7B | 152.8 | 153.9 | 155.9 | 156.8 | 158.8 | 156.6 | 159.1 | 161.1 | 160.6 | 165.6 | 183.9 |

**Split: adversarial**

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 81.4 | 77.0 | 73.6 | 69.3 | 63.9 | 55.7 | 48.7 | 43.8 | 39.0 | 37.9 | 36.0 |
| Qwen-VL-Chat | 65.0 | 64.5 | 63.2 | 61.3 | 56.5 | 54.7 | 52.6 | 49.4 | 48.1 | 46.8 | 46.3 |
| Qwen2-VL-7B | 45.2 | 47.7 | 50.0 | 53.4 | 58.5 | 62.0 | 67.9 | 74.3 | 79.4 | 83.1 | 88.8 |
| Qwen2.5-VL-7B | 154.0 | 154.5 | 157.8 | 155.9 | 158.8 | 158.4 | 157.6 | 160.8 | 161.8 | 166.6 | 183.8 |

---

## Factual summary

- `additive_mlp` ≈ `additive_layer` in length at every β (per-cell differences ≤ ~1.5 chars) — the two additive sites move length essentially identically.
- **Additive** changes length almost only for **LLaVA-1.5** (84.4 → ~12 chars at β=1.0, monotone decrease). The Qwen family is largely unaffected: Qwen-VL-Chat ~flat (64.2 → 62.7), Qwen2-VL slight rise (47.5 → 55.3), Qwen2.5-VL ~flat (154.2 → 158.9).
- **`uniform_rotation_mlp`** has a larger and more consistent effect, with **opposite sign by model**: LLaVA-1.5 84.4 → 35.9 and Qwen-VL-Chat 64.2 → 45.8 (shorten with β); Qwen2-VL 47.5 → 90.6 and Qwen2.5-VL 154.2 → 185.5 (lengthen with β).
- The per-model **sign** of the `uniform_rotation_mlp` length effect **matches the residual-site `uniform_rotation` sweep** (RESEARCH_LOG Run B): negative for LLaVA-1.5 / Qwen-VL-Chat, positive for Qwen2-VL / Qwen2.5-VL. Additive does not reproduce this model-specific length signature — so the rotation geometry carries a model-specific length signature at *both* hook sites, whereas additive's length effect is concentrated in LLaVA.

