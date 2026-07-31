# Why shuffled-control looked "too fast"

Date: 2026-07-29 ~07:12

## Short answer

You are not misunderstanding the design. Shuffled controls **do** need new forwards (wrong image + same captions). Those forwards **did** run. What looked instant is mostly **sparse logging** (one line per finished N), not skipped GPU work.

## Evidence

LLaVA finished all four sizes. Per-cell metadata:

| N | forwards | cache_hits |
|---|---:|---:|
| 50 | 100 | 0 |
| 100 | 200 | 0 |
| 200 | 400 | 0 |
| 500 | 1000 | 0 |

Total **1700** forwards; `shuffled_control_demos850/_act_cache` has **1700** files. Those files are **not** copies of `textual_v2/_act_cache` (spot-check same id `value` variant: different sha256).

Qwen still running at check time (on nd=500; cache ~1338/1700).

## Why it feels fast

1. Log only prints after each full N finishes — no per-example lines — so 50→100→200 look like they appeared suddenly.
2. 1700 LLaVA forwards on an A6000 is on the order of ~10–15 minutes wall clock (similar ballpark to the earlier demos850 LLaVA extract of ~2100 new forwards).
3. PCA fits after caching are CPU and cheap; they are not the long pole.

## Where we are in the process

For each N: load deployed block ids → derange images within block → **forward value+all with swapped images into a dedicated cache** → PCA → sanity gates. No reuse of the truthful-image `textual_v2` activations.
