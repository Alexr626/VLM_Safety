# Qwen AMBER-1500 RunAI progress / ETA (2026-08-06 ~13:50 ET)

Live check via `runai training exec amber-qwen-beta-triple`. Job Running, H100 ~34 GiB / 100% util, two concurrent `run_eval.py` (β=0.5 and β=0.9). β=0.2 worker already `WORKER_OK`.

## Cell status (19 expected: 1 baseline + 18 steered)

| Slice | Status |
|-------|--------|
| baseline | done |
| β=0.2 all 6 steered | done |
| β=0.5 meandiff ×3 | done |
| β=0.5 PCA `layers_all` | ~1425/1500 in flight |
| β=0.5 PCA `5-14`, `15-24` | not started |
| β=0.9 meandiff `layers_all` | done |
| β=0.9 meandiff `5-14` | ~301/1500 in flight |
| β=0.9 meandiff `15-24` + PCA ×3 | not started |

Observed rates while sharing the GPU: ~0.58–0.60 samples/s (β=0.5), ~0.62–0.64 samples/s (β=0.9).

## ETA

Critical path is **β=0.9**: ~7200 samples left at ~0.62/s ≈ **~3.2–3.5 h** from the check (plus small per-cell overhead). Even if β=0.5 finishes in ~1.5 h and β=0.9 then speeds up alone, total is still on the order of **~3 h**.

**Will not finish in the next 1.5 hours before the meeting.**

β=0.5 alone is roughly meeting-time (~1.5 h); the full Qwen grid waits on β=0.9’s remaining five cell-equivalents.
