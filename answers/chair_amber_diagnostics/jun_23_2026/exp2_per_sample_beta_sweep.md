# Exp2 rotation-strength: per-sample beta sweep structure

**Date:** 2026-06-23

## Question

Does each sample run generation for every beta before moving on? Is "15% complete" = 15% of samples fully done across all betas?

## Answer: yes

The outer loop is **per sample** (500 CHAIR images). For each sample, the driver runs **all** generations for that image before advancing:

1. **Baseline** — one caption with no steering (`_gen`)
2. **Each β in the grid** — steered caption at that β (`_gen_steered`), classified vs baseline (empty / identical / changed)
3. **Two mitigation probes at β_max** (0.6): decode-only steering and skip-position-0 steering

For the current fine grid (`0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1`):

```
generations per sample = 1 baseline + 9 betas + 2 probes = 12
total for CHAIR sweep  = 500 × 12 = 6,000 generations
```

**75 / 500 samples in the checkpoint** means 75 images are **fully** complete (baseline + all 9 betas + both probes). It does **not** mean "75 images at β=0.6 only" or partial betas across the set.

Scoring is done **after** all samples are generated: metrics are computed column-wise per β across the completed sample set.

## Contrast with Exp1 (reproduction grid)

Exp1 (`run_exp1_repro_grid.sh`) uses `run_eval.py` with **β as the outer loop**: each cell is one (model, benchmark, intervention, β) run over all 500 images. Exp2 is the opposite structure — one JSON per (model, benchmark) with all β values embedded per sample.
