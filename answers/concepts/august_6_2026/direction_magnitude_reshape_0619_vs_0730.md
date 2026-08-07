# Reshape and magnitude: 06-19 PC1+mean vs 07-30 meandiff directions

Date: 2026-08-06

## Does reshape change magnitude?

**No.** After `PC1 + mean` is formed in the flattened space of length `(L+1)·H`, `.view(L+1, H)` only rearranges the same entries. The flat L2 / Frobenius norm is identical before and after reshape. Per-layer L2 norms are the norms of those contiguous slices; reshape does not rescale them.

## How do stored magnitudes compare? (LLaVA-1.5-7B, decoder rows only)

From on-disk caches under `experiment_artifacts/vti/llava-1.5-7b-hf/`:

| Direction | flat L2 | mean per-layer L2 |
|-----------|---------|-------------------|
| 06-19 author demos, nd70, **PC1+mean** | 15.77 | 2.18 |
| 07-30 demos850 meandiff nd50 | 8.36 | 1.07 |
| 07-30 demos850 meandiff nd100 | 11.26 | 1.44 |
| 07-30 demos850 meandiff nd200 | 9.63 | 1.23 |
| 07-30 demos850 meandiff nd500 | 9.93 | 1.26 |

Per-layer, the 06-19 vector is about **1.4×–5×** larger in L2 than the 07-30 meandiff vectors (ratio larger in early layers, smaller late). That comparison mixes **different demos** with **PC1+mean vs meandiff**.

Holding demos fixed (demos850 `*_r2_partition` PC1+mean vs `*_meandiff_partition` on the same block):

| nd | flat L2 meandiff | flat L2 PC1+mean | ratio | mean layer cosine |
|----|------------------|------------------|-------|-------------------|
| 50 | 8.36 | 9.11 | 1.09 | 0.9975 |
| 100 | 11.26 | 12.07 | 1.07 | 0.9990 |
| 200 | 9.63 | 10.44 | 1.08 | 0.9987 |
| 500 | 9.93 | 10.79 | 1.09 | 0.9989 |

So on the same demos, adding PC1 to the mean only increases flat L2 by ~7–9% and barely turns the direction (layer cosines ≈ 1). Most of the 06-19 vs 07-30 magnitude gap is therefore from **different demo inputs**, not from reshape or from PC1 alone.

## Does stored magnitude matter at eval?

For `additive` (and the rotation variants), `steer()` unit-normalizes the per-layer direction before scaling by `beta`:

```43:43:evaluation/interventions/vti/steer.py
    d = F.normalize(direction.float(), dim=-1)
```

So differences in the **stored** vector’s magnitude do not change additive step size at a fixed `beta`; only the direction (and `beta`) do.
