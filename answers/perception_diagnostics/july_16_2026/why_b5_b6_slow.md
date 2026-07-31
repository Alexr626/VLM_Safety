# Why B5/B6 are slow (S1 smoke)

**Date:** 2026-07-16  
**Context:** S1 full-cell smoke, LLaVA AMBER-25.

B5/B6 are `uniform_rotation` at the **layer** site with β=0.5 / 0.9 — the collapse-prone site flagged in the plan.

Measured on completed S1 manifests:

| Cell | site / β | mean response chars | median words | chars≥800 |
|------|----------|---------------------|--------------|-----------|
| B0–B4 | baseline / mlp or layer@0.2 | ~100 | ~20 | 0 |
| B5 | layer @ 0.5 | **1005** | 27 | **65/125** |
| B6 (partial) | layer @ 0.9 | **~1011** | **1** | all so far |

Wall-clock: B0–B4 ~2.5 min/cell; B5 ~25 min/cell (~60s/item).

Cause: layer-site high-β rotation pushes decode into long / degenerate strings that often run to `max_new_tokens=512` (user override). Generation cost dominates; each item still also does prefill capture + logits forwards.
