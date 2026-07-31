# Answers — demos_v2 qual grid clarifying questions

**Date:** 2026-07-13

| Q | Resolution |
|---|------------|
| Q1 polarity / `h_value` | `value` = truthful caption; `h_values[d]` (or flat `h_value`) = hallucinated caption for dimension `d`. Live path uses `act(value) − act(h_value)`. |
| Q2 PCA | Keep **live** textual path (global flatten + `PC1 + mean`); cache rank-2 components; do **not** switch to per-layer pure PC1 tonight. |
| Q3 run_date | `2026-07-13` |
| Q4 scope | Implement **and** launch overnight (B) |
| Q5 GPU | GPU **0** (only free A6000) |

Implemented and launched; see `RESEARCH_LOG.md` entry 2026-07-13.
