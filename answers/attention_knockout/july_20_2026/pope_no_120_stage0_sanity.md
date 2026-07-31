# POPE-no-120 Stage 0 vs POPE-yes-120 (2026-07-20)

## Setup
- **POPE-yes-120** (unchanged content): gold=yes, misleading=`assertive_toward_no`
- **POPE-no-120** (new): gold=no, misleading=`assertive_toward_yes`; same `filler_b` / neutral

## First-token decidedness (plan check #2) — both sets

| Set | LLaVA | Qwen |
|-----|-------|------|
| POPE-yes-120 | 0.9917 | 1.0000 |
| POPE-no-120 | 1.0000 | 0.9944 |

Both ≥ ~99% → **pass**.

## Flip unions (gate ≥15)

| Set | LLaVA | Qwen |
|-----|-------|------|
| POPE-yes-120 | 6 (fail) | 24 (pass) |
| POPE-no-120 | **2** (fail) | **0** (fail) |

POPE-no-120 did **not** give LLaVA a larger flip set; Qwen has zero flips on gold=no + assertive_toward_yes.

Artifacts: `…/results/stage0_pope_no_120/`, dumps `…/dumps/pope_no_120_baseline/`.
