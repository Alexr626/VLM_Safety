# Clarifying questions — perception diagnostics plan (pre-launch)

**Date:** 2026-07-16  
**Plan:** `implementation_plans/EXPERIMENT_PLAN_perception_diagnostics.md`  
**Override accepted:** all applicable `max_new_tokens` for this diagnostic set to **512** (ignore plan’s “no max_new_tokens changes” note).

## Already confirmed from repo (CURSOR-CONFIRM draft answers)

| # | Item | Finding |
|---|---|---|
| 1 | AMBER-450 | `data/amber/pinned_amber_disc_450.json` (n=450) |
| 1 | AMBER-25 smoke | amber ids in `data/vti/qual_subset_chair5_amber25.json` (n=25) |
| 1 | POPE pin | `data/pope/pinned_eval_ids.json` — 200/split × {random, popular, adversarial} = **600** |
| 2 | Direction slug freshness | LLaVA + Qwen2.5-VL already have `textual_v2/demosv2_9a44f4af_{all}_nd{200,500}_s42_r2_prefix` (hash matches demos_v2.1). Qwen2-VL has **only** legacy `textual_directions_nd70_rank1_seed42.npz` → A7 required |
| 5 | lambdab2 `/data` headroom | ~735G free on `/data` — fine for 10–30 GB dump |

## Open questions for Romanus / Alex

1. **Leading-clause templates (A6 blocker).** D2 specifies 5 conditions but the plan has no clause text. Who supplies the versioned templates — draft here and wait for Alex review, or is there a separate template file/spec?

2. **POPE full-dump scope.** Use all 600 pinned ids (3×200), or a single split / smaller subset?

3. **`max_new_tokens=512` scope.** Only new `perception_diag` dump/eval scripts, or also change global `run_eval.py` / existing driver defaults?

4. **Gating / wait points.** Start Package A (CPU: A1–A5, A7 GPU when free) immediately, but **hard-stop before S0 smoke** until Alex reviews the A6 gallery (G0)?

5. **Package C timing.** Implement C1–C8 offline analysis scripts in parallel with Package A, or only after the dump exists?

6. **RunAI concurrency.** Plan assumes ~4 free H100s / 6 jobs in ~1.5 waves; `IMPLEMENTATION.md` notes `nlm-mh` quota is typically **2 GPUs**. Proceed assuming 2-concurrent and longer wall-clock, or check/raise quota at S2?

7. **A5 parser-audit corpus.** Which existing result trees to sample for the ~300 long-form responses (bias: Qwen + high-β rotation-layer)? e.g. `2026-07-13` demos_v2 qual, chair/amber diagnostics, POPE β grids?

8. **This session’s stop line.** Implement through G0 only (Package A + plumbing + A7), then pause for review — or keep going through S0/S1 lambdab2 smoke once G0 clears?
