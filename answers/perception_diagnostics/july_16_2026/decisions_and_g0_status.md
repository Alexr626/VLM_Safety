# Perception diagnostics — decisions locked 2026-07-16

## User answers
1. Leading templates: Cursor drafts for Alex (`leading_clauses_v1.json`).
2. POPE full dump: all 600 pinned ids (3×200).
3. `max_new_tokens=512`: global (eval defaults + wrappers + this diagnostic).
4. Start Package A; do not block on S0 10-item gallery review.
5. Implement Package C now (offline scripts).
6. RunAI quota: check at S2; expect ~4 H100s when launching later.
7. A5 corpus: 2026-06-22 AMBER responses (Qwen2.5 high-β rotation/additive).
8. Continue into S0/S1 once G0 clears.

## G0 status (in progress)
| Item | Status |
|------|--------|
| A1/A2′/A3/A4 | Done — A1 shows mean-diff dominance; A2′ prefer nd200 |
| A5 | Fail rate 16.3% → LLM fallback **spec'd** (not built) |
| A6 | JSONLs + gallery written; templates are draft for Alex |
| A7 | Running on GPU 0 (Qwen2-VL demos_v2 all@nd200/500) |

## Paths
- Plan: `implementation_plans/EXPERIMENT_PLAN_perception_diagnostics.md`
- Root: `diagnostic_experiments/perception_diag/`
- Artifacts: `…/artifact_checks/`
- Augmented: `…/augment/outputs/`
