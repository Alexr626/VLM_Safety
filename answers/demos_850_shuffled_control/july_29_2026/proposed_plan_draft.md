# Proposed plan draft (blocked from implementation_plans/ until a spec exists)

Date: 2026-07-29

**Hook block:** every file under `implementation_plans/` must declare within its first 10 lines either
`extraction_spec: extractions/<id>_extraction.md` or `design_spec: designs/<id>_design.md`.
Neither exists yet for this shuffled-control demos_850 work. Below is the draft content for Alex
to attach to a new extraction (recommended) or to approve after writing the spec.

Suggested extraction id if Alex writes one:
`extractions/demos_850_shuffled_control_per_block_07_29_26_extraction.md`
— primitives only (control directions + caches + sanity gates). Geometric cosine plots would need
a **design** spec, not this extraction.

---

# demos_850 shuffled-image control directions (per partition block)

Date: 2026-07-29
Kind: extraction + construct-and-verify.
Depends on: completed demos_850 partition directions (`demos850_ba05bd96_*_partition`).
Prior art: `implementation_plans/7-22-26/shuffled_control_direction_sanity_checks_plan_2026-07-22.md`.

## Clarifying questions (answer before implement)

1. **Dimension:** `all` only (8 cells), or all five dimensions (40 cells)? Default below: **`all` only**.
2. **Geometric comparison:** follow-up plan after gating PASS (like 7-22 stage-2), or fold in now?
   Default: **follow-up** (needs a design spec).
3. **Derangement:** one independent derangement **per block size** over that block’s ids, seed **1234**?
   Default: **yes**. (Not one global 850-id derangement.)

## Purpose

For each partition block N ∈ {50, 100, 200, 500} and both LLaVA and Qwen2.5-VL, build an
image-deranged control that uses the **exact same ids, order, and captions** as
`demos850_ba05bd96_all_nd{N}_s42_r2_partition`, with images swapped within the block (derangement,
0 fixed points). Sanity-report only — no cosine, no behavioral runs.

## Grounded facts (2026-07-29)

- Pool hash `ba05bd960cad0c18`; partition blocks match deployed `all`/nd{N} `ids_used` exactly.
- Existing `shuffled_control.py` is hardcoded to demos_v2 `all`/nd200 — leave those artifacts alone.
- Cache key omits image → need a **new** act-cache namespace (not `textual_v2/_act_cache`, not legacy
  `shuffled_control/_act_cache`).

## Cells (assumed `all` only)

2 models × 4 sizes = 8. Qwen `--max_pixels 1003520` enforced.
Forwards: disjoint blocks ⇒ 850 unique ids × 2 variants = **1700**/model if sharing one demos850
shuffled act cache across N.

## Code approach

New module or demos850 mode that does not break demos_v2 `all_nd200` paths. Prefer
`shuffled_control_partition.py` + `extract_demos850_shuffled_control_directions.py`.

- Ids from deployed metadata / partition (assert order match). No `select_prefix_demos`.
- Per-N derangement, seed 1234, model-independent.
- Image override only; captions unchanged.
- Out: `experiment_artifacts/vti/{model_short}/shuffled_control_demos850/all_nd{N}/`
- Cache: `…/shuffled_control_demos850/_act_cache/`
- Derangement files: `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd{N}_s1234.json`

## Gating checks (same as 7-22, parameterized by N)

| Check | Expect |
|---|---|
| Fixed points | 0 |
| Captions unchanged | N of N |
| Id-order vs deployed `ids_used` | 0 mismatches |
| Shape | LLaVA (32,4096); Qwen (28,3584) |
| Cache accounting | total forwards/model = 1700 across four N; per-cell `forwards_executed` + `cache_hits` |
| Lineage | `control_for_deployed_slug = demos850_ba05bd96_all_nd{N}_s42_r2_partition` |

Non-gating: per-layer L2 norms vs deployed (well-formedness only).

## Out of scope (unless Q2 overrides)

Geometric cosine/magnitude (design-spec follow-up). `run_eval` wiring. Touching demos_v2 controls.

## Compute

lambdab2, one free A6000; LLaVA all N then Qwen all N.
