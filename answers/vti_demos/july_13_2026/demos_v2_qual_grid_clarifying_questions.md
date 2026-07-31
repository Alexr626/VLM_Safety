# Clarifying questions — demos_v2.1 qualitative grid plan

**Date:** 2026-07-13  
**Plan:** `implementation_plans/vti_demos_v2_qual_grid_plan.md`

## Pre-answers already confirmed from code/data (no need to re-ask)

1. **Per-dimension valid-pair counts:** all **555/555** rows carry non-empty `h_values` for `{existence, attribute, counting, relation, all}`. `nd=500` will not shrink for any dimension. Content hash matches `9a44f4afde0324b5`. Flat exports `demos_v2_{d}.jsonl` are **not** on disk yet — will use `demos_v2.jsonl` + a dimension selector (plan allows this).
2. **Token-position policy (existing textual extractor):** `forward_vl(image, question + " " + caption)`; PCA input is the **last token** of the full sequence at **every** hidden-state row (`embedding + all decoder layers`), i.e. `hidden_states[layer][0, -1, :]`. Will document unchanged in `IMPLEMENTATION.md`.
3. **`run_evaluation` model reuse:** one `run_evaluation()` call loads the wrapper **once**, then loops benchmarks × interventions. **β is not looped in-process** today (single `--beta` per process). Grouping as plan suggests — one process per `(model, dim, nd, β)` covering 4 interventions × 2 benchmarks — matches current plumbing. Optional in-process β loop is an enhancement, not required for correctness.
4. **Qualitative subset source exists:** 2026-06-22 LLaVA sample bundles under `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/{amber,chair}/` (AMBER-25 `ordered` + CHAIR-5). Will materialize `data/vti/qual_subset_chair5_amber25.json` from those.
5. **Qwen 6/22 Exp1 `max_pixels`:** `run_eval.py` / Exp1 driver have **no** `--max_pixels` plumbing; Exp1 Qwen cells used the wrapper default (`None` / native). Tonight’s pin `1003520` is a deliberate delta vs 6/22 (VRAM sharing). Exp2 already has the pattern to mirror.

## Blockers needing your call

### Q1 — Diff polarity (plan vs live extractor)

- **Plan:** `diff_i = act(h_values[d]) − act(value)` (hallucinated − truthful).
- **Live `obtain_textual_vti`:** stacks `act(value) − act(h_value)` (truthful − hallucinated), then builds the direction.

Which polarity should demos_v2 extraction use?

**Recommendation:** keep the **live extractor polarity** (`value − h_value`) so `+β` steering matches author-demo behavior; document that the plan’s formula was inverted relative to code. Or flip to the plan’s formula if you want “subtract hallucination direction” semantics explicitly — but then `+β` meaning changes vs prior runs.

### Q2 — PCA algorithm (plan vs live extractor)

Live textual path is **not** per-layer pure PC1:

- Flatten each demo’s last-token stack to one `(num_layers+1)*hidden_dim` vector.
- Fit global PCA (`rank`); reconstruct as `(components.sum(dim=1) + mean).mean(0)` and reshape — i.e. **legacy PC+mean**, not unit PC1.
- Sign handling is only `svd_flip` inside `PCA.fit` (max-abs column of `U`); **no** mean-diff sign-align (unlike the visual arm’s `_sign_align`).

Plan asks for:

- **Per-layer** PCA on `(N, hidden_dim)`, keep **PC1 + PC2**, steer with **PC1 only**.
- Sign-align to mean diff (visual-style).
- New cache under `textual_v2/…`.

Confirm the **new** demos_v2 path should follow the **plan** (per-layer pure PC1/PC2 + mean-diff sign-align), leaving the legacy author-demo `obtain_textual_vti` path untouched for old caches. (That is the natural reading; flagging because “same convention as existing extractor” in the plan does not match what the textual code actually does.)

### Q3 — `run_date` for the overnight grid

Plan example is `2026-07-14`. Today is **2026-07-13**. Use `2026-07-14` (shared overnight stamp) or `2026-07-13`?

### Q4 — Scope tonight: implement-only vs implement + launch

Should I:

**(A)** implement plumbing + materialize pin/order files + update `IMPLEMENTATION.md`, then **hand you the exact launch commands** after the sanity gate, or  
**(B)** implement and **launch** the concurrent overnight extraction+grid on the free A6000 once the smoke gate passes?

### Q5 — Free GPU index

Plan: single free A6000, both models concurrent, same `CUDA_VISIBLE_DEVICES`. Which GPU index should I pin (after you confirm `nvidia-smi`), or may I pick the freest card at launch time?
