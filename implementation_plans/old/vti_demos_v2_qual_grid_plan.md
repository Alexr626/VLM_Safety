# Experiment plan — demos_v2.1 textual directions + qualitative-subset steering grid

**Date:** 2026-07-13 (overnight run)
**Author:** analyst (via Romanus)
**Target:** lambdab2, single free A6000 (`CUDA_VISIBLE_DEVICES` per `nvidia-smi`), both model pipelines running **concurrently on the same GPU**
**Scale / purpose:** N=5 (CHAIR) + N=25 (AMBER) qualitative subsets; qualitative shakedown of demos_v2.1-derived directions. NOT a benchmark run. The pinned CHAIR-500/AMBER-450 reproduction moves to RunAI tomorrow (separate plan).

---

## Goal / hypotheses

Characterize textual VTI steering vectors extracted from `data/vti/demos_v2.jsonl` (v2.1, 555 finals, `content_hash_sha256_16=9a44f4afde0324b5`) before scaling:

1. **H1 (plumbing/efficacy):** demos_v2-derived directions steer at all — intervention responses visibly differ from baseline at matched β. If `additive` variants at every β produce responses identical to baseline, suspect broken extraction/plumbing (check direction norms and hook firing) before concluding "no effect".
2. **H2 (sample size):** direction quality varies with `num_demos` ∈ {50, 100, 200, 500} — visible in response quality and in per-layer PC1 explained-variance ratios.
3. **H3 (dimension specificity):** per-dimension vectors (existence / attribute / counting / relation) produce dimension-consistent qualitative effects vs the composed `all` vector.
4. **Prior finding to re-check qualitatively:** rotation variants act as a bias knob, not grounding (h−=0 history); additive should look different in raw responses.

Deliverable for the morning: side-by-side HTML galleries for the **`all`-vector slice** on both models. Per-dimension slices continue after and may finish later — acceptable.

---

## Part A — direction extraction (NEW code, extends `evaluation/interventions/vti/directions.py` path)

### Source data

- Demos file: `data/vti/demos_v2.jsonl` (555 rows). Record filename + content hash `9a44f4afde0324b5` in all cache metadata.
- Per-image pairs, per dimension `d ∈ {existence, attribute, counting, relation, all}`:
  `diff_i = act(h_values[d]) − act(value)` (hallucinated minus truthful), one diff per image, using the **existing textual extraction path**: `forward_vl(image, question + caption)` with the same question (`"Describe this image in detail."` = the row's `question` field) and the same token-position policy as the current author-demo extractor. **Do not change the token policy; document it in IMPLEMENTATION.md** (see Open questions).
- `all` = the composed `h_values.all` variant (single 4-edit caption). **Not** a pooled union of per-dimension pairs.
- Consuming the existing flat exports (`export_vti_flat.py --dimension {d}` → `data/vti/demos_v2_{d}.jsonl`) is acceptable if it lets the existing loader run unchanged — but the **selection policy below overrides the current seed-42 `random.sample`**.

### Nested subsampling (NEW selection policy: `shuffled_prefix`)

- Build ONE master order: sorted final `id`s from `demos_v2.jsonl`, shuffled once with seed **42**. Write it to a tracked file `data/vti/demos_v2_order_s42.json` (list of ids + demos hash in `_meta`).
- For each `num_demos` N ∈ {50, 100, 200, 500}: take the **first N ids valid for that dimension** in master order (if a row lacks a verified `h_values[d]`, skip it and continue down the order; record skipped ids). Nested prefixes isolate the sample-size effect.
- **Extract activations once** per (model, image, caption-variant) for the union needed (500-prefix superset, 6 captions per image: `value` + 5 variants), cache to disk (reuse `ActivationCache` conventions, suffix per variant, e.g. `tt_v2_{d}` / `tt_v2_value` — Cursor's choice, but cache so PCA reruns are free). PCA per (dimension, N) then runs on cached matrices. Do not re-forward per N.

### PCA (rank 2)

- Per layer: stack diffs → (N, hidden_dim), PCA, keep **PC1 and PC2**.
- Sign-align components using the **same convention as the existing extractor** (presumably align to mean diff) so that steering with +β behaves identically to the author-demo directions. Document the convention.
- **Steering uses PC1 only** (component 0). PC2 is cached for tomorrow's projection diagnostic, never steered tonight.
- Log per config: per-layer explained-variance ratios for PC1/PC2, per-layer PC1 norm (should be 1 if normalized — log pre-normalization mean-diff norm too), `n_pairs` actually used, skipped ids.

### Cache (fixes known follow-up #1 — demos identity in slug)

- NEW namespace: `experiment_artifacts/vti/{model_short}/textual_v2/{slug}/directions.npz` + `metadata.json`.
- Slug must encode: demos filename + content hash, dimension, num_demos, seed, rank, selection policy. E.g. `demosv2_9a44f4af_{dim}_nd{N}_s42_r2_prefix`.
- The legacy author-demo cache at `experiment_artifacts/vti/{model_short}/` must be untouched and never read for these runs.
- `metadata.json`: demos path, content hash, dimension, N, ids used, skipped ids, question string, token policy, sign convention, explained-variance summary, model_short, date, git commit.

### Models

| HF id | model_short | dtype | Notes |
|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | `llava-1.5-7b-hf` | float16 | |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2.5-vl-7b-instruct` | bfloat16 | routes via `Qwen2VLWrapper`; pin `max_pixels` (see Part B) for VRAM headroom under GPU sharing |

- 5 dimensions × 4 sizes = 20 direction configs per model; 40 cached sets total.
- **Single GPU, unsharded** (device policy — hooks corrupt under sharding/offload). Both models may run concurrently on the one free A6000; steady-state ~15 GiB each. If OOM occurs during concurrent extraction, serialize (LLaVA first) rather than shard.

### Extraction ordering (so `all` evals start early)

Per model process:
1. Forwards for `value` + `h_values.all` over the 500-prefix (2 × ≤500 forwards) → PCA for `all` × {50,100,200,500} → **hand off to Part B `all` slice immediately**.
2. Forwards for the remaining 4 variants (4 × ≤500) → PCA for the other 16 configs, concurrently with the `all` eval slice or after it (Cursor's judgment on GPU contention; extraction forwards are short prefill calls).

Rough cost: ~3000 prefill forwards per model total; expect on the order of 1–2 h per model on a shared A6000 (estimate, not measured).

---

## Part B — evaluation grid (qualitative subsets)

### Subsets (NEW tracked pin file)

- Ids: the **LLaVA-derived 2026-06-22 qualitative bundle** — the same AMBER-25 (`ordered`) + CHAIR-5 draw the visual-smoke harness pinned (source: `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/{amber,chair}/...`). Used identically for **both** models (cross-model comparability on identical images).
- Materialize once to a tracked file `data/vti/qual_subset_chair5_amber25.json`, shape `{"chair": [...], "amber": [...], "_meta": {...}}`, consumable by `run_eval.py --subset_ids_file`.

### Fixed run facts

| Fact | Value |
|---|---|
| `--run_date` | one shared date for the whole overnight run (e.g. `2026-07-14`) |
| Benchmarks | `chair`, `amber` (registry keys); `--amber_task discriminative` |
| `--chair_prompt` | `"Please Describe this image in detail."` (verbatim, capital D) |
| `--chair_max_new_tokens` | **512** (deviation from the 6/22 frozen 64 — deliberate; recorded; 6/22 CHAIR numbers are NOT comparators) |
| `--max_new_tokens` | 256 (AMBER path, unchanged) |
| Interventions | `vti_textual_additive_layer`, `vti_textual_additive_mlp`, `vti_textual_uniform_rotation_layer`, `vti_textual_uniform_rotation_mlp` |
| β grid | {0.2, 0.5, 0.9} |
| Vector configs | dimension ∈ {all, existence, attribute, counting, relation} × num_demos ∈ {50, 100, 200, 500}, PC1 steering |
| Baselines | `no_intervention` per (model, benchmark) once, same subsets/caps, same run_date |
| Qwen `max_pixels` | pin `1003520` via wrapper kwarg for VRAM headroom under GPU sharing; **record in config + RESEARCH_LOG**; check what 6/22 Exp1 used (Open questions) |
| Greedy decoding | unchanged (`do_sample=False` everywhere) |

Cell count per model: 5 dims × 4 sizes × 3 β × 4 interventions × 2 benchmarks = **480 intervention cells** + 2 baseline cells. Tiny N per cell (5 captions / 25 yes-no), so wall-clock is dominated by process/model-load overhead — group accordingly (below).

### NEW plumbing required (fixes known follow-up #2)

1. `run_eval.py` / `run_evaluation()` gain: `--demos_path`, `--vector_dimension` (or equivalent via per-dimension flat-export paths), `--num_demos`, `--rank`, forwarded to `VTITextualIntervention` (constructor already accepts `demos_path`, `num_demos`, `rank`, `seed`). NEW intervention behavior: `shuffled_prefix` selection (Part A) and **steer with PC1 only when rank=2 is cached**.
2. Result-dir suffix must encode the vector config or the 20 configs collide: `{iv}__b{beta}__d{dim}__nd{N}` (bare `no_intervention` unchanged). Extend the existing `__b{beta}` logic in `eval_runner`.
3. `intervention.config` logs gain: `demos_path`, demos content hash, `dimension`, `num_demos`, `rank`, `steer_component`, selection policy.
4. Qwen `max_pixels` plumbing through `run_eval.py` → `create_wrapper` (Qwen2 wrappers only — base `__init__` rejects unknown kwargs). Mirror the Exp2 script's handling if `run_eval.py` lacks it.

### Grouping / driver (NEW bash driver, e.g. `evaluation/run_scripts/run_demosv2_qual_grid.sh`)

- One driver parameterized by `MODEL`; launch two instances concurrently on the free GPU (`CUDA_VISIBLE_DEVICES` identical for both).
- Group one `run_eval.py` process per (model, dimension, num_demos, β) covering all 4 interventions × both benchmarks in one `--interventions ... --benchmarks chair amber` call → 8 cells per model load, 60 processes per model, instead of 480 loads. If `run_evaluation` can loop β in-process without reloading the model, further grouping is fine — Cursor's call; correctness first.
- **Order:** baselines first, then the full `all` slice (12 processes/model), then dimensions in fixed order existence → attribute → counting → relation. β order within a slice: 0.5, 0.2, 0.9 (most-informative first if the night is cut short).
- `--skip_if_exists` everywhere; everything resumable under the shared run_date.
- Rough wall-clock (estimate): ~5–10 min/process → `all` slice ≈ 1.5–2.5 h/model after its directions land; full grid ≈ 6–10 h/model. Two models concurrent on one A6000 ≈ similar wall-clock if VRAM holds (~15 GiB each steady; Qwen capped). OOM fallback: serialize models, LLaVA first.

### Known-degenerate cells (keep, do not "fix")

LLaVA `uniform_rotation_layer` collapses to empty generations well below β=0.9 (RESEARCH_LOG 2026-06-18, 6/22 Exp2 collapse onset). Expect empty CHAIR captions there; §C empty-handling reports them (`empty_fraction`). Leave the cells in for the record.

### Sanity gate before launching the grid

After the first `all` directions land (per model): one smoke generation — 1 CHAIR image, `vti_textual_additive_mlp`, β=0.5, dim=all, nd=500 — confirm (a) response differs from baseline, (b) `metadata.json` carries the demos hash, (c) result dir name carries `__d{dim}__nd{N}`. Only then launch the grid.

---

## Part C — morning deliverable: HTML galleries

- Extend `helper_scripts/render_smoke_review.py` (or a thin sibling) to auto-discover tonight's cell dirs (`{iv}__b{beta}__d{dim}__nd{N}`) instead of the baked 16-dir smoke order.
- Render **one gallery per (model, benchmark, dimension)**: baseline + that dimension's 48 cells side-by-side per image, base64-embedded, under `evaluation/results/{run_date}/_samples/{model_short}/{benchmark}/`.
- Render the `all` galleries **immediately after the `all` slice completes** (both models), so they exist in the morning regardless of grid progress. Re-render / add per-dimension galleries as slices finish (end-of-driver hook).
- `metric_summary.json` files are still produced (free) but are explicitly secondary: N=5/25 — do not read accuracy/precision tables as results. The galleries are the deliverable.

---

## Logging

- Local only. **No W&B** (deferred; public benchmarks only anyway).
- `RESEARCH_LOG.md` (Cursor appends facts): extraction record (demos hash, per-dimension valid-pair counts, skipped ids, explained-variance headlines, cache slugs, commands, git commit) and eval record (run_date, driver commands, cell counts completed, deviations: CHAIR cap 512, qualitative subsets, Qwen `max_pixels`, GPU-sharing arrangement, any OOM/serialization fallback).

---

## Confirm / falsify

- **H1 confirmed** if intervention responses visibly differ from baseline across the `all` slice; **falsified-or-broken** if additive cells are byte-identical to baseline at all β → check direction norms + hook firing before interpreting.
- **H2 signal:** monotone-ish trends across nd ∈ {50→500} in explained-variance ratios and/or response quality; a null (nd=50 ≈ nd=500 qualitatively) is itself informative for tomorrow's RunAI sizing.
- **H3 signal:** per-dimension vectors visibly modulate their own dimension in AMBER-25 responses (existence vector → existence questions) more than others. This is qualitative triage only; the quantitative version is the RunAI run + tomorrow's PCA projection diagnostic.

---

## Open questions (Cursor: confirm and update IMPLEMENTATION.md before/while implementing)

1. **Per-dimension valid-pair counts** among the 555 finals — do all rows carry all five `h_values`? If any dimension has <500 valid pairs, its nd=500 config shrinks to the actual count; report counts.
2. **Token-position policy** of the existing textual extractor (which positions feed PCA) and the **PC sign-alignment convention** — document both; extend unchanged to rank 2.
3. **Qwen 6/22 Exp1 resolution:** did the 6/22 `run_eval.py` Qwen cells run with a `max_pixels` cap, and via what mechanism? Tonight pins `1003520` regardless (VRAM sharing), but record the delta vs 6/22 if any.
4. **`run_evaluation` model reuse:** does one call reuse a loaded wrapper across interventions/benchmarks (it should — confirm), and can β be looped in-process? Determines final grouping granularity.
5. **Flat exports vs `dimension` kwarg:** consume `demos_v2_{d}.jsonl` flat exports or read `demos_v2.jsonl` with a dimension selector — Cursor's choice; either way the `shuffled_prefix` policy and the master-order file are authoritative for selection.
6. Anything missing from IMPLEMENTATION.md that this plan needed and guessed at — add it rather than silently diverging.
