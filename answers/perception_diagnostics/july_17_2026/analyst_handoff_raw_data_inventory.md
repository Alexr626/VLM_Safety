# Analyst handoff — Perception diagnostics: raw data inventory & reset ask

**Date:** 2026-07-17  
**Audience:** Claude analyst (via Romanus)  
**Purpose:** Inventory what AMBER-25 dump runs actually produced, what the original hypotheses were, and which plan/artifact jargon should be retired so a simpler re-analysis can be designed from raw captures — not from the overbuilt Package C report stack.

---

## Why this handoff

The active plan (`implementation_plans/EXPERIMENT_PLAN_perception_diagnostics(1).md`) compressed **four hypotheses (H-A…H-D)** plus **Package A gates, multi-cell steering grid, Package C offline suite (C1–C8), and multi-model smokes** into one markdown. Implementation followed that plan. The result is a large artifact tree whose filenames and metric names are plan-coded (`B5`, `C3`, `d_lead`, `fire_fraction`, etc.) and hard for a human or agent to digest.

**Ask of the analyst:** Propose a *smaller* viewing/reporting plan that starts from the **raw dump fields** below, answers the *core* research questions with plain-English metrics/plots, and does not assume the C1–C8 stack must be kept.

---

## Initial hypotheses (as written in the plan)

One stated goal: characterize what the deployed textual VTI direction encodes and what **rotational** steering does to decision-relevant representations, in the discriminative (yes/no) regime.

| Code | Plain-language intent | Confirm / falsify (plan wording) |
|------|------------------------|----------------------------------|
| **H-A** | Does steering move representations along a “leading / sycophantic prompt” axis? | Confirm: leading-vs-neutral linear probes show rising dose-response on **steered but neutrally prompted** states as strength ↑ at high-AUC layers. Falsify: probes work on natural leading/neutral data but steered-neutral stays flat. |
| **H-B** | Does steering mainly *translate* the decision cloud without changing yes/no separability? | Confirm: gold-label probe AUC flat across strength while cloud mean moves and first-position yes−no logit margin shifts. Falsify: gold-probe AUC moves materially with strength. |
| **H-C** | Do models differ in how first-token yes/no margins shift under steering? | Confirm: Δ(yes−no) vs baseline tight & negative (LLaVA-1.5), tight & positive (Qwen2.5-VL), ≈0 (Qwen2-VL). Falsify: broad item-dependent shifts, or Qwen2-VL not inert. |
| **H-D** | Is the extracted direction valid on held-out demos? | Confirm: 355 held-out demo pairs separate along the direction (projection AUC ≫ 0.5) under two readouts. Falsify: ≈ chance → stop. |

**Related locked facts already decided in Package A (not re-litigated here):** deployed direction ≈ per-layer **mean-diff (CAA)** with rotational application (not “PC1”); demos_v2 `all@nd200`, slug `demosv2_9a44f4af_all_nd200_s42_r2_prefix`.

Romanus’s framing of the *original* single interest: **steering vector effect on VLM sycophancy** (leading vs neutral prompts) — H-A is the closest match; H-B/C/D and the full cell grid expanded scope.

---

## What was run (dumps with raw data)

### LLaVA-1.5-7B × AMBER-25 — primary local dump

| Path | Notes |
|------|--------|
| `diagnostic_experiments/perception_diag/llava-1.5-7b-hf/dumps/s1_full_cell_smoke/` | Full cell grid B0–B12; **25 items × 5 prompt conditions = 125 rows/cell**; `max_new_tokens=512` |
| `…/dumps/s0_smoke/` | Smaller precursor (B0, B2, B8) |

**Steering cells (plan codes → meaning):**

| Code | Method | Hook site | Strength β |
|------|--------|-----------|------------|
| B0 | none (baseline) | — | 0 |
| B1–B3 | `uniform_rotation` | `mlp` | 0.2 / 0.5 / 0.9 |
| B4–B6 | `uniform_rotation` | `layer` | 0.2 / 0.5 / 0.9 |
| B7–B9 | `additive` | `mlp` | 0.2 / 0.5 / 0.9 |
| B10–B12 | `additive` | `layer` | 0.2 / 0.5 / 0.9 |

**Factual note already in logs:** on LLaVA S1, rotation×layer at high β produces long / unparseable generations (B5/B6 especially). Review plots were later redrawn **excluding B5/B6 overlays**; JSON analyses still contain those cells.

### Qwen2.5-VL-7B × AMBER-25 — complete (2026-07-17)

| Path | Notes |
|------|--------|
| `…/qwen2.5-vl-7b-instruct/dumps/s3b_qwen25_amber25/` | Smoke cells B0–B9 + B11; 125 rows/cell; `max_new_tokens=128`; includes `degeneracy_flag` / `truncated` / `status` |

Headline C1 auto-summary (pooled; see `c1_readout_validity.json`): B0 unp=0.064 mass_p50=0.820; B6 unp=0.576 mass_p50=0.083 deg=0.616 trunc=0.576; mid rotation cells in between.

### Not yet / incomplete for this ask

- Qwen2-VL S3a RunAI smoke dump (helper exists; depends on cluster submit).
- H-D held-out QC dump (`hd_qc/run_hd_qc.py` → terminator / mean-pool activations) — **not** present for LLaVA S1 Package C C7.
- Full AMBER-450 / POPE-600 S4 dumps — not the subject of this inventory.

### Human-readable response galleries (not metrics)

`evaluation/results/2026-07-16/_samples/llava-1.5-7b-hf/amber/` — HTML per cell + compare-neutral + index (model text responses under each steering cell × prompt condition).

---

## Prompt conditions (5) — the “leading template” axis

Built by `augment/build_augmented_jsonl.py` from `templates/leading_clauses_v1.json` into  
`diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl` (25 items).

| `condition_id` | Meaning |
|----------------|---------|
| `neutral` | Question only (no leading clause) |
| `tentative_toward_yes` | Soft clause biasing toward yes, then question |
| `tentative_toward_no` | Soft clause biasing toward no, then question |
| `assertive_toward_yes` | Strong clause biasing toward yes, then question |
| `assertive_toward_no` | Strong clause biasing toward no, then question |

Clause is **prefixed before** the question. `template_id` and gold label (`yes`/`no`) are logged per row.

---

## Per-forward raw capture (what is on disk)

For each `(item_id, condition_id, cell)`:

### 1. Manifest row (`manifest.jsonl`) — text + first-position scores

Typical S1 fields:

| Field | Content |
|-------|---------|
| `item_id`, `condition_id`, `template_id`, `gold`, `qtype`, `cell_id` | Identity |
| `response` | Generated string |
| `parsed_outcome` | `yes` / `no` / `unparseable` |
| `first_vs_parsed_agree` | Whether first-token yes/no score agrees with parsed response |
| `score_yes_logit_sum`, `score_no_logit_sum` | Aggregated first-position variant logits |
| `score_logit_margin_yes_minus_no` | yes − no logit margin |
| `score_p_yes_raw`, `score_p_no_raw`, `score_answer_mass` | Softmax mass on yes/no variants; `answer_mass` ≈ p_yes_raw + p_no_raw |
| `score_p_yes_norm` | P(yes) renormalized over {yes,no} |
| `score_first_token_pred` | Argmax among yes/no variants |
| `score_prefill_seq_len` | Prefill length |

**S3b / newer dumps** also write `degeneracy_flag`, `truncated`, `status` ∈ {ok, oom, error}. **S1 LLaVA manifests often lack those three columns** (pre-v2 writer).

### 2. Activations (`acts/{item_id}__{condition_id}.npy`)

- Shape `(n_layers+1, hidden)` — **last prefill token**, all layers including embedding index 0, `float16`.
- LLaVA-1.5-7B: `(33, 4096)` → 32 decoder layers + embed.

### 3. Prefill position norms (`norms/{item_id}__{condition_id}.npy`)

- Shape `(n_layers+1, seq_len)` — ℓ2 norm of hidden state at each prefill position × layer, `float16`.
- Useful for position-0 / last-token norm profiles without re-running the model.

### 4. Run metadata (`run_metadata.json`)

Model id, direction slug, demos hash, augmentation hashes, `max_new_tokens`, item count, etc.

**Direction tensors** (not inside the dump; shared cache):  
`experiment_artifacts/vti/{model_short}/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/`.

---

## Derived Package C products (optional; currently overbuilt)

Offline scripts under `diagnostic_experiments/perception_diag/analyze/` wrote JSON + plots under the dump, e.g.:

- `c1_readout_validity.json` … `c8_scatter_auc_bundle.json`
- `package_c_plots/`, `c3_plots/`, `c6_plots/`

These encode the plan’s C1–C8 checklist (readout validity, leadedness probes, supervised-frame scatters, decision-state probes, logit-lens shifts, norm profiles, H-D QC, scatter/AUC bundle). **They are derivatives**; the analyst may redesign metrics without treating C1–C8 as required.

HTML index (LLaVA S1):  
`…/s1_full_cell_smoke/package_c_plots/index.html`

---

## Jargon / naming that should change

Per updated research_workflow **Artifact naming**: filenames and plot labels should be plain English; plan codes only as secondary tags.

### Plan / cell codes (keep as IDs in metadata, not as primary names)

| Current | Prefer |
|---------|--------|
| B0, B1, … | `baseline`, `rotation_mlp_beta_0.5`, `rotation_layer_beta_0.2`, `additive_mlp_beta_0.9`, … |
| C1–C8 | Drop as artifact titles; name the measurement (e.g. “first-token yes/no agreement by steering setting”) |
| H-A…H-D | Fine in prose once; avoid as plot titles |
| S0/S1/S3a/S3b/S4, G0/G1/G1b | Run tags only |
| Package A/B/C | Internal workflow labels — not reader-facing |

### Metric / plot jargon to retire or rewrite

| Current | Plain-English alternative |
|---------|---------------------------|
| `d_lead` | leading-prompt probe direction / axis |
| `y_steer` | steering-direction component orthogonal to leading axis |
| `fire_fraction` | probe-positive rate (fraction with positive probe score) |
| `plane_capture_frac` | fraction of mean displacement energy in the 2D frame |
| `answer_mass` | combined first-token probability mass on yes+no variants |
| `first_vs_parsed_agree` | first-token score agrees with parsed response |
| `leadedness` | leading-vs-neutral (sycophancy-related) prompt contrast |
| `transfer` (C2) | probe score on steered runs that used the **neutral** prompt |
| `translation-vs-separability` | cloud mean shift vs gold-label probe AUC |
| `logit-lens` (here) | first-position yes−no logit margin (not full logit-lens over layers) |
| `CAA` / `δ̄` / `eps_coeff` | mean-diff direction; blend coefficient (keep technical once in methods) |
| Filenames like `c3_L27.png`, `c2_fire_fraction_vs_beta.png` | e.g. `supervised_frame_layer_27.png`, `probe_positive_rate_vs_rotation_strength.png` |

### Over-condensation risk (for the new plan)

Avoid packing into one plan: multi-model grids × full β × additive+rotation × probe suites × H-D QC × HTML galleries. Prefer **one primary question per plan**, small-N AMBER-25 raw-data views first, then scale.

---

## Suggested starting materials for the analyst (no new runs required)

1. LLaVA S1 dump tree + `run_metadata.json` (above).  
2. Augmented JSONL for prompt text: `augment/outputs/augmented_amber25.jsonl`.  
3. Response galleries under `evaluation/results/2026-07-16/_samples/…`.  
4. This file + `IMPLEMENTATION.md` § Perception diagnostics (interfaces).  
5. Treat Package C plots as *examples of what was tried*, not as the required deliverable set.

---

## Explicit non-goals for the redesign chat

- Do not assume S4 full dumps exist.  
- Do not require C7 until an H-D QC dump is run.  
- Do not invent new dump columns; if something is missing, say so.  
- Keep interpretation out of factual run records; new viewing plan should specify **what to plot/count** in plain English.
