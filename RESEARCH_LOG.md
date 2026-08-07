# RESEARCH_LOG.md

Append-only, chronological record of the VLM hallucination-mitigation research.

**Conventions**
- Cursor (implementation agent) appends **factual run records**: date, commit, exact commands, output paths, headline metrics, deviations, errors.
- Romanus + analysis agent append **interpretation / decision** entries.
- Do not rewrite history. New entries go at the bottom. Numbers without a command + path are not trusted.

---

## Project context (standing — not a run record)

**Two tracks at Bell Labs (summer 2026):**
1. *Applied:* Fault Management (FM) Text-to-SQL for the TTYN system. Out of scope for this repo/log.
2. *Research (this repo):* hallucination mitigation in VLMs, with a downstream application to counting/object-identification of hardware parts in technician equipment-box photos.

**Research thesis under development:** geometry-aware activation steering for VLM hallucination mitigation. Lineage: contrastive difference-of-means steering (CAA) → applied to VLMs (VTI) → norm-preserving geodesic rotation instead of additive steering (Spherical Steering, LLM-only so far). Open gap: geometry-aware steering has never been applied to a VLM, or to counting. Direction is training-free and prioritizes causal/mechanistic explainability.

**Near-term plan (phased):**
1. Reproduce VTI as a baseline intervention (additive steering) on POPE — establishes a comparable reference and validates the harness. **Textual arm done; vision arm not implemented** (see `IMPLEMENTATION.md` § Visual encoder hooks).
2. Norm-discriminability diagnostic (Spherical Steering Fig. 3 analog): per-layer activation norm, truthful vs hallucinated, on a VLM. One-day go/no-go for the rotation premise.
3. Geometry-aware (norm-preserving rotation) steering intervention; evaluate vs VTI baseline + general-capability control.
4. Extend to counting (probe where counting lives across encoder/projector/LLM taps; test for a counting-error direction).
5. **Implement vision-encoder hooks** (VTI α / `alpha_image` arm): `VisionDispatch`, per-patch ViT activation capture/steering, registry `vti_visual_*`, starting LLaVA-1.5 then Qwen2.5-VL.

**Evaluation suite (target):** POPE, CHAIR, AMBER, MMHal-Bench, HallusionBench + a general-capability control (MMBench/MMStar). POPE first (unambiguous yes/no metric). **CHAIR-s/i and AMBER discriminative scorers implemented (2026-06-22);** CHAIR+AMBER VTI diagnostics run 2026-06-22 (partial — see entry below).

**Models:** LLaVA-1.5-7B, Qwen-VL-Chat, Qwen2-VL / Qwen2.5-VL-7B-Instruct (+ InternVL2/2.5 planned). Comparisons are **per-model** (baseline vs method, method vs SOTA) — cross-model comparison is not a goal.

**Standing constraints / conventions:**
- **Resolution policy (Policy A):** each model runs at its native/default vision resolution; frozen per model across baseline and intervention. Record the per-model vision-token count in each entry.
- **Device policy:** accuracy/scoring runs may use CPU offload or multi-GPU. **Activation-extraction runs (VTI direction-fitting, norm/rotation diagnostics) MUST run with the full model on a single device** — offload/sharding corrupts intermediate activations (mixed-device tensors, bf16↔fp32 precision boundaries, and accelerate placement hooks colliding with intervention hooks). RunAI or a free single A6000 for those.
- **Within-model consistency:** a model's intervention runs must match the environment of its baseline (resolution + device config), or the baseline is re-run.
- **Confidentiality:** benchmarks are public. Do not log internal/Nokia images, schemas, or part data to W&B or external services without explicit instruction.
- **Compute:** lambdab2 (4× A6000 48GB, shared, Ampere sm_86, driver 535 / CUDA 12.2); lambdav1 (4× A4000 16GB, Ampere — too small for a 7B VLM single-card without offload); RunAI (Kubernetes, reserved for single-device + heavy jobs).

---

## Entries

### 2026-06-16 — POPE baselines, no intervention (3 models)

**Type:** factual run record
**Commit:** `f81fcc00926c0b82c60c2a6a27f3c26ce26e1fbb` (local uncommitted changes applied before run: `load_pope` split filter, `Qwen2VLWrapper` Policy A `max_pixels`, POPE acc/F1 comparison table)
**Environment:** lambdab2, `CUDA_VISIBLE_DEVICES=0` (NVIDIA RTX A6000), conda env `vlm_hallucination_mitigation`, transformers 4.50.1
**Device config:** LLaVA-1.5 and Qwen2-VL-7B: single-device `cuda:0` (~13–15 GiB allocated). Qwen-VL-Chat: CPU offload (`device_map=auto`, vision pinned GPU ~5.9 GiB, LLM partial GPU+CPU).
**Eval config:** free-generation, `max_new_tokens=256`, greedy (`do_sample=False`); scorer = first-token-priority yes/no + word-boundaried fallback (`metrics.score_pope_records`); positive class = yes. Metric set: accuracy / precision / recall / F1 (VTI-comparable). Query template: `"Is there a [object] in the image?"` — confirmed verbatim from `data/pope/combined.json` / official POPE JSON.
**Splits:** random, popular, adversarial — `--limit 200` each (600 records/model). Records accumulated in one `responses.json` per model; final `metric_summary.json` has all three splits under `by_category`.

**Commands:**
```bash
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/romanus/huggingface
for MODEL in "llava-hf/llava-1.5-7b-hf" "Qwen/Qwen-VL-Chat" "Qwen/Qwen2-VL-7B-Instruct"; do
  for split in random popular adversarial; do
    python evaluation/run_eval.py \
      --model "$MODEL" \
      --benchmarks pope --interventions no_intervention \
      --pope_split $split --limit 200
  done
done
```

**Output paths:**
- `evaluation/results/llava-1.5-7b-hf/pope/no_intervention/{responses.json, metric_summary.json}`
- `evaluation/results/qwen-vl-chat/pope/no_intervention/{responses.json, metric_summary.json}`
- `evaluation/results/qwen2-vl-7b-instruct/pope/no_intervention/{responses.json, metric_summary.json}`

**Results:**

| Model | Vision tokens | Split | Acc | Prec | Recall | F1 | yes_ratio | n_unparsed |
|-------|---------------|-------|-----|------|--------|----|-----------| -----------|
| llava-1.5-7b-hf | 576 | random | 0.890 | 0.898 | 0.880 | 0.889 | 0.490 | 0 |
| llava-1.5-7b-hf | 576 | popular | 0.865 | 0.854 | 0.880 | 0.867 | 0.515 | 0 |
| llava-1.5-7b-hf | 576 | adversarial | 0.785 | 0.739 | 0.880 | 0.804 | 0.595 | 0 |
| llava-1.5-7b-hf | 576 | **avg** | 0.847 | 0.825 | 0.880 | 0.852 | 0.533 | 0 |
| qwen-vl-chat | 448×448 | random | 0.890 | 0.953 | 0.820 | 0.882 | 0.430 | 0 |
| qwen-vl-chat | 448×448 | popular | 0.870 | 0.911 | 0.820 | 0.863 | 0.450 | 0 |
| qwen-vl-chat | 448×448 | adversarial | 0.810 | 0.804 | 0.820 | 0.812 | 0.510 | 0 |
| qwen-vl-chat | 448×448 | **avg** | 0.857 | 0.885 | 0.820 | 0.851 | 0.463 | 0 |
| qwen2-vl-7b-instruct | native (max_pixels=12845056; dynamic, e.g. 345 on pope_random_00000) | random | 0.905 | 0.976 | 0.830 | 0.897 | 0.425 | 0 |
| qwen2-vl-7b-instruct | native | popular | 0.885 | 0.933 | 0.830 | 0.878 | 0.445 | 0 |
| qwen2-vl-7b-instruct | native | adversarial | 0.850 | 0.865 | 0.830 | 0.847 | 0.480 | 0 |
| qwen2-vl-7b-instruct | native | **avg** | 0.880 | 0.922 | 0.830 | 0.874 | 0.450 | 0 |

**Deviations / errors:**
- Prior partial runs (wrong split due to missing `load_pope` filter on `combined.json`) cleared before rerun.
- Fixed `load_pope()` to filter `combined.json` entries by `category`/`task` matching `split=` (required for correct 3-split accumulation).
- No generation errors; all 1800 samples completed.

**Sanity checks:** `n_unparsed=0` per model; `by_category` contains random + popular + adversarial in all three final summaries; 5 raw responses/model spot-checked — parser verdicts consistent with leading-token / word-boundary rules (e.g. `"There is no existence of..."` → `no`).

**Notes / caveats (factual):**
- Accuracy/F1 baseline only — **not suitable for activation extraction** (Qwen-VL-Chat uses CPU offload).
- VTI-comparable: same metric set, free-generation elicitation, official POPE template. Parser is more robust than VTI's bare substring match (first-token priority + word boundaries).
- Cross-model numbers are not a controlled comparison (different vision resolutions); per-model reference points only.

---

### [DATE] — Interpretation: are the baselines sound vs VTI? — TEMPLATE

*Romanus + analysis agent. Fill after the run entry above.*

- VTI Table 1 reference (LLaVA-1.5, avg of 3 splits): acc ≈ 79.8, F1 ≈ 79.4 (vanilla baseline). Our LLaVA-1.5 baseline: acc `[ ]`, F1 `[ ]`. Within `[ ]` points → harness `[validated | suspect]`.
- yes_ratio sanity: POPE is 50/50 positive/negative; a yes_ratio far from 0.5 indicates yes-bias. Observed: `[ ]`.
- Decision / next step: `[ ]`.

---

### 2026-06-18 — VTI textual arm implemented; layer-site rotation collapse characterized

**Type:** factual run record
**Branch:** `activation_steering` **Commit:** `5f1b7cc` (uncommitted working-tree changes applied: `evaluation/interventions/vti/*`, `hooks.py` debug knobs, `src/mediation.py` qwen2_vl dispatch, `src/paths.py` vti/coco helpers, `data_scripts/download_chair.py --with-train2014`, `tests/test_VTI_text_steer.py`)
**Environment:** lambdab2, `CUDA_VISIBLE_DEVICES=0` (NVIDIA RTX A6000), conda env `vlm_hallucination_mitigation`, transformers 4.50.1, single-device cuda:0 (~13.2 GiB allocated for LLaVA-1.5).

**Scope:** Phase 1 textual VTI (decoder-only), per `implementation_plans/vti_phase1_textual.md`. Three steering geometries (`additive`, `uniform_rotation`, `gated_rotation`) × two hook sites (`mlp` = steer MLP sub-block output before residual add; `layer` = steer full decoder-layer output after residual). 6 registry interventions `vti_textual_{variant}_{site}`. Directions: PCA rank-1 on last-token `clean − hallucinated` states from 70 paired COCO demos (`data/vti/demos.jsonl`, images `data/coco/train2014/`), cached at `experiment_artifacts/vti/{model_short}/`. Defaults `alpha_text=0.9, eps_coeff=0.1, num_demos=70, rank=1, seed=42`.

**Bugs found and fixed during bring-up:**
- `get_intervention` passed `model_id=` as keyword but the `no_intervention` factory took positional `_model_id` → `TypeError`. Fixed factory signatures in `evaluation/interventions/__init__.py`.
- `get_hiddenstates` concatenated 1-D last-token vectors with `torch.cat(dim=0)`, flattening all layers into one vector → `IndexError: Dimension out of range ... got 1` in `obtain_textual_vti`. Fixed to `torch.stack` → `(num_layers+1, hidden_dim)`. Added `tests/test_VTI_text_steer.py::test_demo_activation_stack_shape`.

**Smoke test (LLaVA-1.5, POPE random, `--limit 10`, all 6 variants):**
```bash
python evaluation/run_eval.py --model llava-hf/llava-1.5-7b-hf \
  --benchmarks pope --pope_split random --limit 10 \
  --interventions vti_textual_additive_mlp vti_textual_additive_layer \
    vti_textual_uniform_rotation_mlp vti_textual_uniform_rotation_layer \
    vti_textual_gated_rotation_mlp vti_textual_gated_rotation_layer \
  --output_dir evaluation/results
```
Observed (POPE acc/f1 on 10 samples): `additive_mlp`, `additive_layer`, `uniform_rotation_mlp`, `gated_rotation_mlp` → 100/100 with non-empty responses; `uniform_rotation_layer`, `gated_rotation_layer` → 0/0 with **all responses empty** (model emits EOS as first generated token). Output under `evaluation/results/llava-1.5-7b-hf/pope/vti_textual_*/`.

**Layer-site collapse diagnosis (single POPE sample `pope_random_00000`, "Is there a snowboard in the image?"):**
- Per-token residual norms at the layer output: normal range (~18–78), a few high-norm interior positions (380, 719, 311, 589); position 0 modest (~8.3). No NaN/Inf.
- `cosine(t, steered) ≈ 0.996` uniformly across all tokens and all 32 layers (matches the designed ~5° rotation for `eps_coeff=0.1, alpha=0.9`); steering is applied correctly.
- Decoded `generated` confirmed empty output = single EOS (id 2) after the prompt; the ~576 `32000` ids are LLaVA `<image>` placeholder tokens, not EOS.

**Strength/mitigation sweep** (`evaluation/interventions/vti/_debug_layer_rotation.py`, `uniform_rotation`, `layer`, 1 sample):
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/interventions/vti/_debug_layer_rotation.py
```
| Config | Response |
|--------|----------|
| `alpha=0.9` (full) | `''` (empty / collapse) |
| `alpha=0.3, 0.1, 0.05, 0.01, 0.0` | `"Yes, there is a snowboard in the image."` (identical to baseline) |
| `alpha=0.9`, decode-only (no prefill steering) | identical to baseline |
| `alpha=0.9`, position 0 unsteered during prefill | identical to baseline |

On this sample: layer-site `uniform_rotation` collapses to empty only at `alpha=0.9`; at `alpha ≤ 0.3` output is byte-identical to the no-hook baseline (no observable effect). Skipping prefill steering, or leaving position 0 (BOS) unsteered during prefill, prevents the collapse at `alpha=0.9` (both reproduce baseline). The `mlp` site does not collapse at `alpha=0.9`.

**New tooling (added, not yet run at scale):** `_debug_layer_rotation.py` extended to sweep alpha over N POPE samples and classify each steered response vs the no-hook baseline as empty / identical / changed, plus decode-only and skip-position-0 probes at the max alpha; writes `experiment_artifacts/vti_layer_debug/{model_short}/sweep_{variant}_layer_{split}_n{N}.json`. Default knobs `steer_prefill=True, skip_first_token=False` in `vti_hook_ctx` reproduce production behavior; they are debug-only.

**Deviations / open items:**
- No `no_intervention` baseline yet for `Qwen/Qwen2.5-VL-7B-Instruct` (planned in Phase 1 §7).
- Multi-sample alpha sweep not yet run; single-sample result cannot distinguish "layer site inert below collapse" from "did not flip this easy sample."

---

### 2026-06-18 (overnight) — VTI textual-arm POPE reproduction (4 models) + layer-rotation strength sweep

**Type:** factual run record
**Branch:** `activation_steering` (uncommitted working-tree changes; same VTI module set as the 2026-06-18 entry above, plus: per-split + dated results layout in `evaluation/runners/eval_runner.py`, `--run_date` in `run_eval.py`, `print_comparison_table` per-split POPE columns).
**Environment:** lambdab2, `CUDA_VISIBLE_DEVICES=0` (NVIDIA RTX A6000 48 GB), conda env `vlm_hallucination_mitigation`, transformers 4.50.1, single-device cuda:0. Two jobs run concurrently in separate `nohup` shells (repro + sweep), each one model in memory at a time.
**Coefficient:** all runs used the textual coefficient `beta = 0.9` (the code default; this knob was named `alpha_text` at run time, renamed `beta` immediately after — see IMPLEMENTATION.md). Vision arm not implemented. `eps_coeff=0.1, num_demos=70, rank=1, seed=42`.

**Run A — reproduction.** `evaluation/run_scripts/run_vti_pope_repro.sh` (`RUN_DATE=2026-06-18`, `LIMIT=200`). Models: LLaVA-1.5-7B, Qwen-VL-Chat, Qwen2-VL-7B-Instruct, Qwen2.5-VL-7B-Instruct. Interventions: `no_intervention`, `vti_textual_additive_mlp`, `vti_textual_additive_layer`, `vti_textual_uniform_rotation_mlp`. POPE splits random/popular/adversarial (pinned 200/split). Logs: `evaluation/results/_logs/repro_20260618_234450.log`. Output: `evaluation/results/2026-06-18/{model_short}/pope_{split}/{iv}/{responses.json,metric_summary.json}`.

POPE accuracy/F1 (×100):

| Model | iv | rnd | pop | adv |
|-------|----|-----|-----|-----|
| llava-1.5-7b-hf | no_intervention | 89.0/88.9 | 86.5/86.7 | 78.5/80.4 |
| llava-1.5-7b-hf | additive_mlp | 91.0/90.5 | 89.0/88.7 | 84.0/84.3 |
| llava-1.5-7b-hf | additive_layer | 91.5/91.1 | 89.5/89.2 | 84.5/84.9 |
| llava-1.5-7b-hf | uniform_rotation_mlp | 88.5/88.4 | 87.0/87.1 | 79.0/80.7 |
| qwen-vl-chat | no_intervention | 89.0/88.2 | 87.0/86.3 | 81.0/81.2 |
| qwen-vl-chat | additive_mlp | 88.5/87.6 | 87.0/86.2 | 81.0/81.0 |
| qwen-vl-chat | additive_layer | 88.5/87.6 | 87.0/86.2 | 81.0/81.0 |
| qwen-vl-chat | uniform_rotation_mlp | 89.0/88.7 | 88.0/87.8 | 79.5/80.8 |
| qwen2-vl-7b-instruct | no_intervention | 90.5/89.7 | 88.5/87.8 | 85.0/84.7 |
| qwen2-vl-7b-instruct | additive_mlp | 90.0/89.2 | 89.5/88.8 | 85.0/84.7 |
| qwen2-vl-7b-instruct | additive_layer | 90.0/89.2 | 89.5/88.8 | 85.0/84.7 |
| qwen2-vl-7b-instruct | uniform_rotation_mlp | 90.0/89.4 | 89.0/89.4 | 84.0/84.4 |
| qwen2.5-vl-7b-instruct | no_intervention | — | 86.0/84.1 | 86.0/84.1 |
| qwen2.5-vl-7b-instruct | additive_mlp | — | 86.5/84.9 | 86.5/84.9 |
| qwen2.5-vl-7b-instruct | additive_layer | — | 86.5/84.9 | 86.5/84.9 |
| qwen2.5-vl-7b-instruct | uniform_rotation_mlp | — | 88.0/87.1 | 83.0/82.7 |

Vision-token counts unchanged from 2026-06-16 entry (Policy A). All `n_unparsed=0`.

**Run B — layer-rotation strength sweep.** `evaluation/run_scripts/run_vti_layer_sweep.sh` (n=200, split=random, betas `0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1`, variants `uniform_rotation` + `gated_rotation`, hook_site=layer, 4 models). Log: `evaluation/results/_logs/sweep_20260618_234525.log`. Output (relocated post-run): `evaluation/vti_rotation_strength/results/2026-06-18/{model_short}/sweep_{variant}_layer_random_n200.json` (was `experiment_artifacts/vti_layer_debug/`; the script + outputs were moved to `evaluation/vti_rotation_strength/` and the strength knob renamed `alpha`→`beta` after this run, so these JSONs carry the older `metrics_by_alpha` schema).

`uniform_rotation @ layer`, selected betas — acc / yes_ratio / mean_len / flips(c→w / w→c), baseline = no hooks:

| Model | baseline acc / yes_r / mlen | β=0.6 | β=0.5 | β=0.3 | β=0.2 | β=0.1 |
|-------|------|------|------|------|------|------|
| llava-1.5-7b-hf | 0.89 / 0.49 / 87.2 | 0.55 / 0.05 / 3.6 (78/10) | 0.795 / 0.295 / 29.9 (30/11) | 0.885 / 0.415 / 45.1 (9/8) | 0.905 / 0.455 / 69.0 (3/6) | 0.895 / 0.465 / 79.9 (3/4) |
| qwen-vl-chat | 0.89 / 0.43 / 64.1 | 0.69 / 0.58 / 8.1, 11 empty (21/9) | 0.87 / 0.53 / 20.6 (11/9) | 0.89 / 0.48 / 41.1 (5/5) | 0.88 / 0.46 / 51.0 (4/2) | 0.885 / 0.455 / 59.1 (3/2) |
| qwen2-vl-7b-instruct | 0.905 / 0.425 / 49.9 | 0.88 / 0.395 / 143.2 (4/2) | 0.885 / 0.41 / 112.5 (2/2) | 0.895 / 0.425 / 86.2 (1/0) | 0.895 / 0.425 / 68.3 (2/0) | 0.90 / 0.43 / 56.6 (1/0) |
| qwen2.5-vl-7b-instruct | 0.87 / 0.37 / 154.3 | 0.915 / 0.545 / 250.1 (13/22) | 0.925 / 0.465 / 211.6 (4/15) | 0.90 / 0.41 / 163.8 (1/7) | 0.885 / 0.395 / 154.7 (1/4) | 0.875 / 0.385 / 155.5 (1/2) |

Mitigation probes at β=0.6 (all models): `decode_only` (steer decode steps only, no prefill) → acc/yes_ratio = baseline, flips=0 for every model. `skip_pos0` (leave seq position 0 unsteered during prefill) → no empties; LLaVA acc 0.55→0.87, mean_len 3.6→36.3 (collapse averted); Qwen2.5 acc 0.87→0.915 (22 w→c).

**Factual observations (for the analysis agent; not interpretation):**
- `gated_rotation` rows are numerically identical to `uniform_rotation` for every model and every beta, differing only at the ≤1-char level in mean_len (e.g. LLaVA β=0.35 mlen 38.2 vs 37.8). The two variants' acc/yes_ratio/flip counts match exactly.
- Among the four models, `qwen2.5-vl-7b-instruct` is the only one with `w→c > c→w` at every beta (net wrong→correct), and the only one whose accuracy exceeds its no-hook baseline under rotation (peak 0.925 at β=0.5 vs 0.87 baseline). Its baseline `yes_ratio` is 0.37 (others 0.43–0.49) and rotation raises it toward 0.5; its baseline mean_len is 154 (others 50–87). For the other three models rotation lowers `yes_ratio` and `c→w ≥ w→c` at the betas where flips occur.

**Deviations / errors:**
- Qwen2.5-VL POPE **random** failed at model load: `OSError: ... does not appear to have a file named model-00005-of-00005.safetensors` after a CDN read timeout on the xet download (`us.aws.cdn.hf.co` Read timed out → "Trying to resume download…"). The script's per-split loop continued; **popular** and **adversarial** then loaded from the now-complete cache and produced the numbers above. Qwen2.5 `random` is pending a rerun (shard now cached).
- First repro launch (`repro_20260618_233008.log`) was killed: it skipped LLaVA because stale pre-schema result JSONs existed under the old date-less path. Fixed by adding the per-split + dated results layout and archiving the stale trees to `evaluation/results/_archive_preschema_20260618_234142/`; relaunched as `repro_20260618_234450.log`.
- Eval-runner path bug fixed before this run: POPE splits previously shared one `pope/{iv}/` dir, so with `--skip_if_exists` only `random` ran. Now `pope_{split}` per split.

**Open items:**
- Rerun Qwen2.5-VL POPE random (only missing cell).
- `beta` was fixed at 0.9 (code default); paper grid-searches α,β over {0.1…1.0} and reports VTI at α=0.2, β=0.4 — a beta grid + larger N (paper uses 3000/split) are not yet run.

### 2026-06-19 — VTI textual-arm POPE reproduction (full β grid 0.1–1.0, 4 models) + rotation-strength sweep (β grid, precision/recall + flip decomposition)

**Type:** factual run record
**Branch:** `activation_steering` **Commit:** `5f1b7cc` (uncommitted working-tree changes vs the 2026-06-18 entries: `--beta` arg in `evaluation/run_eval.py` threaded through `run_evaluation`; `eval_runner.py` appends `__b{beta}` to intervention output dirs when `beta` is set and present in the intervention config; rotation-strength experiment relocated to `evaluation/vti_rotation_strength/rotation_strength.py` with precision/recall + confusion-matrix flip decomposition; new driver scripts `run_vti_pope_beta_grid.sh`, `run_rotation_strength_sweep.sh`, `run_beta_grid_local.sh`).
**Environment:** lambdab2, `CUDA_VISIBLE_DEVICES=0` (NVIDIA RTX A6000, 47.5 GB total, ~13.2 GB allocated for LLaVA-1.5), conda env `vlm_hallucination_mitigation`, transformers 4.50.1, float16, single-device cuda:0. Two `nohup` jobs run concurrently (Run A repro + Run B sweep), one model in memory at a time.
**Params (both runs):** `eps_coeff=0.1, num_demos=70, rank=1, seed=42`; directions cached at `experiment_artifacts/vti/{model_short}/`. Vision arm not implemented. POPE 200 samples/split (pinned eval ids; guard verified `ok: subset matches pinned manifest`).

**Run A — reproduction, full β grid.** `evaluation/run_scripts/run_vti_pope_beta_grid.sh` (β-outermost order `0.4 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9 1.0`):
```bash
# models: llava-hf/llava-1.5-7b-hf, Qwen/Qwen-VL-Chat, Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
# ivs:    vti_textual_{additive_mlp,additive_layer,uniform_rotation_mlp}; splits: random/popular/adversarial
python evaluation/run_eval.py --model $MODEL --benchmarks pope --pope_split $SPLIT \
  --interventions $IV --beta $BETA --limit 200 --run_date 2026-06-19 \
  --output_dir evaluation/results
```
Log: `evaluation/results/_logs/beta_grid_20260619_110137.log`. Output: `evaluation/results/2026-06-19/{model_short}/pope_{split}/{iv}__b{beta}/{responses.json,metric_summary.json}` (baselines under `.../no_intervention/`). All cells present (Qwen2.5 `random`, which failed at load on 2026-06-18, succeeded this run from cached shards). All `n_unparsed=0`.

POPE accuracy/F1 (×100), **averaged over the 3 splits**, per model (per-split breakdown in `evaluation/results/2026-06-19/meeting_summary_2026-06-19.md` and on disk):

_llava-1.5-7b-hf_ (split-averaged acc/F1)

| β | additive_mlp | additive_layer | uniform_rotation_mlp |
|---|---|---|---|
| baseline | 84.7/85.3 | 84.7/85.3 | 84.7/85.3 |
| 0.1 | 83.7/84.2 | 83.7/84.2 | 84.5/85.2 |
| 0.2 | 83.7/84.2 | 84.2/84.8 | 84.5/85.2 |
| 0.3 | 85.3/85.9 | 85.3/85.9 | 84.5/85.2 |
| 0.4 | 85.3/85.9 | 85.3/85.9 | 84.2/84.9 |
| 0.5 | 85.5/85.9 | 85.5/85.9 | 84.0/84.8 |
| 0.6 | 86.2/86.4 | 86.2/86.4 | 84.0/84.8 |
| 0.7 | 86.7/86.8 | 86.7/86.8 | 84.3/85.0 |
| 0.8 | 87.8/87.9 | 87.8/87.9 | 84.8/85.4 |
| 0.9 | 88.0/87.8 | 88.5/88.4 | 84.8/85.4 |
| 1.0 | 88.3/88.0 | 88.3/88.0 | 85.3/85.9 |

_qwen-vl-chat_ (split-averaged acc/F1)

| β | additive_mlp | additive_layer | uniform_rotation_mlp |
|---|---|---|---|
| baseline | 85.7/85.2 | 85.7/85.2 | 85.7/85.2 |
| 0.1 | 85.5/85.1 | 85.5/85.1 | 85.2/84.8 |
| 0.2 | 85.5/85.1 | 85.5/85.1 | 85.5/85.2 |
| 0.3 | 85.5/85.1 | 85.3/84.9 | 86.2/86.0 |
| 0.4 | 85.7/85.2 | 85.8/85.4 | 85.5/85.4 |
| 0.5 | 85.3/84.9 | 85.7/85.2 | 85.5/85.4 |
| 0.6 | 85.8/85.4 | 85.7/85.2 | 85.8/85.8 |
| 0.7 | 86.2/85.6 | 86.0/85.5 | 86.2/86.3 |
| 0.8 | 85.5/84.9 | 86.3/85.8 | 86.2/86.3 |
| 0.9 | 85.5/84.9 | 85.5/84.9 | 85.5/85.7 |
| 1.0 | 85.5/84.9 | 85.5/84.9 | 86.2/86.6 |

_qwen2-vl-7b-instruct_ (split-averaged acc/F1)

| β | additive_mlp | additive_layer | uniform_rotation_mlp |
|---|---|---|---|
| baseline | 88.0/87.4 | 88.0/87.4 | 88.0/87.4 |
| 0.1 | 88.2/87.6 | 88.2/87.6 | 87.7/87.1 |
| 0.2 | 88.3/87.7 | 88.3/87.7 | 87.7/87.1 |
| 0.3 | 88.0/87.4 | 88.0/87.4 | 88.3/87.7 |
| 0.4 | 88.0/87.4 | 88.0/87.4 | 87.7/87.1 |
| 0.5 | 88.0/87.4 | 88.3/87.7 | 88.0/87.4 |
| 0.6 | 88.3/87.7 | 88.2/87.6 | 87.7/87.4 |
| 0.7 | 88.2/87.6 | 87.8/87.3 | 87.3/87.3 |
| 0.8 | 87.8/87.6 | 88.2/87.6 | 87.2/87.1 |
| 0.9 | 88.2/87.6 | 88.2/87.6 | 87.7/87.7 |
| 1.0 | 87.3/87.0 | 87.3/87.0 | 87.3/87.9 |

_qwen2.5-vl-7b-instruct_ (split-averaged acc/F1)

| β | additive_mlp | additive_layer | uniform_rotation_mlp |
|---|---|---|---|
| baseline | 86.3/84.4 | 86.3/84.4 | 86.3/84.4 |
| 0.1 | 86.3/84.4 | 86.0/84.1 | 86.5/84.8 |
| 0.2 | 86.0/84.1 | 86.5/84.8 | 86.8/85.2 |
| 0.3 | 86.5/84.8 | 86.5/84.8 | 86.8/85.2 |
| 0.4 | 86.5/84.8 | 86.5/84.8 | 87.2/85.7 |
| 0.5 | 86.5/84.8 | 86.5/84.8 | 87.7/86.4 |
| 0.6 | 86.3/84.6 | 86.3/84.6 | 87.3/86.2 |
| 0.7 | 86.8/85.2 | 86.8/85.2 | 88.2/87.3 |
| 0.8 | 86.8/85.2 | 86.8/85.2 | 87.7/86.8 |
| 0.9 | 86.8/85.2 | 86.8/85.2 | 86.8/86.1 |
| 1.0 | 86.8/85.2 | 86.8/85.2 | 86.7/86.1 |

**Run B — rotation-strength sweep.** `evaluation/vti_rotation_strength/run_scripts/run_rotation_strength_sweep.sh`:
```bash
# variant uniform_rotation @ hook_site=layer (residual stream), split=random, n=200
# betas: 0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1 ; gated_rotation dropped (identical to uniform per 2026-06-18)
python evaluation/vti_rotation_strength/rotation_strength.py --model $MODEL \
  --variant uniform_rotation --betas 0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1 \
  --n 200 --split random --run_date 2026-06-19
```
Log: `evaluation/results/_logs/rotstrength_20260619_110207.log`. Output: `evaluation/vti_rotation_strength/results/2026-06-19/{model_short}/sweep_uniform_rotation_layer_random_n200.json`. New schema per β: `accuracy, precision, recall, f1, yes_ratio, mean_len, empty` + flip decomposition `n_decision_flips, n_flip_correct_to_wrong (c→w), n_flip_wrong_to_correct (w→c), flip_tp_to_fn, flip_tn_to_fp (h+ = induced hallucination), flip_fp_to_tn (h- = removed hallucination), flip_fn_to_tp`. Baseline = no hooks.

_llava-1.5-7b-hf_ — `uniform_rotation @ layer`, random, n=200 (baseline acc 89.0, F1 88.9, yes_r 0.490, mlen 87.2)

| β | acc | prec | rec | F1 | yes_r | mlen | empty | flips | c→w | w→c | h+ | h- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 89.5 | 92.5 | 86.0 | 89.1 | 0.465 | 79.9 | 0 | 7 | 3 | 4 | 0 | 3 |
| 0.20 | 90.5 | 94.5 | 86.0 | 90.0 | 0.455 | 69.0 | 0 | 9 | 3 | 6 | 0 | 5 |
| 0.25 | 89.5 | 95.4 | 83.0 | 88.8 | 0.435 | 56.8 | 0 | 13 | 6 | 7 | 0 | 6 |
| 0.30 | 88.5 | 96.4 | 80.0 | 87.4 | 0.415 | 45.1 | 0 | 17 | 9 | 8 | 0 | 7 |
| 0.35 | 87.0 | 96.2 | 77.0 | 85.6 | 0.400 | 38.2 | 0 | 20 | 12 | 8 | 0 | 7 |
| 0.40 | 86.5 | 98.7 | 74.0 | 84.6 | 0.375 | 36.0 | 0 | 25 | 15 | 10 | 0 | 9 |
| 0.45 | 82.0 | 98.5 | 65.0 | 78.3 | 0.330 | 35.0 | 0 | 34 | 24 | 10 | 0 | 9 |
| 0.50 | 79.5 | 100.0 | 59.0 | 74.2 | 0.295 | 29.9 | 0 | 41 | 30 | 11 | 0 | 10 |
| 0.60 | 55.0 | 100.0 | 10.0 | 18.2 | 0.050 | 3.6 | 0 | 88 | 78 | 10 | 0 | 10 |

_qwen-vl-chat_ — `uniform_rotation @ layer`, random, n=200 (baseline acc 89.0, F1 88.2, yes_r 0.430, mlen 64.1)

| β | acc | prec | rec | F1 | yes_r | mlen | empty | flips | c→w | w→c | h+ | h- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 88.5 | 92.3 | 84.0 | 88.0 | 0.455 | 59.1 | 0 | 5 | 3 | 2 | 3 | 0 |
| 0.20 | 88.0 | 91.3 | 84.0 | 87.5 | 0.460 | 51.0 | 0 | 6 | 4 | 2 | 4 | 0 |
| 0.25 | 88.0 | 90.4 | 85.0 | 87.6 | 0.470 | 45.0 | 0 | 8 | 5 | 3 | 5 | 0 |
| 0.30 | 89.0 | 90.6 | 87.0 | 88.8 | 0.480 | 41.1 | 0 | 10 | 5 | 5 | 5 | 0 |
| 0.35 | 90.0 | 91.7 | 88.0 | 89.8 | 0.480 | 38.4 | 0 | 10 | 4 | 6 | 4 | 0 |
| 0.40 | 89.5 | 89.9 | 89.0 | 89.5 | 0.495 | 34.7 | 0 | 13 | 6 | 7 | 6 | 0 |
| 0.45 | 88.0 | 85.9 | 91.0 | 88.3 | 0.530 | 31.3 | 0 | 20 | 11 | 9 | 11 | 0 |
| 0.50 | 87.0 | 85.9 | 91.0 | 88.3 | 0.530 | 20.6 | 0 | 20 | 11 | 9 | 11 | 0 |
| 0.60 | 69.0 | 78.5 | 91.0 | 84.3 | 0.580 | 8.1 | 11 | 30 | 21 | 9 | 21 | 0 |

_qwen2-vl-7b-instruct_ — `uniform_rotation @ layer`, random, n=200 (baseline acc 90.5, F1 89.7, yes_r 0.425, mlen 49.9)

| β | acc | prec | rec | F1 | yes_r | mlen | empty | flips | c→w | w→c | h+ | h- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 90.0 | 96.5 | 83.0 | 89.2 | 0.430 | 56.6 | 0 | 1 | 1 | 0 | 1 | 0 |
| 0.20 | 89.5 | 96.5 | 82.0 | 88.6 | 0.425 | 68.3 | 0 | 2 | 2 | 0 | 1 | 0 |
| 0.25 | 89.5 | 96.5 | 82.0 | 88.6 | 0.425 | 76.0 | 0 | 2 | 2 | 0 | 1 | 0 |
| 0.30 | 89.5 | 96.5 | 82.0 | 88.6 | 0.425 | 86.2 | 0 | 1 | 1 | 0 | 1 | 0 |
| 0.35 | 88.5 | 96.4 | 81.0 | 88.0 | 0.420 | 91.5 | 0 | 1 | 1 | 0 | 1 | 0 |
| 0.40 | 88.0 | 96.4 | 80.0 | 87.4 | 0.415 | 98.1 | 0 | 2 | 2 | 0 | 1 | 0 |
| 0.45 | 87.5 | 97.5 | 79.0 | 87.3 | 0.405 | 108.7 | 0 | 2 | 2 | 0 | 1 | 0 |
| 0.50 | 88.5 | 97.6 | 80.0 | 87.9 | 0.410 | 112.5 | 0 | 4 | 2 | 2 | 1 | 1 |
| 0.60 | 88.0 | 98.7 | 78.0 | 87.2 | 0.395 | 143.2 | 0 | 6 | 4 | 2 | 0 | 1 |

_qwen2.5-vl-7b-instruct_ — `uniform_rotation @ layer`, random, n=200 (baseline acc 87.0, F1 85.1, yes_r 0.370, mlen 154.3)

| β | acc | prec | rec | F1 | yes_r | mlen | empty | flips | c→w | w→c | h+ | h- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 87.5 | 98.7 | 76.0 | 85.9 | 0.385 | 155.5 | 0 | 3 | 1 | 2 | 1 | 0 |
| 0.20 | 88.5 | 98.7 | 78.0 | 87.2 | 0.395 | 154.7 | 0 | 5 | 1 | 4 | 1 | 0 |
| 0.25 | 89.0 | 98.8 | 79.0 | 87.8 | 0.400 | 159.0 | 0 | 6 | 1 | 5 | 1 | 0 |
| 0.30 | 90.0 | 98.8 | 81.0 | 89.0 | 0.410 | 163.8 | 0 | 8 | 1 | 7 | 1 | 0 |
| 0.35 | 89.5 | 97.6 | 81.0 | 88.5 | 0.415 | 167.6 | 0 | 9 | 2 | 7 | 2 | 0 |
| 0.40 | 90.5 | 96.5 | 84.0 | 89.8 | 0.435 | 178.7 | 0 | 13 | 3 | 10 | 3 | 0 |
| 0.45 | 91.5 | 96.6 | 86.0 | 91.0 | 0.445 | 191.9 | 0 | 15 | 3 | 12 | 3 | 0 |
| 0.50 | 92.5 | 95.7 | 89.0 | 92.2 | 0.465 | 211.6 | 0 | 19 | 4 | 15 | 4 | 0 |
| 0.60 | 91.5 | 88.1 | 96.0 | 91.9 | 0.545 | 250.1 | 0 | 35 | 13 | 22 | 13 | 0 |

**Run B — cross-model β tables (rows = model, cols = β; random split, n=200).** These make the β↔length relationship explicit: mean response length (characters) *decreases* with β for LLaVA-1.5 and Qwen-VL-Chat but *increases* with β for Qwen2-VL and Qwen2.5-VL.

_Accuracy (%)_

| Model | baseline | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 89.0 | 89.5 | 90.5 | 89.5 | 88.5 | 87.0 | 86.5 | 82.0 | 79.5 | 55.0 |
| Qwen-VL-Chat | 89.0 | 88.5 | 88.0 | 88.0 | 89.0 | 90.0 | 89.5 | 88.0 | 87.0 | 69.0 |
| Qwen2-VL-7B | 90.5 | 90.0 | 89.5 | 89.5 | 89.5 | 88.5 | 88.0 | 87.5 | 88.5 | 88.0 |
| Qwen2.5-VL-7B | 87.0 | 87.5 | 88.5 | 89.0 | 90.0 | 89.5 | 90.5 | 91.5 | 92.5 | 91.5 |

_F1 (%)_

| Model | baseline | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 88.9 | 89.1 | 90.0 | 88.8 | 87.4 | 85.6 | 84.6 | 78.3 | 74.2 | 18.2 |
| Qwen-VL-Chat | 88.2 | 88.0 | 87.5 | 87.6 | 88.8 | 89.8 | 89.5 | 88.3 | 88.3 | 84.3 |
| Qwen2-VL-7B | 89.7 | 89.2 | 88.6 | 88.6 | 88.6 | 88.0 | 87.4 | 87.3 | 87.9 | 87.2 |
| Qwen2.5-VL-7B | 85.1 | 85.9 | 87.2 | 87.8 | 89.0 | 88.5 | 89.8 | 91.0 | 92.2 | 91.9 |

_Yes-ratio_

| Model | baseline | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 0.490 | 0.465 | 0.455 | 0.435 | 0.415 | 0.400 | 0.375 | 0.330 | 0.295 | 0.050 |
| Qwen-VL-Chat | 0.430 | 0.455 | 0.460 | 0.470 | 0.480 | 0.480 | 0.495 | 0.530 | 0.530 | 0.580 |
| Qwen2-VL-7B | 0.425 | 0.430 | 0.425 | 0.425 | 0.425 | 0.420 | 0.415 | 0.405 | 0.410 | 0.395 |
| Qwen2.5-VL-7B | 0.370 | 0.385 | 0.395 | 0.400 | 0.410 | 0.415 | 0.435 | 0.445 | 0.465 | 0.545 |

_Mean response length (characters)_

| Model | baseline | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 87.2 | 79.9 | 69.0 | 56.8 | 45.1 | 38.2 | 36.0 | 35.0 | 29.9 | 3.6 |
| Qwen-VL-Chat | 64.1 | 59.1 | 51.0 | 45.0 | 41.1 | 38.4 | 34.7 | 31.3 | 20.6 | 8.1 |
| Qwen2-VL-7B | 49.9 | 56.6 | 68.3 | 76.0 | 86.2 | 91.5 | 98.1 | 108.7 | 112.5 | 143.2 |
| Qwen2.5-VL-7B | 154.3 | 155.5 | 154.7 | 159.0 | 163.8 | 167.6 | 178.7 | 191.9 | 211.6 | 250.1 |

_Decision flips vs baseline_

| Model | baseline | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 0 | 7 | 9 | 13 | 17 | 20 | 25 | 34 | 41 | 88 |
| Qwen-VL-Chat | 0 | 5 | 6 | 8 | 10 | 10 | 13 | 20 | 20 | 30 |
| Qwen2-VL-7B | 0 | 1 | 2 | 2 | 1 | 1 | 2 | 2 | 4 | 6 |
| Qwen2.5-VL-7B | 0 | 3 | 5 | 6 | 8 | 9 | 13 | 15 | 19 | 35 |

**Factual observations (numeric; for the analysis agent, not interpretation):**
- `gated_rotation` was not run this round (dropped after the 2026-06-18 finding that its rows were numerically identical to `uniform_rotation`).
- Reproduction at the split-averaged level: LLaVA-1.5 accuracy increases with β for both additive sites (additive_layer 84.7 baseline → 88.5 at β=0.9), peaking at β=0.9–1.0; LLaVA `uniform_rotation_mlp` stays within ~0.7 pt of baseline at all β. Qwen-VL-Chat and Qwen2-VL split-averaged accuracy stay within ~1 pt of their baselines across the entire β grid for all three interventions. Qwen2.5-VL is likewise within ~0.5 pt of baseline for the additive sites, but its `uniform_rotation_mlp` rises with β to a peak of 88.2 at β=0.7 (+1.9 over its 86.3 baseline) before falling back to ~86.7 at β=1.0.
- Rotation-strength sweep: `qwen2.5-vl-7b-instruct` is the only model with `w→c > c→w` at every β; its accuracy exceeds its no-hook baseline (0.87) at every β, peaking 0.925 at β=0.5, with recall rising 0.76→0.96 (β=0.1→0.6) and mean_len 154→250. LLaVA-1.5 accuracy decreases monotonically (0.89→0.55 at β=0.6) with yes_ratio 0.49→0.05, recall 0.88→0.10, flips c→w-dominated, `h-`-only (no induced). Qwen-VL-Chat yes_ratio rises 0.43→0.58, flips are `h+`-only (induced; `h-`=0), 11 empty responses at β=0.6. Qwen2-VL has ≤6 decision flips across the whole grid with mean_len rising 50→143.
- Mean response length (characters) is monotone in β with opposite sign by model: LLaVA-1.5 87.2→3.6 and Qwen-VL-Chat 64.1→8.1 (length *falls* as β rises); Qwen2-VL 49.9→143.2 and Qwen2.5-VL 154.3→250.1 (length *grows* as β rises). See the "Mean response length" table above.
- Per-sample answer + length trajectories (β ascending; pred/ok/len/response per β) illustrating these shifts and the associated yes/no flips, for LLaVA-1.5, Qwen2-VL, Qwen2.5-VL: `evaluation/results/2026-06-19/rotation_strength_examples_2026-06-19.md` (e.g. LLaVA `pope_random_00171` baseline yes(✗)→no(✓) with length 129→2; Qwen2.5 `pope_random_00166` baseline no(✗)→yes(✓) at β=0.45 with length 169→236).
- Full per-split tables and by-model / by-β / by-intervention cross-tabulations: `evaluation/results/2026-06-19/meeting_summary_2026-06-19.md`.

**Deviations / errors:**
- Two accidental concurrent launches of the beta-grid (and one launch started before unsaved code edits were written) were killed; the final clean runs are `beta_grid_20260619_110137.log` and `rotstrength_20260619_110207.log`. No partial artifacts from the killed launches were written under `2026-06-19/`.
- Qwen2.5-VL POPE `random` (failed at model load on 2026-06-18 due to a CDN timeout on shard 5/5) completed this run from the now-cached shards; the 2026-06-18 open item for this cell is closed.

---

### 2026-06-22 — Response-length analysis of the 2026-06-19 reproduction grid (additive + uniform_rotation_mlp)

**Type:** factual analysis of existing run outputs (no new run).
Derived from `evaluation/results/2026-06-19/{model}/pope_{split}/{iv}__b{beta}/responses.json` as mean `len(response)` in characters (same unit as the Run B sweep `mean_len`), split-averaged over random/popular/adversarial, n=200/split. Motivation: compare whether the grid interventions move output length the way residual-site `uniform_rotation` does (Run B tables). Full per-split tables: `answers/concepts/jun_22_2026/grid_response_length_vs_beta.md`.

Mean response length (characters), rows = model, cols = β, split-averaged. `baseline` = no_intervention.

_additive_mlp_

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.4 | 79.9 | 74.7 | 71.0 | 65.0 | 58.8 | 51.0 | 40.8 | 32.8 | 24.1 | 12.4 |
| Qwen-VL-Chat | 64.2 | 64.9 | 65.5 | 65.9 | 65.5 | 65.3 | 64.5 | 64.4 | 63.5 | 63.1 | 62.7 |
| Qwen2-VL-7B | 47.5 | 48.4 | 49.4 | 49.3 | 49.8 | 51.1 | 52.4 | 52.6 | 54.2 | 55.3 | 55.3 |
| Qwen2.5-VL-7B | 154.2 | 155.1 | 155.6 | 154.2 | 157.2 | 155.5 | 158.9 | 157.3 | 159.0 | 159.4 | 158.9 |

_additive_layer_

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.4 | 79.9 | 74.2 | 71.1 | 65.2 | 58.8 | 51.1 | 40.9 | 32.7 | 24.1 | 12.3 |
| Qwen-VL-Chat | 64.2 | 65.7 | 65.2 | 65.9 | 65.8 | 65.3 | 65.6 | 64.5 | 64.1 | 63.4 | 63.3 |
| Qwen2-VL-7B | 47.5 | 48.0 | 48.8 | 49.5 | 51.0 | 50.4 | 51.1 | 52.8 | 53.8 | 54.0 | 56.5 |
| Qwen2.5-VL-7B | 154.2 | 153.7 | 155.7 | 155.7 | 156.4 | 156.5 | 157.1 | 158.9 | 159.4 | 159.9 | 158.6 |

_uniform_rotation_mlp_

| Model | baseline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 84.4 | 79.7 | 75.5 | 71.6 | 66.8 | 58.3 | 50.4 | 44.2 | 39.8 | 37.9 | 35.9 |
| Qwen-VL-Chat | 64.2 | 63.6 | 62.2 | 59.9 | 55.6 | 53.8 | 51.7 | 49.0 | 47.5 | 46.3 | 45.8 |
| Qwen2-VL-7B | 47.5 | 49.7 | 51.6 | 54.7 | 59.9 | 63.7 | 69.5 | 75.7 | 81.0 | 84.4 | 90.6 |
| Qwen2.5-VL-7B | 154.2 | 156.0 | 158.8 | 157.2 | 160.3 | 159.4 | 160.9 | 163.2 | 163.7 | 167.7 | 185.5 |

**Factual observations (numeric):**
- `additive_mlp` and `additive_layer` produce near-identical mean lengths at every β (per-cell differences ≤ ~1.5 chars).
- Under the additive sites, length changes are concentrated in LLaVA-1.5 (84.4 → ~12 chars at β=1.0, monotone decrease). Qwen-VL-Chat is ~flat (64.2 → 62.7), Qwen2-VL rises mildly (47.5 → 55.3), Qwen2.5-VL is ~flat (154.2 → 158.9) across β=0.1→1.0.
- `uniform_rotation_mlp` produces a larger, model-specific length effect with opposite sign by model: LLaVA-1.5 84.4 → 35.9 and Qwen-VL-Chat 64.2 → 45.8 (decrease with β); Qwen2-VL 47.5 → 90.6 and Qwen2.5-VL 154.2 → 185.5 (increase with β).
- The per-model sign of the `uniform_rotation_mlp` length effect matches the residual-site `uniform_rotation` sweep (Run B): negative for LLaVA-1.5 and Qwen-VL-Chat, positive for Qwen2-VL and Qwen2.5-VL. The additive sites do not reproduce this pattern (Qwen family ~flat).

---

### 2026-06-22 — Implement MMHal-Bench (judge) + CHAIR scorers (plan: `implementation_plans/plan_mmhal_chair_scoring.md`)

**Type:** code implementation + scorer-level validation (no benchmark sweep yet; no W&B).
Branch/commit: uncommitted working tree at time of writing.

**Step 0 verification findings (resolved by reading the repo):**
- 0a MMHal sample shape (`src/dataset.py::load_mmhal_bench`): question = `text`; gt answer = `label`; the 8 MMHal types are in `task` (= `question_type`: attribute/adversarial/comparison/counting/relation/environment/holistic/other, 12 each, confirmed by count); `category` is the question *topic* (outdoor/indoor/…, 8 each). The human image-content description IS surfaced as `raw.image_content` (list). No loader change needed; no judge-input deviation.
- 0b CHAIR `image_id`: present as `raw.coco_id` (int). No loader change needed.
- 0c `data/coco/annotations/instances_val2014.json` present. No synonym map was vendored → added canonical Rohrbach et al. `synonyms.txt` (80 COCO classes) at `evaluation/classifiers/chair_synonyms.txt`.
- 0d Dispatch: `compute_metric_records(records, benchmark)` → `_BENCHMARK_SCORERS[benchmark]`; eval loaders (`evaluation/benchmarks/__init__.py`) pass `raw` through into `EvalSample.metadata["raw"]`, so scorers can read `coco_id` / `image_content` / `question_type`.
- B-DEC-1 (caption routing): already correct. The CHAIR eval loader passes `text` (the caption prompt) as `question` into `intervention.generate(...)`, which calls `wrapper.generate_vl(image, question)` — VTI steering hooks are therefore active for CHAIR. No change needed.

**Code changes:**
- NEW `evaluation/classifiers/judges.py`: `Judge` protocol; `get_judge(spec)` with providers `mock` (default, offline), `openai[:model]` (default `gpt-4-0314`), `anthropic[:model]`, `gemini[:model]`; verbatim official MMHal judge prompt (`MMHAL_JUDGE_TEMPLATE`, 5 examples); `parse_mmhal_rating` (`rating: {0..6}`, exactly-one-match else `None`); hallucination cutoff `rating < 3` (`HALLUCINATION_RATING_THRESHOLD = 3`). Judge is text-only (consumes `image_content`, not the image).
- NEW `evaluation/classifiers/chair_objects.py`: synonym-map loader (expanded surface forms, no nltk/pattern deps), COCO gt loader (`instances_val2014.json`, module-cached), `parse_caption_objects` (longest-n-gram match).
- `evaluation/classifiers/metrics.py`: `score_mmhal_records(records, *, judge)` → `metric="mmhal_judge"` with `avg_score`/`hallucination_rate`/`n_unparsed_judge`/`by_category` (keyed by question type incl. `counting`); `score_chair_records` → `metric="chair"` with `chair_i`/`chair_s`/`avg_objects_mentioned`/`avg_caption_len_chars`. `compute_metric_records(..., *, judge=None)` threads the judge to MMHal only (falls back to mock).
- `evaluation/runners/eval_runner.py`: resolve judge iff `mmhal_bench` in benchmarks; per-benchmark `max_new_tokens` (CHAIR uses `chair_max_new_tokens`, default 64; others use `--max_new_tokens`); pass judge to scorer; `print_comparison_table` renders CHAIR `s/i %` (lower better) and MMHal `avg_score` (higher better) with a legend.
- `evaluation/run_eval.py`: added `--judge` (default `mock`) and `--chair_max_new_tokens` (default 64).
- `src/paths.py`: added `coco_annotations_dir()`.

**Decisions taken (defaults; overridable):** judge default `mock`; CHAIR `max_new_tokens` frozen at 64; CHAIR synonyms = canonical Rohrbach release; parse failures surfaced as `n_unparsed_judge` (not scored 0, unlike official script).

**Validation (scoring path only; no baselines):**
- Scorer unit checks (no model): MMHal parser handles single/zero/ambiguous/case-insensitive ratings; hallucination cutoff `r2→1, r3→0, r6→0`; CHAIR synonym map = 80 classes; CHAIR parser fires on real captions (e.g. "person riding a bicycle next to a dog and a giraffe" → mentioned {person,bicycle,dog,giraffe}; image 391895 gt {bicycle,motorcycle,person} → hallucinated {dog,giraffe}); "unicorn" correctly unmatched; "dining table" matched as bigram.
- End-to-end smoke (real generation), `llava-hf/llava-1.5-7b-hf`, `CUDA_VISIBLE_DEVICES=0`, `--benchmarks mmhal_bench chair --judge mock --limit 4 --interventions no_intervention --chair_max_new_tokens 64`: completed, both summaries written with the documented schema; MMHal `avg_score=4.00` (mock), `n_unparsed_judge=0`, `by_category` populated for the 4 sampled types; CHAIR `chair_i=0.286 chair_s=0.75 avg_objects_mentioned=3.5 avg_caption_len_chars=267.25 n_missing_image_id=0`. **These are limit-4 mock-judge smoke numbers, not baselines; the smoke output dir was deleted.** Real baselines (plan Validation step 4: `no_intervention`, all four models, Policy A resolution, fixed CHAIR `max_new_tokens`, named judge for MMHal) not yet run.

**Deviations from plan:** none material. The single non-default scorer choice (parse failure → unparsed rather than 0) is per the plan's A2 instruction and differs from the official MMHal script; documented in code + `IMPLEMENTATION.md`.

---

### 2026-06-22 — Implement AMBER discriminative scoring (plan: `implementation_plans/plan_amber_discriminative_scoring.md`)

**Type:** code implementation + scorer-level + small generation validation (no benchmark sweep yet; no W&B).
Branch/commit: uncommitted working tree at time of writing.

**Step 0 verification findings (resolved by reading the repo):**
- 0a gold location/encoding: the plan assumed "a gold→sample join exists somewhere." It does NOT — discriminative gold is **not surfaced** by `load_amber`: `combined.json` (built from the AMBER query files) carries only `id`/`image`/`query`, and `label` is empty (`""`) for all 14216 discriminative items, `category` = the literal string `"discriminative"`. Consequence: the prior `score_amber_records` did `if not gt: continue`, so it **skipped every discriminative record** — any earlier AMBER discriminative `task_accuracy` was vacuous (no such results were on disk). Gold lives in `data/amber/data/annotations.json` as `{id, type, truth:"yes"/"no"}`, joinable by question id (combined.json `raw.id`).
- 0b qtype tag: not surfaced by the loader either; it is the annotation `type`. Mapping to AMBER's three discriminative dimensions (official `inference.py` `de/da/dr`): `discriminative-hallucination` → **existence** (4924), `discriminative-attribute-*` (state/number/action) → **attribute** (7628), `discriminative-relation` + `relation` → **relation** (1664). Totals reconcile to 14216.
- 0c metric convention: AMBER-official P/R uses positive class = `no` (negative/hallucination detection). Per plan default, used **POPE convention (positive = `yes`)** so AMBER/POPE read on the same axis; recorded as `positive_class`/`metric_convention` in the summary. `neg_item_accuracy` carries the negative-detection signal regardless.
- 0d parser: reused POPE's `_normalize_yes_no` (no second parser written). `--amber_task` already threads through `run_evaluation`; no duplicate added.

**Code changes:**
- `src/dataset.py::load_amber`: added `_load_amber_annotations` + `_amber_discriminative_qtype`; for discriminative items the loader now joins `annotations.json` by `raw.id` and sets `label` = gold yes/no and `category` = existence/attribute/relation. Also now applies `task` filter + `limit` **before** image loading (behavior-preserving; avoids opening the full ~14k-image set on limited runs). Generative items untouched.
- `evaluation/classifiers/metrics.py`: added `score_amber_discriminative_records` (POPE-style: `accuracy/precision/recall/f1`, `yes_ratio`, `neg_item_accuracy`/`pos_item_accuracy`, `n_neg_total`/`n_pos_total`, `n_unparsed`, `by_qtype` keyed existence/attribute/relation; `metric="amber_discriminative"`). `score_amber_records` is now a task dispatcher (discriminative → new scorer; generative → unchanged `_score_amber_generative` placeholder). Registry key `amber` unchanged.
- `evaluation/runners/eval_runner.py`: `print_comparison_table` renders AMBER as `acc/neg_item_acc/yes_ratio` integer percents (column `AMBER a/n/yr`) with a legend; falls back to plain accuracy for generative. No generation/CLI change (`--amber_task` already wired).

**Decisions taken (defaults; overridable):** POPE convention (positive=yes) as primary (AMBER-official positive=no can be an added output later); gold + qtype surfaced via loader extension (vs. score-time join) so saved records are self-contained; AMBER generative left as-is (out of scope).

**Validation:**
- Loader (full discriminative set): n=14216, qtypes {attribute 7628, existence 4924, relation 1664} (matches source counts), gold {no 9427, yes 4789}. Gold-join is by explicit id (not positional): AMBER's paired attribute questions verified to carry opposite gold (e.g. "Is the sky sunny?"→yes, "Is the sky gloomy?"→no; "Is the mountain short?"→yes, "Is the mountain tall?"→no).
- Scorer synthetic checks: perfect responder → acc 1.0, neg/pos item acc 1.0; yes-drift responder ("always Yes") → acc=base-rate, `yes_ratio=1.0`, `neg_item_accuracy=0.0`, `pos_item_accuracy=1.0` (the targeted agreeableness signature); `n_neg_total+n_pos_total==n_total`; `by_qtype` keyed existence/attribute/relation.
- End-to-end smoke (real generation), `llava-hf/llava-1.5-7b-hf`, `CUDA_VISIBLE_DEVICES=0`, `--benchmarks amber --amber_task discriminative --limit 16 --interventions no_intervention`: completed; summary `metric="amber_discriminative"`, `accuracy_overall=0.75 yes_ratio=0.50 neg_item_accuracy=0.75 pos_item_accuracy=0.75 n_neg_total=8 n_pos_total=8 n_unparsed=0` (first-16 are all attribute qtype), `n_neg_total+n_pos_total==n_total`, parser fired (n_unparsed=0). **Limit-16 smoke, not a baseline; output dir deleted.** Real baselines (plan Validation step 4: `no_intervention`, all four models, Policy A) not yet run.

**Deviations from plan:** the plan's premise that a gold join already existed was incorrect (see 0a) — surfaced here and fixed via the loader extension the plan anticipated as a fallback. No other deviations.

---

### 2026-06-22 — CHAIR+AMBER diagnostics, Step 0: CHAIR max-new-tokens provenance check

**Type:** Step-0 provenance probe for the CHAIR+AMBER diagnostics (reproduction grid + rotation-strength sweep). Single model, no intervention, tiny n; not an experiment cell. No W&B. Branch/commit: uncommitted working tree.

**Setup:** `llava-hf/llava-1.5-7b-hf`, `no_intervention`, `CUDA_VISIBLE_DEVICES=0` (GPU 0 free; 1–3 at ~48 GB). 20 COCO val2014 images drawn with a fixed seed (`seed=1234`, sampled from `data/chair/combined.json`), identical ids at both caps. Verbatim VTI prompt `"Please Describe this image in detail."` (capital D). Captions generated via `wrapper.generate_vl(image, prompt, max_new_tokens=cap)` (same path the steered runs use). Scored with `score_chair_records`; `coverage` = pooled `sum|mentioned ∩ gt| / sum|gt|`. Script: `evaluation/chair_amber_diagnostics/step0_chair_token_cap.py`. Artifact: `evaluation/results/2026-06-22/_diagnostics/step0_chair_token_cap.json`.

**Results (n=20):**

| cap | CHAIR_s | CHAIR_i | avg_objects | avg_caption_len_chars | coverage(recall) |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.300 | 0.1667 | 2.40 | 266.2 | 0.5714 |
| 512 | 0.650 | 0.2400 | 3.75 | 465.1 | 0.8143 |

**Paper baseline (VTI / Liu et al., ICLR 2025, Table 2, max_new_tokens=512, LLaVA-1.5 "Vanilla", n=500):** CHAIR_S = 51.0, CHAIR_I = 15.2, Recall = 75.2, Avg.Len = 102.2 (Avg.Len in the paper is words, not chars).

**Cap-vs-baseline note (which lands closer):** on CHAIR_I (image-level, the metric VTI's vision arm targets) cap 64 (16.67) is closer to the paper's 15.2 than cap 512 (24.0); on CHAIR_S and recall cap 512 (65.0 / 81.4) is closer to the paper's 51.0 / 75.2 than cap 64 (30.0 / 57.1). n=20 is small and noisy (CHAIR_S especially, being a sentence-level binary over 20 captions), and this scorer uses a rule-based synonym matcher (no nltk/pattern), so absolute offsets vs the paper's n=500 are expected.

**Decision (pre-registered rule applied):** 512 is NOT dramatically closer across the board (it is farther on CHAIR_I), so the default holds — **CHAIR `max_new_tokens` is frozen at 64** for every cell of both experiments and both models. No flag/stop raised.

---

### 2026-06-22 — CHAIR+AMBER diagnostics: scaffolding implemented + experiments launched (interrupted)

**Type:** code implementation + scorer/driver smoke validation, then a first (incomplete) GPU launch. No W&B. Branch/commit: uncommitted working tree.

**Open items confirmed before building:** CHAIR routes the caption prompt through `intervention.generate` (steering active) ✓; AMBER `by_qtype` populates existence/attribute/relation ✓; `vti_textual_uniform_rotation_layer` registered in `ALL_INTERVENTIONS` ✓. Two plan-vs-repo conflicts surfaced and handled: (a) the verbatim VTI CHAIR prompt is capital-D `"Please Describe this image in detail."` while the loader/`combined.json` store lowercase — handled via a new `--chair_prompt` override (default keeps stored prompt); (b) the plan's `query_discriminative.json` for AMBER existence is the FULL discriminative file, not existence-only — stratified instead by the annotation-derived `category` (existence/attribute/relation).

**Code changes:**
- `src/dataset.py`: `load_chair(subset_ids=, prompt_override=)` and `load_amber(subset_ids=)` — pinned-id subset filtered BEFORE image load (takes precedence over `limit`); CHAIR prompt override per-sample.
- `evaluation/classifiers/metrics.py::score_chair_records`: §C empty-caption handling — empty/whitespace captions treated as undefined; `chair_i`/`chair_s`/`avg_objects_mentioned` computed over `n_nonempty`; new fields `n_empty`/`empty_fraction`/`n_nonempty`; `avg_caption_len_chars` kept over all scoreable (shows collapse).
- `evaluation/runners/eval_runner.py` + `evaluation/run_eval.py`: `--subset_ids_file` (`{benchmark:[ids]}` or flat list; threaded as `subset_ids` per benchmark) and `--chair_prompt`.
- NEW `evaluation/chair_amber_diagnostics/`: `draw_subsets.py`, `step0_chair_token_cap.py`, `rotation_strength_chair_amber.py` (Exp 2 driver; reuses generic gen/classify/flip helpers from `evaluation/vti_rotation_strength/rotation_strength.py`), `make_diagnostic_summary.py`, and `run_scripts/{run_prep_subsets,run_step0_chair_cap,run_exp1_repro_grid,run_exp2_rotation_strength,run_report}.sh`.
- Pinned subsets drawn (seed 1234): `data/chair/pinned_chair_500.json` (n=500), `data/amber/pinned_amber_disc_450.json` (n=450; existence/attribute/relation = 150 each; gold no=276/yes=174).

**Validation (no full sweep):** infra unit-checked offline (subset filter, prompt override, §C empties). Exp 2 driver smoke (LLaVA, GPU 0, 3-sample throwaway subsets, β={0.4,0.6}, run_date `_smoke`, since deleted): both arms wrote the documented schema — AMBER `metrics_by_beta` with `by_qtype` + flip accounting + probes + gold-join sanity; CHAIR §C fields present and the length collapse visible (avg_len 260→4.7 at β=0.6, LLaVA-shortens signature).

**First launch (INCOMPLETE — recorded as run history):** Exp 1 + Exp 2 started for both models on GPU 0 in two model-split foreground terminals (`RUN_DATE=2026-06-22`); the SSH tunnel dropped and the foreground processes were SIGHUP-killed mid-Experiment-1. Partial cells on disk: real baselines completed for both models/benchmarks (e.g. LLaVA CHAIR no_intervention chair_s=0.316 chair_i=0.170 n=500 avg_len=269.8; LLaVA AMBER no_intervention acc=0.767 yes_ratio=0.398 neg_item_acc=0.801 n=450, by_qtype all three dims), plus a partial β grid (LLaVA chair through β≈0.2, amber through β≈0.1; Qwen2.5 only β=0.4). Experiment 2 did not start (no `sweep_*.json`). Two leftover `responses.checkpoint.json` (mid-cell kills) — resume-safe (a cell is "done" only when `metric_summary.json` exists; `--skip_if_exists` resumes the rest). Root cause: foreground run without `nohup`/tmux. Relaunch is via `nohup` (Exp 1 resumes, Exp 2 runs fresh).

---

### 2026-06-25 — CHAIR+AMBER diagnostics summary (pointer)

Consolidated diagnostic report for run_date 2026-06-22 written to `evaluation/results/2026-06-22/_diagnostic_summary_chair_amber.md` (Experiment 1 reproduction grid + Experiment 2 rotation-strength sweep; facts only).

---

### 2026-06-27 — CHAIR+AMBER diagnostics: consolidated results (run_date 2026-06-22)

**Type:** factual run record (aggregated from on-disk cells; no new GPU run)
**Branch/commit:** uncommitted working tree
**Source artifact:** `evaluation/results/2026-06-22/_diagnostic_summary_chair_amber.md` (generated by `evaluation/chair_amber_diagnostics/make_diagnostic_summary.py`)
**Models:** `llava-hf/llava-1.5-7b-hf`, `Qwen/Qwen2.5-VL-7B-Instruct`
**Benchmarks:** CHAIR (pinned n=500, `data/chair/pinned_chair_500.json`), AMBER discriminative (pinned n=450, `data/amber/pinned_amber_disc_450.json`)
**Frozen eval knobs:** `--chair_max_new_tokens 64`, `--chair_prompt "Please Describe this image in detail."`, `--amber_task discriminative`, subset seed 1234
**Interventions:** Exp1 = `vti_textual_additive_mlp`, `vti_textual_additive_layer`, `vti_textual_uniform_rotation_mlp` × β grid `{0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9}`; Exp2 = `uniform_rotation @ layer` × β `{0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1}`. Vision arm not run (textual β only).

**Completion status:**
- **LLaVA-1.5:** Exp1 CHAIR+AMBER full β grid on disk. Exp2 CHAIR+AMBER sweeps complete (n=500 / n=450).
- **Qwen2.5-VL:** Exp1 **partial** — CHAIR and AMBER cells through β≈0.4 only (not full grid). Exp2 AMBER sweep complete (n=450); Exp2 CHAIR sweep **missing** (empty table in summary).
- First launch 2026-06-22 was SIGHUP-killed mid-Exp1; cells resumed with `nohup` + `--skip_if_exists`.

**Baselines (no_intervention):**

| Model | Benchmark | Headline metrics | n |
|-------|-----------|------------------|---|
| llava-1.5-7b-hf | CHAIR | chair_s=31.6%, chair_i=17.0%, avg_objects=2.3, avg_len=269.8 chars | 500 |
| llava-1.5-7b-hf | AMBER disc | acc=76.7%, neg_item_acc=80.1%, yes_ratio=39.8%, f1=70.3% | 450 |
| qwen2.5-vl-7b-instruct | CHAIR | chair_s=18.4%, chair_i=14.0%, avg_objects=1.5, avg_len=310.3 chars | 500 |
| qwen2.5-vl-7b-instruct | AMBER disc | acc=64.2%, neg_item_acc=82.6%, yes_ratio=14.4%, f1=51.0% | 450 |

**Experiment 1 — selected cells (β=0.4 unless noted):**

| Model | Benchmark | Intervention | metrics | notes |
|-------|-----------|--------------|---------|-------|
| LLaVA | CHAIR | additive_mlp @ 0.4 | chair_s/i = 31.8% / 17.0% | Δchair_i ≈ 0 vs baseline |
| LLaVA | CHAIR | uniform_rotation_mlp @ 0.9 | chair_s/i = 29.8% / 16.9% | Δchair_i ≈ -0.2 |
| LLaVA | AMBER | additive_mlp @ 0.4 | acc/neg/yes = 77.3% / 83.3% / 36.4% | yes_ratio −3.3 pts |
| LLaVA | AMBER | uniform_rotation_mlp @ 0.4 | acc/neg/yes = 77.8% / 79.0% / 42.2% | yes_ratio +2.4 pts |
| Qwen2.5 | CHAIR | uniform_rotation_mlp @ 0.4 | chair_s/i = 19.6% / 14.7% | Δchair_i +0.6 |
| Qwen2.5 | AMBER | uniform_rotation_mlp @ 0.4 | acc/neg/yes = 69.1% / 78.3% / 23.8% | acc +4.9 pts; n_unparsed=76 |

**Experiment 2 — rotation @ layer bookends (LLaVA):**

| Benchmark | β | metrics | length / collapse |
|-----------|---|---------|-------------------|
| CHAIR | 0.4 | chair_s/i = 20.6% / 13.1% | avg_len=250.0 |
| CHAIR | 0.6 | chair_s/i = 0.0% / 0.0% | avg_len=6.9 (caption collapse) |
| AMBER | 0.4 | acc/neg/yes = 66.7% / 92.4% / 14.7% | h+=6, h−=40 |
| AMBER | 0.6 | acc/neg/yes = 62.4% / 98.6% / 2.9% | n_unparsed=1 |

**Experiment 2 — Qwen2.5-VL AMBER sweep:** baseline acc=76.0% neg_item_acc=96.4% yes_ratio=17.8% (n_unparsed=21); β=0.4 acc=84.2% neg_item_acc=81.9% yes_ratio=43.8%; β=0.6 acc=69.1% neg_item_acc=50.4% yes_ratio=68.7% (h+=127 at β=0.6).

**Mitigation probes (Exp2 β_max):** LLaVA CHAIR `decode_only` chair_i=16.0; `skip_pos0` chair_i=16.8. LLaVA AMBER `decode_only` restores baseline acc=76.7%. Qwen2.5 AMBER `decode_only` acc=76.0%; `skip_pos0` acc=69.1% neg_item_acc=50.0%.

**Output paths:**
- Exp1: `evaluation/results/2026-06-22/{model_short}/{chair|amber}/{iv}__b{beta}/metric_summary.json`
- Exp2: `evaluation/results/2026-06-22/{model_short}/{chair|amber}_rotation_strength/sweep_uniform_rotation_layer_n{N}.json`
- Summary: `evaluation/results/2026-06-22/_diagnostic_summary_chair_amber.md`

**Open items:** Qwen2.5 Exp1 remaining β cells; Qwen2.5 Exp2 CHAIR sweep; vision-encoder VTI arm (α) not implemented.

---

### 2026-06-27 — IMPLEMENTATION.md updated for vision-encoder hook planning

**Type:** documentation (no run)
**Change:** `IMPLEMENTATION.md` now documents decoder-only hook status, per-model vision encoder module paths/dims/token counts, reference VTI vision-arm code in `VTI/`, design constraints, and an implementation checklist for `VisionDispatch` + `vti_visual_*` interventions. Standing near-term plan in this log updated with vision-arm step.

---

### 2026-06-28 — Visual VTI implementation + build acceptance (LLaVA-1.5)

**Type:** implementation + acceptance probes (not full smoke grid)
**Branch/commit:** uncommitted workspace
**Model:** `llava-hf/llava-1.5-7b-hf`

**Code landed:** `src/vision_dispatch.py`; `evaluation/interventions/vti/visual_{perturb,capture,directions,hooks,intervention}.py`; registry keys `vti_visual_{additive,uniform_rotation}_{mlp,layer}` (4 variants); `--alpha` / `run_evaluation(alpha=...)` with `__a{alpha}` result dirs; smoke driver `diagnostic_experiments/vti_visual_smoke/run_smoke.py` + `run_scripts/run_vti_visual_smoke.sh`; `analyze_gate.py`; `tests/test_visual_vti_reference_parity.py`. `IMPLEMENTATION.md` updated (vision arm no longer “planned only”).

**Commands run:**
```bash
python -m pytest tests/test_visual_vti_reference_parity.py -q   # 2 passed
python -m pytest tests/test_VTI_text_steer.py -q                # 6 passed
CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/vti_visual_smoke/run_smoke.py --stage verify
# minimal hook smoke (reduced extraction: num_demos=3, num_trials=2):
CUDA_VISIBLE_DEVICES=0 python -c '... vti_visual_additive_layer alpha=0.4 ...'  # inline probe
```

**Acceptance results:**
- `verify_vision_layout`: family=llava, n_hidden_state_rows=25, n_encoder_layers=24, n_tokens=577, hidden_dim=1024.
- Minimal hook smoke: CHAIR sample `chair_000000146973`, `max_new_tokens=32`, non-empty response (len=132 chars).
- Direction cache (probe config): `experiment_artifacts/vti/llava-1.5-7b-hf/visual/patch_mask_r0.99_zero_top_pc_nd3_nt2_s42_efaf8757706e/` (`directions.npz`, `metadata.json`, `perturbed_examples/`).

**Bug fixed during acceptance:** `prepare_pixel_tensor` now uses `processor.image_processor` only (full LLaVA processor requires text).

**Not run yet:** full smoke stages 0/1/1b (pinned AMBER-25 + CHAIR-5 grid); end-to-end reference parity cosine test vs vendored `obtain_visual_vti`; production direction cache (`num_demos=70`, `num_trials=50`).

**To launch full smoke:**
```bash
bash diagnostic_experiments/vti_visual_smoke/run_scripts/run_vti_visual_smoke.sh
```

---

## 2026-07-12 — demos_v2 pipeline implementation + mock dry-run

**Branch/commit:** uncommitted workspace
**Compute:** lambdab2 CPU-only (no GPU, no W&B)

**Code landed:** `data_scripts/vti_demos_v2/` (stages 0–5, `mllm_client`, `validators`, `export_vti_flat`, `render_review`, `config`); `src/paths.py` helpers `vti_demos_v2_dir` / `vti_demos_v2_path`; `tests/test_demos_v2_validators.py`. `IMPLEMENTATION.md` updated.

**Commands:**
```bash
python -m pytest tests/test_demos_v2_validators.py -q   # 16 passed
python data_scripts/vti_demos_v2/stage0_mine_candidates.py --n-candidates 5
python data_scripts/vti_demos_v2/stage1_verify_anchors.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage2_write_truthful.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage3_make_variants.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py --provider mock --limit 5
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 5
python data_scripts/vti_demos_v2/export_vti_flat.py --dimension counting
python data_scripts/vti_demos_v2/render_review.py --stage 0 --limit 5
python data_scripts/vti_demos_v2/render_review.py --stage 3
python data_scripts/vti_demos_v2/render_review.py --stage final
```

**Yields (mock dry-run):** stage0=5 candidates; stages 1–4 = 5/5 pass, 0 reject; final `data/vti/demos_v2.jsonl` n=5, content hash `8b621806e2a2d0fa`. Co-occurrence cache written to `data/vti/v2/cooccurrence.json`.

**Note:** `stage0_candidates.jsonl` currently holds only the 5 dry-run rows. Re-run stage0 with `--n-candidates 300` before the paid pilot / full pipeline. Direction-cache demos identity + `run_eval --demos_path` remain out of scope (plan §12).

---

## 2026-07-12/13 — demos_v2 full-300 paid POC run

**Branch/commit:** uncommitted workspace
**Compute:** lambdab2 CPU + Anthropic API (no GPU). Driver: `data_scripts/vti_demos_v2/run_full300.sh`.

**Code state at run:** plural-aware stage-2 category match; stage-3 `deterministic_existence_insert` (incl. `includes`/… leads) + text-only Haiku grammar polish; LLM existence fallback; `io_utils.read_jsonl` beautify-tolerant.

**Command:**
```bash
nohup bash data_scripts/vti_demos_v2/run_full300.sh \
  > data/vti/v2/logs/full300_nohup.out 2>&1 &
```

**Funnel (from `data/vti/v2/stage*_summary.json`):**

| Stage | Pass | Reject | Headline reject / notes |
|-------|------|--------|-------------------------|
| 0 | 300 | — | kept from mine |
| 1 | 240 | 60 | count/relation/distractor vision fails (sonnet-4-6) |
| 2 | 232 | 8 | `structural_validation_failed` (opus-4-8) |
| 3 | 208 | 24 | `deterministic_diff_failed`; existence sources: 199 `deterministic+grammar`, 9 `llm` (haiku-4-5) |
| 4 | 131 | 77 | `faithfulness_mismatch` (sonnet-4-6) |
| 5 | 131 | — | `--n-final 300`; all stage-4 passes kept |

**Outputs:** `data/vti/demos_v2.jsonl` n=131, `content_hash_sha256_16=ad44185346c48958`, `pipeline_version=demos_v2_2026-07-12`. Stage artifacts under `data/vti/v2/`.

**Docs:** `IMPLEMENTATION.md` § `data/vti/demos_v2` expanded to document the POC pipeline (stages, schemas, funnel, run commands). Further improvements TBD.

---

## 2026-07-13 — demos_v2.1 diversity-pipeline implementation

**Branch/commit:** uncommitted workspace. **Compute:** CPU only; no paid model run.

**Reconciliation findings:** old Stage 0 horizontal selection used aggregate category extrema and allowed multi-instance categories; v2.1 mining now uses singleton non-crowd pairs and `geometry.evaluate_all_relation_types` edge predicates. Old Stage 3 replacement was caption-global; v2.1 spans carry sentence indexes and replacements are sentence-scoped. Span consumers are stages 2–3 and validators/review code.

**Implemented:** option-set Stage 0/1 records, seeded Stage 1b allocation, four relation types, `at_most` counting, sentence-scoped spans, `h_values.all`, dynamic Stage 4 failure dimensions, subtype export, distribution summaries, and v2.1 run driver.

**Verification command:**
```bash
conda run -n vlm_hallucination_mitigation python -m pytest \
  tests/test_demos_v2_validators.py tests/test_demos_v2_allocator.py -q
```
**Result:** 22 passed. No full or paid dry-run was executed; v2.0 final artifacts were preserved.

## 2026-07-13 — Task-0 rejection diagnostics (v2.0 300-run)

**Command:** `python data_scripts/vti_demos_v2/analyze_rejections.py`  
**Output:** `data/vti/v2/rejection_analysis.json` (copy also under `data/vti/v2_poc_2026-07-12/`)

**Stage-1 rejects (n=60):** relation_check=35, counting_check=24, distractor=1.

**Stage-4 rejects (n=77), by v2.0 8-statement idx map:** truthful_S3=38, existence_false=27, truthful_S1=15, truthful_S4_relation=13, relation_false=6, truthful_S2=2, counting_false=2.

**§2 reconciliation (pre–v2.1):**
1. Stage-0 horizontal used aggregate category extrema (`max A.x2` vs `min B.x1`) and allowed multi-instance categories — not per-instance edge-to-edge.
2. Stage-3 `_replace_once` was caption-global (first match).
3. Spans consumed by stage2/3, validators, render_review only.

**Frozen v2.0 artifact copy:** `data/vti/v2_poc_2026-07-12/` (includes `demos_v2.jsonl`). Live `data/vti/v2/stage0_candidates.jsonl` remains the v2.0 option-less schema until a fresh v2.1 mine.

---

## 2026-07-13 — demos_v2.1 full-1000 Stage 0→5 assemble

**Compute:** lambdab2 (Anthropic API only; no GPU eval). **Pipeline:** `demos_v2.1_2026-07-13`.

**Providers:** Stage 2 `anthropic:claude-opus-4-8`; Stage 3 `anthropic:claude-haiku-4-5`; Stages 1/4 default `anthropic:claude-sonnet-5` after mid-run switch (early Stage 1 calls used `claude-sonnet-4-6`). `MLLMClient` disables Sonnet 5 adaptive thinking via `extra_body`.

**Commands (resume path after killing `run_full.sh`; do not re-wipe):**
```bash
python data_scripts/vti_demos_v2/stage1_verify_anchors.py
python data_scripts/vti_demos_v2/stage1b_allocate.py
python data_scripts/vti_demos_v2/stage2_write_truthful.py
python data_scripts/vti_demos_v2/stage3_make_variants.py
python data_scripts/vti_demos_v2/stage4_verify_faithfulness.py
python data_scripts/vti_demos_v2/stage5_assemble.py --n-final 1000
```
Log: `logs/demos_v2_1_continue_sonnet5_2026-07-13.log`.

**Funnel (from stage summaries + `stage5_summary.json`):**

| Stage | Pass | Reject |
|-------|------|--------|
| 0 | 1000 | — |
| 1 | 835 | 165 |
| 1b | 835 | — |
| 2 | 827 | 8 |
| 3 | 813 | 14 |
| 4 | 555 | 258 |
| 5 | 555 | — |

**Outputs:** `data/vti/demos_v2.jsonl` n=555, `content_hash_sha256_16=9a44f4afde0324b5`, `pipeline_version=demos_v2.1_2026-07-13`. Stage artifacts under `data/vti/v2/`. Stage-3 existence: `n_existence_deterministic_grammar=771`, `n_existence_llm=41`.

**Code deviations during run:** Stage 3 `NameError: relation` fixed (`relation = rec["relation"]`); Stage 1/4 provider switched to Sonnet 5; validators previously fixed for distractor-leak on `all` and sentence-scoped attributes. `IMPLEMENTATION.md` § demos_v2 rewritten for v2.1 live schema + this funnel.


---

## 2026-07-13 — demos_v2.1 textual directions + qualitative-subset steering grid (launch)

**Plan:** `implementation_plans/vti_demos_v2_qual_grid_plan.md`  
**Target:** lambdab2 GPU 0 (A6000, idle); both models concurrent on the same GPU  
**Commit at launch:** `79e4da5` (working tree has uncommitted implementation for this run)  
**Run date:** `2026-07-13`

### Clarifications applied (vs plan text)

- **PCA:** live textual path kept for consistency — global flatten PCA + steering = reshape(`PC1 + mean`); rank-2 fitted so PC2 is cached in `components.npz`; **not** per-layer pure PC1.
- **Diff polarity:** live path `act(value) − act(h_value)` (`value` = truthful caption; `h_value` / `h_values[d]` = hallucinated caption for dimension `d`).
- **run_date:** `2026-07-13` (not 07-14).

### Pins / plumbing added

- `data/vti/demos_v2_order_s42.json` (555 ids, seed 42, hash `9a44f4afde0324b5`)
- `data/vti/qual_subset_chair5_amber25.json` (CHAIR-5 + AMBER-25 from 2026-06-22 LLaVA sample bundles)
- `data/vti/qual_subset_chair1_smoke.json` (1 CHAIR id for smoke gate)
- `evaluation/interventions/vti/directions_v2.py`, extract CLI, `run_demosv2_qual_grid.sh`, `helper_scripts/render_demosv2_qual_review.py`
- `run_eval.py` / `run_evaluation`: `--demos_path --vector_dimension --num_demos --rank --max_pixels`; result suffix `{iv}__b{beta}__d{dim}__nd{N}`

### Valid-pair counts (555 finals)

All dimensions have 555/555 non-empty `h_values` → `nd=500` does not shrink.

### Launch commands

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_demosv2_qual_grid.sh \
  llava-hf/llava-1.5-7b-hf \
  > logs/demosv2_qual_llava_2026-07-13.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup bash evaluation/run_scripts/run_demosv2_qual_grid.sh \
  Qwen/Qwen2.5-VL-7B-Instruct \
  > logs/demosv2_qual_qwen_2026-07-13.log 2>&1 &
```

**Frozen eval facts:** CHAIR prompt `"Please Describe this image in detail."`; `--chair_max_new_tokens 512`; AMBER discriminative; β∈{0.5,0.2,0.9}; nd∈{50,100,200,500}; dims `all` then existence→attribute→counting→relation; Qwen `max_pixels=1003520` (Exp1 had no cap); no W&B.

**Expected outputs:** directions under `experiment_artifacts/vti/{model_short}/textual_v2/`; cells under `evaluation/results/2026-07-13/{model_short}/{chair,amber}/`; galleries `…/_samples/…/*_demosv2_{dim}_review.html`. Smoke under `evaluation/results/2026-07-13_smoke/`.

**Note:** completion metrics / EV headlines / cell counts to be appended when the overnight run finishes.

---

## 2026-07-14 — demos_v2.1 qual grid morning status (partial)

**LLaVA-1.5-7b-hf:** finished. Log ends with `=== demos_v2 qual grid finished for llava-1.5-7b-hf ===`. Cells: 241 chair + 241 amber (1 baseline + 240 intervention). Galleries written for dims `{all, existence, attribute, counting, relation}` under `evaluation/results/2026-07-13/_samples/llava-1.5-7b-hf/{amber,chair}/`.

**Qwen2.5-VL-7B-Instruct:** still running at check time (GPU 0). Complete through counting (48 cells/dim × 4 dims + baseline); relation in progress (`nd=50 β=0.9` at check). Chair 205 / amber 203 complete of 241. Galleries present for `{all, existence, attribute, counting}` only. No OOM/fatal in `logs/demosv2_qual_qwen_2026-07-13.log`.

**Smoke (both):** passed — hash `9a44f4afde0324b5`, `__dall__nd500` dir suffix, additive≠baseline (`evaluation/results/2026-07-13_smoke/`).

**PC1 EVR (nd=500, from metadata.json):** LLaVA all/exist/attr/count/rel = 0.195/0.226/0.156/0.232/0.220; Qwen = 0.646/0.492/0.783/0.680/0.563. All direction slugs present (20/model).

---

## 2026-07-15 — late log: VTI visual-arm smoke AMBER results (run 2026-07-02)

**Type:** factual run record (late append; run completed 2026-07-02, not previously logged)  
**Target:** lambdab2  
**Model:** `llava-hf/llava-1.5-7b-hf` (`llava-1.5-7b-hf`)  
**Driver:** `diagnostic_experiments/vti_visual_smoke/`  
**Purpose:** qualitative first-light for the visual VTI arm, with textual-arm Stage 0 as AMBER gate positive control.

**Subset / scoring (AMBER only logged here):**
- AMBER: 25 pinned discriminative items from `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/amber/llava-1.5-7b-hf_amber_response_samples.json` (`ordered`; stratified 5-per-(qtype × gold) from that draw)
- Metric: `amber_discriminative` (POPE-aligned yes-positive); per-cell also stores `per_item_p_yes.json` for offline gate readout (c-AUC, acc@0.5, h+/h− vs baseline)
- Stage 2 discrepancy toggles (`legacy_pc_plus_mean`, `mask_fill=mean`, etc.) were not run

**Command:**
```bash
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/romanus/huggingface
RUN_DATE=2026-07-02 bash diagnostic_experiments/vti_visual_smoke/run_scripts/run_vti_visual_smoke.sh
```
(stages: `verify` → `0` → `1` → `1b` → `analyze`)

**Cells (16 unique conditions on AMBER):**
- Stage 0 (textual gate control): `no_intervention`; `vti_textual_uniform_rotation_layer` β∈{0.2,0.4}; `vti_textual_additive_layer` β=0.9
- Stages 1/1b (visual arm): `vti_visual_{additive,uniform_rotation}_{layer,mlp}` × α∈{0.2,0.4,0.9}

**Output root:** `evaluation/results/2026-07-02/llava-1.5-7b-hf/amber/{intervention}/`  
(`responses.json`, `metric_summary.json`, `per_item_p_yes.json` per cell)  
**Gate report / plots:** `evaluation/results/2026-07-02/_diagnostics/vti_visual_smoke_summary.md`, `…/gate_plots/`  
**Derived aggregates (written 2026-07-15):**  
`evaluation/results/2026-07-02/llava_amber_responses_scored.json`,  
`evaluation/results/2026-07-02/metric_summaries.json`

**AMBER-25 headline metrics** (from each cell’s `metric_summary.json` + gate columns from `_diagnostics/vti_visual_smoke_summary.md`; all cells `n_total=25`, `n_unparsed=0`):

| Cell | Stage | acc | neg_acc | yes_ratio | f1 | c-AUC | acc@0.5 | h+ | h− |
|------|-------|-----|---------|-----------|----|-------|---------|----|----|
| `no_intervention` | baseline | 84.0% | 100.0% | 24.0% | 75.0% | 0.933 | 80.0% | — | — |
| `vti_textual_additive_layer__b0.9` | Stage 0 | 72.0% | 86.7% | 28.0% | 58.8% | 0.767 | 72.0% | 2 | 0 |
| `vti_textual_uniform_rotation_layer__b0.2` | Stage 0 | 80.0% | 100.0% | 20.0% | 66.7% | 0.913 | 80.0% | 0 | 0 |
| `vti_textual_uniform_rotation_layer__b0.4` | Stage 0 | 72.0% | 100.0% | 12.0% | 46.2% | 0.827 | 72.0% | 0 | 0 |
| `vti_visual_additive_layer__a0.2` | Stage 1/1b | 80.0% | 93.3% | 28.0% | 70.6% | 0.947 | 84.0% | 0 | 0 |
| `vti_visual_additive_layer__a0.4` | Stage 1/1b | 84.0% | 100.0% | 24.0% | 75.0% | 0.967 | 84.0% | 0 | 0 |
| `vti_visual_additive_layer__a0.9` | Stage 1/1b | 84.0% | 93.3% | 32.0% | 77.8% | 0.947 | 84.0% | 0 | 0 |
| `vti_visual_additive_mlp__a0.2` | Stage 1/1b | 80.0% | 93.3% | 28.0% | 70.6% | 0.947 | 84.0% | 0 | 0 |
| `vti_visual_additive_mlp__a0.4` | Stage 1/1b | 80.0% | 93.3% | 28.0% | 70.6% | 0.967 | 84.0% | 0 | 0 |
| `vti_visual_additive_mlp__a0.9` | Stage 1/1b | 84.0% | 93.3% | 32.0% | 77.8% | 0.947 | 84.0% | 0 | 0 |
| `vti_visual_uniform_rotation_layer__a0.2` | Stage 1/1b | 84.0% | 100.0% | 24.0% | 75.0% | 0.940 | 84.0% | 0 | 0 |
| `vti_visual_uniform_rotation_layer__a0.4` | Stage 1/1b | 84.0% | 100.0% | 24.0% | 75.0% | 0.967 | 84.0% | 0 | 0 |
| `vti_visual_uniform_rotation_layer__a0.9` | Stage 1/1b | 88.0% | 100.0% | 28.0% | 82.4% | 0.927 | 84.0% | 0 | 0 |
| `vti_visual_uniform_rotation_mlp__a0.2` | Stage 1/1b | 84.0% | 100.0% | 24.0% | 75.0% | 0.933 | 84.0% | 0 | 0 |
| `vti_visual_uniform_rotation_mlp__a0.4` | Stage 1/1b | 80.0% | 93.3% | 28.0% | 70.6% | 0.933 | 84.0% | 0 | 0 |
| `vti_visual_uniform_rotation_mlp__a0.9` | Stage 1/1b | 80.0% | 93.3% | 28.0% | 70.6% | 0.973 | 84.0% | 0 | 0 |

**CHAIR:** cells were also written under `evaluation/results/2026-07-02/llava-1.5-7b-hf/chair/` with driver `chair_max_new_tokens=64`, but **headline CHAIR metrics are not logged here** — captions were truncated / partially generated at that cap, which biases `chair_s` / `chair_i`. Re-run with a higher cap before treating CHAIR numbers as valid.

**Deviation from prior “not run yet” note (2026-06-28 entry):** full smoke stages 0/1/1b on pinned AMBER-25 (+ CHAIR-5 on disk) did complete on 2026-07-02; only AMBER is recorded in this late log.


## 2026-07-19 — Perception dump retention (cleanup)

Removed retired perception-plan code (analyses, gates, HD QC, RunAI perception helpers, experiment plan markdowns) and derived plots/JSON overlays. **Kept** only steered capture dumps and the scripts that produce them.

### LLaVA-1.5 AMBER-25 dump

- **Path:** `diagnostic_experiments/perception_diag/llava-1.5-7b-hf/dumps/amber25_all_steering_settings/`
- **Created:** 2026-07-16 via dump driver (recreate: `run_scripts/run_llava_amber25_dump.sh`) with `--cells all --run_tag amber25_all_steering_settings --max_new_tokens 512`
- **Contents:** 13 steering-setting directories (plain-English names below); 25 items × 5 prompt conditions = 125 rows/setting; `manifest.jsonl` + `acts/*.npy` + `norms/*.npy` + `run_metadata.json`
- **Direction:** `demosv2_9a44f4af_all_nd200_s42_r2_prefix`

### Qwen2.5-VL AMBER-25 dump

- **Path:** `diagnostic_experiments/perception_diag/qwen2.5-vl-7b-instruct/dumps/amber25_steering_settings/`
- **Created:** 2026-07-17 via dump driver (recreate: `run_scripts/run_qwen25_amber25_dump.sh`) with `--cells default_subset --run_tag amber25_steering_settings --max_new_tokens 128 --max_pixels 1003520`
- **Contents:** baseline + rotation_{mlp,layer}_{0.2,0.5,0.9} + additive_mlp_{0.2,0.5,0.9} + additive_layer_0.5; 125 rows/setting; same file layout (manifest includes `degeneracy_flag` / `truncated` / `status`)
- **Direction:** same slug as LLaVA dump

### Steering-setting directory names (replaces former B0–B12)

`baseline`; `rotation_mlp_{0.2,0.5,0.9}`; `rotation_layer_{0.2,0.5,0.9}`; `additive_mlp_{0.2,0.5,0.9}`; `additive_layer_{0.2,0.5,0.9}`.

### Inputs used to build dumps

- Prompt JSONL: `diagnostic_experiments/perception_diag/augment/outputs/augmented_amber25.jsonl` (from `templates/leading_clauses_v1.json` via `augment/build_augmented_jsonl.py`)

## 2026-07-19 — Perception dump path rename (plain English)

Renamed dump run tags and cell directories; rewrote `cell_id` / `run_tag` in manifests and metadata; updated `run_dump.py` cell registry and recreate scripts.

| Former | New |
|--------|-----|
| `…/dumps/s1_full_cell_smoke/` | `…/dumps/amber25_all_steering_settings/` |
| `…/dumps/s3b_qwen25_amber25/` | `…/dumps/amber25_steering_settings/` |
| `B0`…`B12` | `baseline`, `rotation_mlp_0.2`, … (see table above) |
| `run_scripts/run_s1_full_cell_smoke.sh` | `run_scripts/run_llava_amber25_dump.sh` |
| `run_scripts/run_s3b_qwen25_amber25.sh` | `run_scripts/run_qwen25_amber25_dump.sh` |

Note: Qwen `additive_layer_0.5/manifest.jsonl` had been pretty-printed; restored to JSONL during rename.

## 2026-07-19 — AMBER-100 + POPE-30 pins, augment, baseline dumps (started)

### Pins (new)

- `data/amber/pinned_amber_disc_100.json` — 100 IDs; keeps all AMBER-25; fill from `pinned_amber_disc_450.json` pin order; **20 per stratum** × {existence×no, attribute×yes/no, relation×yes/no}; verified counts match.
- `data/pope/pinned_pope_existence_yes_30.json` — **10 gold=yes** per split from `pinned_eval_ids.json` order (random / popular / adversarial); verified all labels yes.

### Augment

```bash
python diagnostic_experiments/perception_diag/augment/build_augmented_jsonl.py --subsets amber100
```

- `augment/outputs/augmented_amber100.jsonl` — 100 rows (×5 conditions in variants); hash `56b6d2c9b36375c2`
- Meta merged into `augment/outputs/augmentation_meta.json`

### Baseline dumps (separate run tags; not merged with amber25 trees)

Driver: `run_scripts/run_amber100_pope30_baseline_dumps.sh {llava|qwen25}`  
Flags: `--cells baseline --max_new_tokens 128`; Qwen also `--max_pixels 1003520`.  
GPU: lambdab2 `CUDA_VISIBLE_DEVICES=0`.  
Log: `diagnostic_experiments/perception_diag/logs/amber100_baseline_dumps_20260719_145123.log`

| Model | Run tag | Path | Manifest rows | Acts npy |
|-------|---------|------|---------------|----------|
| LLaVA-1.5-7B | `amber100_baseline` | `…/llava-1.5-7b-hf/dumps/amber100_baseline/` | 500 (`ran=500`) | 500 |
| Qwen2.5-VL-7B | `amber100_baseline` | `…/qwen2.5-vl-7b-instruct/dumps/amber100_baseline/` | 500 | 500 |

Each dump has a single setting dir `baseline/` (not concatenated with AMBER-25 dumps). Completed 2026-07-19 for AMBER-100 baselines.

## 2026-07-20 — Leading-clause attention knockout: Stage 0 (offline)

**Experiment:** `leading_clause_attention_knockout`  
**Commit:** b4b9607 (working tree has new experiment code; commit may lag)  
**Target:** lambdab2, zero GPU

### Commands

```bash
python diagnostic_experiments/perception_diag/augment/build_augmented_jsonl.py --subsets amber100 pope30
bash diagnostic_experiments/leading_clause_attention_knockout/run_scripts/run_stage0_flip_sets.sh
```

### Augment (fillers added)

- Templates: `filler_clauses_v1.json` → conditions `filler_a`, `filler_b` (15 prefix tokens on LLaVA + Qwen2.5 tokenizers)
- `augmented_amber100.jsonl` hash `2a4573547894eac8` (7 conditions/item)
- Meta: `prefix_token_counts_by_model` in `augmentation_meta.json`

### Stage-0 outputs

- Shared: `diagnostic_experiments/leading_clause_attention_knockout/results/stage0/`
  - `first_token_answer_agreement_baseline_dumps.csv`
  - `flip_set_summary_by_model_and_stratum.csv`
  - `flip_set_items_by_model_benchmark_stratum.csv`
  - `stable_set_selected.csv`
  - `stage0_gate_report.json`
- Per model: `{model}/results/stage0/flip_and_stable_sets.json`

### Headline numbers

| Model | Dump | First-token↔parsed agree (parseable) | Flip union (AMBER+POPE) | Stable set |
|-------|------|--------------------------------------|-------------------------|------------|
| LLaVA-1.5-7B | amber100 | 1.0000 (500/500) | | |
| LLaVA-1.5-7B | pope30 | 1.0000 (150/150) | **27** | 27 |
| Qwen2.5-VL-7B | amber100 | 0.9849 (457/464); 7 disagree (generation parse ≠ first-token); 36 unparseable | | |
| Qwen2.5-VL-7B | pope30 | 0.9800 (147/150); 3 disagree all `p_yes==p_no` ties on `tentative_toward_no` | **11** | 11 |

LLaVA AMBER-100 flip strata: relation×yes=11, relation×no=8, existence×no=3, attribute×no=3, attribute×yes=2; POPE-30 flips=0.  
Qwen AMBER-100 flip strata: attribute×yes=6, attribute×no=1, relation×yes=1; POPE-30 flips=3 (1/split).

**Gate:** Qwen flip union 11 < 15 → knockout **not** started for Qwen. LLaVA clears gate.  
**ε proposal (10th pct \|Δlogit-margin\| on flip rows):** 1.0546875 (n=51 flip rows). Pending analyst confirmation before finalizing recovery plots.

**Deviation:** Rewrote Qwen `amber100_baseline/baseline/manifest.jsonl` from pretty-printed multi-line JSON objects to single-line JSONL (`--normalize_manifests`).

## 2026-07-20 — Leading-clause attention knockout: LLaVA cells (lambdab2)

**Experiment:** `leading_clause_attention_knockout`  
**Target:** lambdab2 `CUDA_VISIBLE_DEVICES=0` (A6000)  
**Models run:** LLaVA-1.5-7B only. Qwen2.5-VL knockout **not run** (Stage-0 flip union 11 < 15).  
**ε used in analysis:** 1.0546875 (Stage-0 10th-percentile proposal; pending analyst confirmation)

### Commands

```bash
CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/leading_clause_attention_knockout/run_knockout.py --model llava --scope all_downstream
CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/leading_clause_attention_knockout/run_knockout.py --model llava --scope last_token
CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/leading_clause_attention_knockout/run_knockout.py --model llava --scope all_downstream --skip_verify --filler_peak_window layers_10-19
python diagnostic_experiments/leading_clause_attention_knockout/analyze_knockout.py --model_short llava-1.5-7b-hf
```

Eager verify (pre-sweep): `max_blocked_attention=0.0` on layers 15–24 for one flip item (`amber_disc_00326` / `assertive_toward_no`).

### Result paths

- `…/llava-1.5-7b-hf/results/llava_block_all_downstream_reading/` (944 clause/no-ko records + 54 filler peak-window records)
- `…/llava-1.5-7b-hf/results/llava_block_last_token_reading/` (944 records)
- Plots under each cell dir and `…/results/plots/`
- Logs: `diagnostic_experiments/leading_clause_attention_knockout/logs/`

### Headline metrics (LLaVA)

**Full-depth sanity (all_layers clause knockout vs neutral on flip evaluations, n=37):** agreement=0.5135; `pass_ge_0_80=false` for both query scopes.

**Flip recovery rate by layer window** (`block_all_downstream_reading`):

| window | flip_recovery_rate |
|--------|-------------------|
| layers_0-9 | 0.5135 |
| layers_5-14 | 0.5405 |
| layers_10-19 | 0.7568 |
| layers_15-24 | 0.2703 |
| layers_20-29 | 0.2162 |
| layers_22-31 | 0.1622 |
| all_layers | 0.5135 |

**Flip recovery** (`block_last_token_reading`): peak among windows = layers_0-9 at 0.4865; all_layers 0.5135.

**Filler vs neutral (no knockout, n=54):** filler_a agreement=0.5741, mean Δlogit-margin=+2.55; filler_b agreement=0.6296, mean Δlogit-margin=+1.81.

**Filler-span knockout specificity** (`all_downstream`, flip-set evaluations n=54): layers_10-19 recovery=0.7222; all_layers recovery=0.4815.

**Stable-set false-flip rate** (`all_downstream`, n=54 evaluations): all_layers 0.8148; layers_10-19 0.0370; layers_15-24 0.0000.

### Deviations / notes

- Tokenization invariant implemented as **question-token suffix** of `tok(prefix+question)` (BPE may drop the trailing-space token vs `tok(prefix)+tok(question)`). Clause span in the multimodal sequence = `clause_len` tokens immediately before the matched question span; leftmost clause token may differ from bare `tok(prefix)` after the chat-template newline (interior tokens required to match).
- LLaVA processor emits already-expanded image tokens in `input_ids` (576× `image_token_id`); span mapping does not double-expand.
- Plan sanity check 3 fails (full-depth agreement 0.5135 < 0.80). Per plan: stop for discussion before interpreting window curves.
- Qwen knockout skipped by Stage-0 gate.

## 2026-07-20 — Alex overrides + LLaVA plot refresh; Qwen knockout started

**Overrides (via Romanus):** proceed with Qwen knockout despite flip union 11 and decidedness ~98%; keep ε=1.0546875; Gate 3 (full-depth) remains open for discussion.

**Plot / analysis changes (`analyze_knockout.py`):**
- Layer windows plotted in ascending start-layer order (`layers_0-9` … `layers_22-31`).
- Continuous recovery plot/CSV now uses score=`yes_logit_sum` with **mean** recovery `(ko−leading)/(neutral−leading)`; canonical artifact `mean_yes_logit_recovery_by_layer_window.{csv,png}`. The previous filename `logit_difference_recovery_by_layer_window.png` was overwritten with the same mean yes-logit curve for path continuity.
- Write-up: `answers/attention_knockout/july_20_2026/attention_blocking_and_filler_metrics.md`

**Qwen run:** `run_knockout.py --model qwen25 --allow_small_flip_set` (both scopes), lambdab2 `CUDA_VISIBLE_DEVICES=0`.

### Qwen tokenization note (same day)

Qwen2.5 BBPE merges the first question token with the prefix boundary (`tok(question)` is not always an exact suffix of `tok(prefix+question)`). `assert_tokenization_invariant` now allows a one-token left-edge mismatch; the merged boundary token is assigned to the clause (blocked) span. Eager verify on Qwen: `max_blocked_attention=0.0` (layers 10–19). Full Qwen cells relaunched with `--allow_small_flip_set`.

## 2026-07-20 — Reporting metrics switched to yes-token probability

`analyze_knockout.py` continuous recovery and filler shift now use `scores.p_yes_raw` (not logits / logit margins):

- Recovery: `(p_yes_ko − p_yes_leading) / (p_yes_neutral − p_yes_leading)`, **mean** over flip evaluations
- Artifact: `mean_yes_probability_recovery_by_layer_window.{csv,png}`
- Filler dilution shift: `mean_yes_probability_shift_vs_neutral`
- ε recomputed per cell as 10th percentile of `|p_yes_neutral − p_yes_leading|` on flip rows (LLaVA ≈0.093; Qwen ≈0.032). Prior logit-margin ε=1.05 is not used for these plots.
- Removed obsolete `*logit*` recovery plot/CSV artifacts under this experiment.

Re-ran analysis for LLaVA + Qwen (both scopes). Raw knockout records unchanged (still store logits and probs).

## 2026-07-20 — knockout_records.jsonl: add prompt + question_neutral

Enriched existing `knockout_records.jsonl` for all four cells (LLaVA last-token / all-downstream; Qwen last-token / all-downstream) by joining `(item_id, condition_id)` to `diagnostic_experiments/perception_diag/augment/outputs/{augmented_amber100,augmented_pope30}.jsonl`. Each record now includes `prompt` (full scored text) and `question_neutral`. Missing prompts: 0. Files rewritten as one JSON object per line. `run_knockout.py` updated so future runs write these fields. No GPU re-run.

## 2026-07-20 — filler_b length-matched to assertive_toward_no

Updated `diagnostic_experiments/perception_diag/templates/filler_clauses_v1.json` to v2: `filler_b` prefix is now `Answer the question about the image. ` (trailing space). Token count = 8 on LLaVA-1.5-7B and Qwen2.5-VL-7B, matching `assertive_toward_no`. Prior 15-token filler_b text superseded for new runs. Rebuilt `augment/outputs/augmented_pope120_assertive_no.jsonl` (hash=3db73dd8e287cc27). POPE-120 baseline dumps not restarted yet.

## 2026-07-20 — POPE-120 assertive protocol Stage 0 + drop second filler paraphrase

**Data:** `data/pope/pinned_pope_existence_yes_120.json` (40 gold=yes per random/popular/adversarial). Augment: `augment/outputs/augmented_pope120_assertive_no.jsonl` conditions = neutral, assertive_toward_no, filler_b (`Answer the question about the image. `, 8 tokens = assertive length on LLaVA + Qwen).

**Templates:** `filler_clauses_v1.json` v3 — sole filler `filler_b`. Pipeline code/artifacts/knockout_records stripped of the prior second filler paraphrase.

**Baseline dumps (lambdab2 GPU0):** `pope120_yes_assertive_no_baseline` for LLaVA and Qwen (360 rows each = 120×3).

**Stage 0** (`--dumps pope120:pope120_yes_assertive_no_baseline --out_subdir stage0_pope120`):

| Model | first-token↔parsed | flip_union (assertive_toward_no) | Gate (≥15) |
|-------|--------------------|----------------------------------|------------|
| llava-1.5-7b-hf | 0.9917 (n=360) | 6 (2 per split) | FAIL |
| qwen2.5-vl-7b-instruct | 1.0000 (n=351) | 24 (8 per split) | PASS |

Artifacts: `…/results/stage0_pope120/` and per-model `…/{model}/results/stage0_pope120/flip_and_stable_sets.json`. Condition behavior: `…/stage0_pope120/condition_behavior_sanity.json`. Full knockout not started (LLaVA flip gate).

## 2026-07-20 — POPE-no-120 Stage 0 (gold=no, assertive_toward_yes)

Renamed yes-120 artifacts for clarity: pin remains `pinned_pope_existence_yes_120.json` (protocol_name POPE-yes-120); augment → `augmented_pope_yes_120.jsonl`; dumps → `pope_yes_120_baseline`; Stage 0 → `stage0_pope_yes_120` (content unchanged).

**New:** `data/pope/pinned_pope_existence_no_120.json` (40 gold=no × 3 splits). Augment `augmented_pope_no_120.jsonl` conditions: neutral, assertive_toward_yes, filler_b. Dumps `pope_no_120_baseline` (LLaVA + Qwen). Stage 0 `results/stage0_pope_no_120/`.

| Model | first-token↔parsed | flip_union | Gate (≥15) |
|-------|--------------------|------------|------------|
| llava-1.5-7b-hf | 1.0000 (n=360) | **2** | FAIL |
| qwen2.5-vl-7b-instruct | 0.9944 (n=360) | **0** | FAIL |

Compare POPE-yes-120: LLaVA flip=6, Qwen flip=24. POPE-no-120 did not enlarge the LLaVA flip set.

## 2026-07-20 — LLaVA Stage-0 flip breakdown (AMBER-100 vs POPE-30)

Offline analysis of `llava-1.5-7b-hf/results/stage0/flip_and_stable_sets.json` + amber100/pope30 baseline manifests. Artifacts under `…/stage0/amber100_flip_breakdown/` and copies in `…/results/plots/`. Finding recorded in summary JSON: all 27 flip-union IDs are AMBER; 0 POPE-30.

## 2026-07-21 — Perception dump / augment paths moved under data/{amber,pope}

No new GPU runs. Reorganized shared perception assets out of
`diagnostic_experiments/perception_diag/{model}/dumps/` and
`…/augment/outputs/` into the corresponding benchmark data dirs.

**New layout:**
- Pins (unchanged location): `data/amber/pinned_*.json`, `data/pope/pinned_*.json`
- Augmented JSONLs: `data/amber/augmented_amber{25,100}.jsonl`, `data/pope/augmented_pope{30,_yes_120,_no_120}.jsonl`
- Dumps: `data/{amber|pope}/dumps/{model_short}/{run_tag}/`

**Code updates:** `src/paths.py` helpers (`perception_dump_dir`, `augmented_jsonl_path`, …);
`run_dump.py`, `augment/build_augmented_jsonl.py`, dump run scripts, `compute_flip_sets.py`,
`run_knockout.py`, `remap_lambdab2_paths.py`, `.gitignore`, `IMPLEMENTATION.md`.

**Verification:** Stage-0 path resolution on LLaVA AMBER-100+POPE-30 manifests
(`agree=1.0`; amber100 `flip_union=27`) against the new dump paths.

## 2026-07-21 — Layer-windowed steering Stage A (LLaVA): aggregator crash; AMBER resumed

**Plan:** `implementation_plans/layer_windowed_steering_yes_no_probability_plan_2026-07-21.md`.

**Error:** aggregator `build_windowed_steering_summary.py` raised `JSONDecodeError` on pretty-printed `data/amber/dumps/llava-1.5-7b-hf/amber100_baseline/baseline/manifest.jsonl`. Stage-A shell `set -e` aborted before A2; **no AMBER steered dumps were written at that time**.

**Recovery:** aggregator `load_manifest` updated to parse pretty-printed concatenated JSON; Stage scripts now continue dumps if aggregator fails.

**A2 resumed (2026-07-21 ~12:38):**
```
CUDA_VISIBLE_DEVICES=0 python diagnostic_experiments/perception_diag/run_dump.py \
  --model llava-hf/llava-1.5-7b-hf \
  --augmented_jsonl data/amber/augmented_amber100.jsonl \
  --windowed_grid --conditions gold_conditional \
  --run_tag amber100_windowed_steering --max_new_tokens 128 --device_map cuda:0
```
Log: `diagnostic_experiments/perception_diag/logs/llava_amber100_windowed_steering_20260721_123844.log`.
## 2026-07-21 — POPE-30-yes / POPE-30-no dataset rebuild (CPU only)

**Plan:** `implementation_plans/7-21-26/pope_yes_no_30_dataset_rebuild_plan_2026-07-22.md`
**Target:** lambdab2, CPU, no GPU.

**Commands:**
```
python data_scripts/rebuild_pope_yes_no_30_pins.py
python diagnostic_experiments/perception_diag/augment/build_augmented_jsonl.py --subsets pope30_yes pope30_no --skip_token_counts
```

**Outputs:**
- `data/pope/pinned_pope_existence_yes_30.json` (rewritten in place) — n=30, gold=yes×30, unique (image,question)=30, distinct images=10, content_hash=`f83d5577d7a2356bfe5b686c62f28bc045db6ca22c83b6f047e7cc1ce4b16903`.
- `data/pope/pinned_pope_existence_no_30.json` (new) — n=30, gold=no×30, unique (image,question)=30, distinct images=5 (subset of yes images), 10/split, content_hash=`9e4a7d34323b051ba1bba89c6866df1b05314462200f81ca9e55aa735b41c4e5`. Image overlap with yes set: 5. Collisions skipped: 6 (1 popular, 5 adversarial).
- `data/pope/augmented_pope30_yes.jsonl` — 30 rows, hash=`de763202bb451a23`, template_set=`leading_clauses_v1+filler_clauses_v1`
- `data/pope/augmented_pope30_no.jsonl` — 30 rows, hash=`c31942ead5a2848c`, same template set

**No-set objects by split (for eyeballing negative difficulty):**
- random: car, sandwich, couch, pizza, sink, cell phone, sheep, broccoli, tennis racket, teddy bear
- popular: dining table, chair, dining table, car, chair, dining table, chair, cup, dining table, chair
- adversarial: backpack, dog, handbag, motorcycle, truck, vase, truck, bus, dining table, truck

**Code:** `data_scripts/rebuild_pope_yes_no_30_pins.py` (new); `augment/build_augmented_jsonl.py` subset keys `pope30_yes` / `pope30_no`. `IMPLEMENTATION.md` updated for pin/augment naming.

## 2026-07-21 — POPE-30-yes/no windowed steering control: code + launch (in progress)

**Plan:** `implementation_plans/7-21-26/pope_yes_no_windowed_steering_control_plan_2026-07-22.md`
**Target:** lambdab2 GPU 0 (A6000), unsharded.

**Code (pre-run):**
- `run_dump.py` `--windowed_grid` now includes in-grid `baseline` first → 64 LLaVA / 55 Qwen cells.
- `build_windowed_steering_summary.py` datasets: `pope30_yes`, `pope30_no`, `amber100` (with `dataset` field; in-grid baseline for new tags).
- `build_pope30_mlp_2x2_summary.py` rewritten for per-dataset outputs + gold=no accuracy/`flips_no_to_yes`.
- Drivers: `run_scripts/run_pope30_yes_no_orchestrator.sh`, `run_llava_pope30_yes_no_concurrent.sh`, `run_qwen_pope30_yes_no_concurrent.sh`.

**Deviation from plan ordering:** Alex chat override — run POPE-30-yes and POPE-30-no **concurrently** per model on GPU 0 (plan text was sequential no→yes; only approved cross-model overlap was Qwen-no with LLaVA-yes). Sequence actually used: wait for prior GPU-0 notebook → concurrent LLaVA yes+no → concurrent Qwen yes+no.

**Launch:**
```
nohup bash diagnostic_experiments/perception_diag/run_scripts/run_pope30_yes_no_orchestrator.sh \
  > diagnostic_experiments/perception_diag/logs/pope30_yes_no_orchestrator_nohup_20260721_175918.log 2>&1 &
```
Orchestrator PID (bash): 3902400. At launch GPU0 used=8733 MiB by `jupyter-nbconvert` PID 3849009; orchestrator polling until free (<1500 MiB) then starts dumps.

**Run tags / augments:**
- `pope30_no_windowed_steering` ← `data/pope/augmented_pope30_no.jsonl`
- `pope30_yes_windowed_steering` ← `data/pope/augmented_pope30_yes.jsonl`
- gold_conditional; max_new_tokens=128; demos_v2 all@nd200; Qwen max_pixels=1003520

**Status:** waiting on GPU 0; results / reproducibility / summary numbers to be appended when dumps finish.

## 2026-07-21 — Qwen POPE-30-yes/no queued after GPU-0 clear

Watcher: `run_scripts/wait_gpu0_then_qwen_pope30_yes_no.sh` (logs `logs/qwen_after_gpu0_clear_*.log`).
Waits until notebook + both LLaVA `run_dump.py` (and LLaVA concurrent wrapper) exit, then launches `run_qwen_pope30_yes_no_concurrent.sh` (POPE-30-yes and POPE-30-no concurrently).

## 2026-07-21 — LLaVA POPE-30-yes/no windowed-steering plots regenerated

**Dumps:** `data/pope/dumps/llava-1.5-7b-hf/{pope30_yes_windowed_steering,pope30_no_windowed_steering}/` (64 cells each including in-grid `baseline`).

**Command:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model llava-1.5-7b-hf --datasets pope30_yes pope30_no
```

**Code change:** `plot_pope30_windowed_steering.py` now takes `--datasets`, gold-aware accuracy/flips, in-grid baseline; writes under `plots/<model>/<dataset>/`. Removed prior flat PNGs under `plots/llava-1.5-7b-hf/*.png`.

**Baseline headline (n=30 per condition):**
- pope30_yes neutral: accuracy=0.833 (25/30 yes), mean_p_yes_raw=0.8343
- pope30_yes assertive_toward_no: accuracy=0.767 (23/30 yes), mean_p_yes_raw=0.7206
- pope30_no neutral: accuracy=0.800 (24/30 no), mean_p_yes_raw=0.2555
- pope30_no assertive_toward_yes: accuracy=0.833 (25/30 no), mean_p_yes_raw=0.2635

**Outputs (9 PNGs each):**
- `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/llava-1.5-7b-hf/pope30_yes/`
- `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/llava-1.5-7b-hf/pope30_no/`

`IMPLEMENTATION.md` updated for plotter paths/CLI.

## 2026-07-21 — POPE-30-no mean-probability plots use P(no)

**Change:** `plot_pope30_windowed_steering.py` plots mean `score_p_no_raw` for `pope30_no` (filenames `pope30_mean_p_no_raw_by_layer_window_*.png`); `pope30_yes` still uses mean `score_p_yes_raw`.

**Regenerated (LLaVA only):**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model llava-1.5-7b-hf --datasets pope30_no
```
Removed prior `pope30_no/pope30_mean_p_yes_raw_*.png`.

**Baseline mean P(no):** neutral 0.7322; assertive_toward_yes 0.7337.

## 2026-07-21 — Deleted legacy POPE-30 dump trees

Removed from disk (LLaVA + Qwen):
- `data/pope/dumps/{model}/pope30_windowed_steering/`
- `data/pope/dumps/{model}/pope30_existence_yes_baseline/`

Kept: `pope30_yes_windowed_steering`, `pope30_no_windowed_steering`, `pope_yes_120_baseline`, `pope_no_120_baseline`.
Prior RESEARCH_LOG entries that documented those deleted dumps were removed or scrubbed in the same cleanup; `IMPLEMENTATION.md` updated.

## 2026-07-21 — Cleaned perception_diag logs of legacy POPE-30 runs

Deleted:
- `logs/llava_windowed_steering_stage_a_20260721_104423.log` (tag `pope30_windowed_steering`)
- `logs/qwen_windowed_steering_stage_b_20260721_110008.log` (tag `pope30_windowed_steering`)
- `logs/resume_llava_after_qwen_pope_20260721_135226.log` (watcher for deleted Qwen dump)

Rewrote `amber100_pope30_baseline_dumps_20260719_145123.log` → `amber100_baseline_dumps_20260719_145123.log` (AMBER-100 baseline sections only; `pope30_existence_yes_baseline` sections removed).

Kept yes/no orchestrator + dump logs (`llava_pope30_{yes,no}_*`, `qwen_pope30_{yes,no}_*`, `pope30_yes_no_orchestrator_*`, `qwen_after_gpu0_clear_*`) and AMBER-100 windowed/pause logs.

## 2026-07-22 — Qwen POPE-30-yes/no windowed-steering plots

**Dumps:** `data/pope/dumps/qwen2.5-vl-7b-instruct/{pope30_yes_windowed_steering,pope30_no_windowed_steering}/` (55 cells each including in-grid `baseline`; logs DONE 2026-07-22 ~00:03 / 00:10).

**Command:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model qwen2.5-vl-7b-instruct --datasets pope30_yes pope30_no
```

**Baseline headline (n=30):**
- pope30_yes neutral: accuracy=0.733 (22/30 yes), mean_p_yes_raw=0.7131
- pope30_yes assertive_toward_no: accuracy=0.536 (15/28 parseable yes), mean_p_yes_raw=0.3212
- pope30_no neutral: accuracy=1.000 (30/30 no), mean_p_no_raw=0.9419
- pope30_no assertive_toward_yes: accuracy=0.967 (29/30 no), mean_p_no_raw=0.8454

**Outputs (9 PNGs each):**
- `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/qwen2.5-vl-7b-instruct/pope30_yes/`
- `…/plots/qwen2.5-vl-7b-instruct/pope30_no/`

Removed leftover flat legacy PNGs under `plots/qwen2.5-vl-7b-instruct/*.png`.

## 2026-07-22 — AMBER-100 attribute/relation windowed-steering plots (LLaVA)

**Plan:** `implementation_plans/7-22-26/amber100_attribute_relation_windowed_steering_plots_plan_2026-07-22.md`
**Target:** lambdab2, CPU (plots only).

**Baseline normalize:** rewrote `data/amber/dumps/llava-1.5-7b-hf/amber100_baseline/baseline/manifest.jsonl` from pretty-printed concatenated JSON to true one-object-per-line JSONL (500 records). Backup: `manifest.jsonl.pretty_backup`.

**Command:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model llava-1.5-7b-hf \
  --datasets amber100_attribute_yes amber100_attribute_no amber100_relation_yes amber100_relation_no
```

**Inputs:** steered `data/amber/dumps/llava-1.5-7b-hf/amber100_windowed_steering/` (63 cells); baseline `amber100_baseline/baseline/` filtered to (qtype, gold) subsets (n=20 × 2 conditions). Gate: every subset/cell had exactly 20 items.

**Baseline headline (n=20 per condition):**
- attribute_yes neutral: accuracy=0.800, mean_p_yes_raw=0.7591; assertive_toward_no: accuracy=0.700, mean_p_yes_raw=0.6380
- attribute_no neutral: accuracy=0.700, mean_p_no_raw=0.5936; assertive_toward_yes: accuracy=0.750, mean_p_no_raw=0.6250
- relation_yes neutral: accuracy=0.650, mean_p_yes_raw=0.5385; assertive_toward_no: accuracy=0.100, mean_p_yes_raw=0.3150
- relation_no neutral: accuracy=0.900, mean_p_no_raw=0.6266; assertive_toward_yes: accuracy=0.800, mean_p_no_raw=0.6080

**Outputs (9 PNGs × 4 subsets = 36):**
- `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/llava-1.5-7b-hf/amber100_attribute_yes/`
- `…/amber100_attribute_no/`
- `…/amber100_relation_yes/`
- `…/amber100_relation_no/`

`IMPLEMENTATION.md` updated for AMBER subset keys on the plotter.

## 2026-07-22 — Merged LLaVA×Qwen POPE-30-yes/no joint plots

**Command:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --datasets pope30_yes --joint_mean_p_yes --joint_flips
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --datasets pope30_no --joint_mean_p_yes --joint_flips
```

**Outputs** (`…/windowed_steering_summary/plots/merged_plots/`):
- `pope30_yes_mean_p_yes_raw_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png`
- `pope30_no_mean_p_no_raw_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png`
- `pope30_yes_flips_from_baseline_yes_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png`
- `pope30_no_flips_from_baseline_no_by_layer_window_llava_vs_qwen_additive_vs_rotation_mlp.png`

Axes: rows = model (LLaVA / Qwen); columns = additive @ mlp / rotation @ mlp; condition sub-rows within each pane. New dumps only (`pope30_{yes,no}_windowed_steering`). Added `--joint_flips` to plotter; joint defaults land under `merged_plots/`.

## 2026-07-22 — Flip-from-baseline plot y-axis scale tightened

**Change:** `plot_pope30_windowed_steering.py` — `flips_ylim_max(n_items, observed_max) = max(round(n_items/4), ceil(observed_max))`; applied in `plot_flips_method` and `plot_joint_flips_model_by_mlp`. Integer y ticks use `MaxNLocator(integer=True, nbins=8)`.

**Commands:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model llava-1.5-7b-hf \
  --datasets pope30_yes pope30_no amber100_attribute_yes amber100_attribute_no \
             amber100_relation_yes amber100_relation_no
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --model qwen2.5-vl-7b-instruct --datasets pope30_yes pope30_no
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --datasets pope30_yes --joint_flips
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --datasets pope30_no --joint_flips
```

**Outputs:** all `*flips_from_baseline*` PNGs under `…/windowed_steering_summary/plots/{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}/{dataset}/` and `…/plots/merged_plots/` regenerated. Preferred y-max for n=30 is 8; figures with taller bars expand (e.g. Qwen POPE-30-no rotation @ layer observed max expands above 8).


## 2026-07-22 — LLaVA AMBER-100 merged plots (gold yes vs no × MLP)

**Command:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --joint_amber_llava
```

**Layout:** LLaVA only; rows = gold=yes then gold=no; columns = additive @ mlp | rotation @ mlp; each cell stacks neutral + assertive. Metrics: accuracy, mean P (top P(yes) / bottom P(no)), flips. rotation @ layer omitted.

**Outputs** (`…/windowed_steering_summary/plots/merged_plots/amber-100-runs/`):
- `amber100_attribute_accuracy_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`
- `amber100_attribute_mean_p_yes_vs_p_no_raw_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`
- `amber100_attribute_flips_from_baseline_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`
- `amber100_relation_accuracy_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`
- `amber100_relation_mean_p_yes_vs_p_no_raw_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`
- `amber100_relation_flips_from_baseline_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`

POPE joint default output dir updated to `merged_plots/pope-30-runs/`.


## 2026-07-22 — LLaVA POPE-30 merged flips (gold yes vs no × MLP)

**Command:**
```
MPLBACKEND=Agg python diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py \
  --joint_pope_llava
```

**Output:**
`…/plots/merged_plots/pope-30-runs/pope30_flips_from_baseline_by_layer_window_llava_gold_yes_vs_no_additive_vs_rotation_mlp.png`

Layout matches AMBER LLaVA gold-yes/no merged flips (top gold=yes, bottom gold=no; columns additive|rotation @ mlp; neutral + assertive nested).


## 2026-07-22 — AMBER-100 LLaVA response galleries + summary cleanup

**Command:**
```
python diagnostic_experiments/perception_diag/build_amber100_llava_response_galleries.py
```

**New HTML (under `…/windowed_steering_summary/`):**
- `amber100_relation_gold_yes_leading_toward_no_baseline_response_gallery.html` (6.26 MB) — relation × gold=yes × assertive_toward_no baseline only; header counts: no=18, yes=2 (n=20)
- `amber100_attribute_relation_gold_no_additive_mlp_layers_0_9_response_gallery.html` (11.35 MB) — attribute+relation × gold=no; neutral + assertive_toward_yes; baseline vs additive_mlp_{0.2,0.5,0.9}_layers_0_9

**Deleted from `windowed_steering_summary/` (stale vs deleted dump tags / pre-split POPE):**
- `pope30_leading_toward_no_mlp_beta09_early_and_all_layers_response_gallery.html`
- `qwen_pope30_leading_toward_no_rotation_mlp_layers_5_14_and_18_27_response_gallery.html`
- `qwen_pope_one_example_neutral_vs_leading_toward_no.html`
- `pope30_mlp_2x2_accuracy_and_flips_summary.json`
- `pope30_mlp_2x2_mean_p_yes_raw_summary.json`
- `pope30_yes_reproducibility_llava.json`
- `pope30_yes_reproducibility_qwen.json`

**Kept:** `pope30_{yes,no}_mlp_2x2_*.json`, `windowed_steering_consolidated_results.json`, `plots/`.


## 2026-07-22 — Flip plots: leading-clause baseline bars

**Change:** `plot_pope30_windowed_steering.py` — grey hatched bars on assertive (leading-clause) flip panes now show directed flips from neutral→assertive baseline (`n_leading_clause_flips`); neutral panes remain 0. Colored bars unchanged (flips from same-condition baseline → steered). Legend label: “leading clause alone (vs neutral)”.

**Example LLaVA leading-clause flip counts (neutral→assertive, directed):**
- pope30_yes assertive_toward_no: 2
- pope30_no assertive_toward_yes: 1
- amber100_relation_yes assertive_toward_no: 11
- amber100_attribute_no assertive_toward_yes: 2

**Regenerated:** all per-model `*flips_from_baseline*` PNGs (LLaVA POPE+AMBER, Qwen POPE) and merged flip PNGs under `merged_plots/{pope-30-runs,amber-100-runs}/`.


## 2026-07-22 — Flip plots: remove grey leading-clause bars

**Change:** Removed grey hatched bars from all flip plots (per-method and merged). Flip bars are steering-only: same-condition no-intervention baseline → steered. Leading-clause effect is not overlaid on these plots.

**Regenerated:** all `*flips_from_baseline*` PNGs (LLaVA POPE+AMBER, Qwen POPE, merged pope-30-runs / amber-100-runs).



## 2026-07-22 — Shuffled-control direction sanity checks (all/nd200)

**Plan:** `implementation_plans/7-22-26/shuffled_control_direction_sanity_checks_plan_2026-07-22.md`  
**Target:** lambdab2 `CUDA_VISIBLE_DEVICES=0` (A6000 free). No W&B. No behavioral / cosine comparison.  
**Commit at run time:** `f1aaf63` (local uncommitted: new `shuffled_control.py` + extract CLI).  
**Control for:** `demosv2_9a44f4af_all_nd200_s42_r2_prefix` (`all`, nd200, seed 42, rank 2).

**Code added:**
- `evaluation/interventions/vti/shuffled_control.py`
- `evaluation/run_scripts/extract_shuffled_control_directions.py`

**Derangement (once, model-independent):**
- `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json`
- seed 1234; n_ids=200; fixed points=0

**Commands:**
```
CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/romanus/huggingface \
  python evaluation/run_scripts/extract_shuffled_control_directions.py \
  --model llava-hf/llava-1.5-7b-hf
# log: logs/shuffled_control_llava_2026-07-22.log  (~139s wall)

CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/romanus/huggingface \
  python evaluation/run_scripts/extract_shuffled_control_directions.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct --max_pixels 1003520
# log: logs/shuffled_control_qwen25_2026-07-22.log  (~184s wall)
```

**Artifacts per model** under `experiment_artifacts/vti/{model_short}/shuffled_control/`:
- `all_nd200/directions.npz` + `metadata.json` + `components.npz`
- `_act_cache/` (400 `sample_*.npz` files each)
- `shuffled_control_sanity_report_{model_short}.md`

| Model | model_short | Direction shape | Forwards executed | Gating |
|---|---|---|---|---|
| LLaVA-1.5-7B | `llava-1.5-7b-hf` | (32, 4096) | 400 | all PASS |
| Qwen2.5-VL-7B | `qwen2.5-vl-7b-instruct` | (28, 3584) | 400 | all PASS |

**Gating primitives (both cells):** fixed points 0; captions unchanged 200/200; positional id mismatches 0; forwards 400/400; norms finite. Per-layer L2 tables are in the sanity reports (non-gating well-formedness readout only).

**Deviations from plan:** none on construction. Module path used is `evaluation/interventions/vti/directions_v2.py` (not `src/directions/…`); Qwen `model_short` filled as `qwen2.5-vl-7b-instruct`; Qwen extract used `--max_pixels 1003520` to match the deployed demos_v2 extract path.


## 2026-07-22 — Geometric comparison: deployed vs shuffled-control (all/nd200)

**Plan:** `implementation_plans/7-22-26/geometric_comparison_deployed_vs_shuffled_control_plan_2026-07-22.md`  
**Target:** lambdab2 CPU-only (no GPU, no model load, no W&B, no behavioral runs).  
**Commit at run time:** `f1aaf63` (local uncommitted: comparison script + outputs under `diagnostic_experiments/perception_diag/control/geometric_comparison/`).  
**Depends on:** shuffled-control sanity checks (both cells PASS; RESEARCH_LOG entry above).

**Code added:**
- `diagnostic_experiments/perception_diag/control/geometric_comparison/compare_deployed_vs_shuffled_control.py`

**Command:**
```
MPLCONFIGDIR=/tmp/matplotlib \
  python diagnostic_experiments/perception_diag/control/geometric_comparison/compare_deployed_vs_shuffled_control.py
```

**Inputs per model:**
- Deployed: `experiment_artifacts/vti/{model_short}/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/`
- Shuffled-control: `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/`
- Loader: `load_textual_v2_directions`; PC1 from `components.npz` key `pc0` decoder rows `[1:]`

**Preconditions:**
- Shape match PASS: LLaVA `(32, 4096)`; Qwen `(28, 3584)`.
- PC1/direction fraction (decoder): LLaVA max deployed 0.0896 / shuffled 0.0935; Qwen max deployed 0.00810 / shuffled 0.0475. Cosine-plot open markers at/above 90th percentile of `max(PC1/direction)` (LLaVA threshold 0.0925, 4 layers; Qwen threshold 0.0448, 3 layers).

**Outputs** under `diagnostic_experiments/perception_diag/control/geometric_comparison/`:
- `geometric_comparison_{model_short}.csv`
- `cosine_deployed_vs_shuffled_control_by_layer_{model_short}.png`
- `magnitude_deployed_vs_shuffled_control_by_layer_{model_short}.png`
- `geometric_comparison_summary.md`

| Model | Cosine min / median / max | Notes (facts) |
|---|---|---|
| LLaVA-1.5-7B | 0.821 / 0.948 / 0.999 | Lowest at decoder layer 15 (0.821); early layers ≈1.0 |
| Qwen2.5-VL-7B | 0.329 / 0.922 / 0.998 | Layers 15–27 cosines: 0.866, 0.804, 0.701, 0.590, 0.584, 0.539, 0.514, 0.465, 0.433, 0.382, 0.399, 0.419, 0.329. At layer 26 L2 norms: deployed 65.53 vs shuffled 11.83; at layer 27: 36.31 vs 5.70 |

**Deviations from plan:** none. CSV columns match the plan; PC1 fraction also recorded in the summary table (not a CSV column). Annotation uses ≥90th-percentile of max PC1/direction fraction rather than a fixed mean-dominance cutoff (plan left threshold to implementer).


## 2026-07-28 — demos_850 disjoint-partition extraction: code + gates (mining pending)

**Plan:** `implementation_plans/7-28-26/demos_850_disjoint_partition_activation_and_direction_extraction_plan_2026-07-28.md`  
**Target:** lambdab2 (CPU mining by Alex; GPU check 0.4 on device 0).  
**Status:** implementation complete through step 0 / extract scaffolding; steps 1–5 (mining) not run by Cursor; steps 6–9 wait on ≥295 new stage-4 passes.

**Code / path changes:**
- `data_scripts/vti_demos_v2/stage0_mine_candidates.py`: `--summary-out` (default unchanged)
- `data_scripts/vti_demos_v2/assemble_demos_850.py` (+ `--write-partition`)
- `evaluation/interventions/vti/directions_partition.py`
- `evaluation/run_scripts/extract_demos850_partition_directions.py`
- `helper_scripts/verify_demos850_partition_extraction.py`
- `src/paths.py`: `vti_demos_850_path()`, `vti_demos_850_partition_path()`
- `.gitignore`: un-ignore `demos_850.jsonl`, `demos_850_partition_s42.json`
- `IMPLEMENTATION.md`: demos_850 subsection

**Check 0.1:** `demos_v2.jsonl` n=555 unique=555; all five `h_values` non-empty; question fixed. sha256[:16]=`9a44f4afde0324b5`.

**Check 0.3 / P2 snapshot:** `data/vti/v2/_summaries_snapshot_555_2026-07-28/` (seven `stage*_summary.json` + `pre_topup_checksums.txt` including stage jsonl, `demos_v2.jsonl`, `demos_v2_order_s42.json`, and 40 `demosv2_9a44f4af_*/directions.npz` hashes).

**Check 0.4 (cache fidelity), GPU 0, 3 ids × variants `value`/`all`, thresholds cosine≥0.9999 and max|Δ|/‖cached‖₂≤1e-3:**

| Model | Command / log | pass |
|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | `logs/demos850_cache_fidelity_llava_2026-07-28.log`; report `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_cache_fidelity_llava-1.5-7b-hf_2026-07-28.json` | True |
| `Qwen/Qwen2.5-VL-7B-Instruct` (`max_pixels=1003520`) | `logs/demos850_cache_fidelity_qwen25_2026-07-28.log`; report `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_cache_fidelity_qwen2.5-vl-7b-instruct_2026-07-28.json` | True |

Shared `_act_cache/` may be extended after assembly (350 new ids × 6 variants × 2 models). Mining / assembly / direction extraction not yet run.


## 2026-07-29 — demos_850 shuffled-control extract launched (concurrent GPU 0)

**Plan:** `implementation_plans/7-29-26/demos_850_shuffled_control_per_partition_block_plan_2026-07-29.md`  
**Spec:** `extractions/shuffle_control_vector_diff_sample_size_07_29_26_extraction.md`  
**Target:** lambdab2 GPU 0; LLaVA and Qwen run concurrently (Alex override vs plan’s sequential default).

**Code:**
- `evaluation/interventions/vti/shuffled_control_partition.py`
- `evaluation/run_scripts/extract_demos850_shuffled_control_directions.py`

**Derangements (CPU, seed 1234):**
- `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd{50,100,200,500}_s1234.json`

**Commands:**
```
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation/run_scripts/extract_demos850_shuffled_control_directions.py \
  --model llava-hf/llava-1.5-7b-hf --num_demos 50 100 200 500 --derangement_seed 1234 \
  > logs/demos850_shuffled_control_llava_2026-07-29.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u evaluation/run_scripts/extract_demos850_shuffled_control_directions.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct --num_demos 50 100 200 500 --derangement_seed 1234 --max_pixels 1003520 \
  > logs/demos850_shuffled_control_qwen25_2026-07-29.log 2>&1 &
```

**At launch:** GPU 0 ~30792 MiB used (LLaVA ~14110 + Qwen ~16670); both on `all/nd=50`.  
Gate/pass counts and final act-cache file totals to be appended when both jobs exit.


## 2026-07-29 — per-layer PCA vs global PCA (demos850 deployed vs shuffled-image control)

**Plan:** `implementation_plans/7-29-26/per_layer_pca_deployed_vs_control_cosine_plan_2026-07-29.md`  
**Design:** `designs/perlayer_pca_control_07_29_26.md`  
**Commit:** `2c3dd46` (working tree may have uncommitted implementation)  
**Target:** lambdab2 CPU only; zero forward passes; no GPU; no model load.

**Code landed:**
- `evaluation/interventions/vti/directions_partition.py` — `fit_locus` on `partition_slug` (default `"global"`)
- `evaluation/interventions/vti/directions_v2.py` — same field on `textual_v2_slug`
- `evaluation/interventions/vti/perlayer_pca.py`
- `evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py`
- `helper_scripts/verify_perlayer_pca_control_extraction.py`
- `diagnostic_experiments/perlayer_pca_control/compare_perlayer_vs_global_pca_deployed_vs_control.py`

**Commands:**
```
python helper_scripts/verify_perlayer_pca_control_extraction.py --mode record
python evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py \
  --models llava-1.5-7b-hf qwen2.5-vl-7b-instruct \
  --arms deployed control --num_demos 50 100 200 500
python helper_scripts/verify_perlayer_pca_control_extraction.py --mode verify
python diagnostic_experiments/perlayer_pca_control/compare_perlayer_vs_global_pca_deployed_vs_control.py \
  --models llava-1.5-7b-hf qwen2.5-vl-7b-instruct --num_demos 50 100 200 500
```

**Gating (Checks 1–7):** all PASS. Global-fit SHA-256 byte-identical pre vs post.
Act-cache `(name, size, mtime_ns)` triples unchanged (5100 + 1700 per model).
Mean identity 16/16 cells (`max_abs_diff` within `1e-6 * max|global_mean|`).
Per-layer shapes `(32, 4096)` / `(28, 3584)`; `forwards_executed=0`;
`act_cache_read_only=true`.

**Deviation from plan — Check 4:** plan required max-abs > 0 at every layer row
including embedding row 0. Observed: row 0 is identical (`emb0_abs=0`) for all
5 probed ids × both models (last-token input embedding is caption-only under
image derangement); every decoder row differs and stack max-abs > 0. Gate
narrowed to decoder rows 1..L; embedding-row abs reported in the check detail.

**New direction cells (16):**
- `experiment_artifacts/vti/{model}/textual_v2_perlayer/demos850_ba05bd96_all_nd{N}_s42_r2_partition_perlayer/`
- `experiment_artifacts/vti/{model}/shuffled_control_demos850_perlayer/all_nd{N}_perlayer/`
for `model ∈ {llava-1.5-7b-hf, qwen2.5-vl-7b-instruct}`, `N ∈ {50,100,200,500}`.

**Analysis out:** `diagnostic_experiments/perlayer_pca_control/`
(CSVs, 2×2 facet PNGs, `perlayer_vs_global_pca_control_summary.md`,
`perlayer_pca_extraction_manifest_2026-07-29.json`).

**Headline band summaries (median / mean cosine, deployed vs control):**

| Model | Scheme | N | Band | Median | Mean |
|---|---|---:|---|---:|---:|
| LLaVA | global | 200 | 0-10 | 0.9960 | 0.9917 |
| LLaVA | global | 200 | 18-31 | 0.8878 | 0.8851 |
| LLaVA | perlayer | 200 | 0-10 | 0.9961 | 0.8257 |
| LLaVA | perlayer | 200 | 18-31 | 0.9291 | 0.8068 |
| Qwen | global | 200 | 0-10 | 0.9595 | 0.9701 |
| Qwen | global | 200 | 18-27 | 0.2613 | 0.3039 |
| Qwen | perlayer | 200 | 0-10 | 0.9766 | 0.6717 |
| Qwen | perlayer | 200 | 18-27 | 0.3267 | 0.3814 |

Full tables (all N) in
`cosine_by_layer_band_and_sample_size_{model}.csv` and the summary md.

**Check 8 (PC1 share of direction norm, deployed nd200):**
- LLaVA global: min/median/max = 0.0283 / 0.0743 / 0.1003; descending steps = 1
- LLaVA perlayer: 0.2025 / 0.7529 / 1.1496; descending steps = 8
- Qwen global: 0.00181 / 0.00237 / 0.00792; descending steps = 2
- Qwen perlayer: 0.0149 / 0.4292 / 1.1327; descending steps = 3

(Share can exceed 1 when `||PC1|| / ||PC1+mean||` and PC1 vs mean are
anti-aligned.) Full Check 8/9 rows in the extraction manifest.

**Wall clock:** extract ~117 s; verify ~4 s; analysis ~9 s.


## 2026-07-29 — perlayer_pca_control directory reorganized

Moved flat outputs under `diagnostic_experiments/perlayer_pca_control/` into:
`scripts/`, `verification/`, `plots/`, `tables/`, `summaries/`.
Updated path defaults in the two analysis scripts and
`helper_scripts/verify_perlayer_pca_control_extraction.py` (`OUT_DIR` →
`…/verification/`). No direction artifacts or plot contents regenerated;
files were relocated only.

## 2026-07-30 — steering visual reasoning validation: implementation + overnight launch

Plan: `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md`

**Code landed (no grid metrics yet):**
- `evaluation/interventions/vti/directions_meandiff.py` — CPU raw mean-difference over demos_850 partitions; 0 forwards; raises on act-cache miss
- `evaluation/run_scripts/extract_demos850_meandiff_directions.py`
- `helper_scripts/verify_demos850_meandiff_extraction.py`
- `VTITextualIntervention` / `run_evaluation` / `run_eval.py`: `--directions_dir`, `--layer_set`; meandiff result-dir suffix `__meandiff__layers_{label}`
- `step0_chair_token_cap.py`: `--max_pixels`, model-suffixed output, truncation readouts
- `--chair_max_new_tokens` CLI default **256** (`run_eval.py`); `run_exp1_repro_grid.sh` / `readme.md` aligned
- Driver: `evaluation/run_scripts/run_steering_visual_reasoning_validation.sh` (AMBER → CHAIR → POPE)
- Orchestrator: `evaluation/run_scripts/launch_steering_visual_reasoning_overnight.sh`
- Analysis: `evaluation/steering_visual_reasoning_validation/{build_result_tables,make_plots}.py`

**Overnight CHAIR-cap policy (Alex):** if the 256/512 probe reports any caption at the 256 token cap on either model, set `CHAIR_CAP=512` and continue; never halt; never skip CHAIR. Decision written to `evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/chair_cap_probe_decision.json`.

**Launch:** orchestrator under nohup → logs/steering_visual_reasoning_overnight_orchestrator_2026-07-30.log; grid logs `logs/steering_visual_reasoning_validation_{llava,qwen25}_2026-07-30.log` after probe. Target: lambdab2 GPU 0, both models concurrent after 300s stagger.

**Status at this entry:** implementation complete; orchestrator start / probe / grid PIDs to be confirmed from status JSON after launch.

## 2026-07-30 — overnight babysitter + CHAIR_CAP auto-bump to 512

**Probe decision:** Qwen step0 at 256 had `n_captions_at_token_cap=6` (and 6 without terminal punctuation); at 512 that count was 0. LLaVA had 0 at both caps. Per overnight policy, orchestrator set `CHAIR_CAP=512` for the full grid (both models). Decision file: `evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/chair_cap_probe_decision.json`.

**Babysitter:** `evaluation/run_scripts/babysit_steering_visual_reasoning_grids.py` — monitors both driver PIDs; on crash (exit before finished banner) relaunches the same driver so `--skip_if_exists` resumes. If the crash log shows CUDA/OOM and the sibling model is still running, waits for the sibling to finish before restarting (avoids immediate re-OOM under concurrent footprint). Max 8 restarts/model. Status: `…/babysitter_status.json`. Log: `logs/steering_visual_reasoning_babysitter_2026-07-30.log`. Future overnight launches start the babysitter from the orchestrator after both grids are up.

## 2026-07-31 morning — pause LLaVA on lambdab2; prepare RunAI H100 resume

**Lambdab2:** Stopped both babysitters and the LLaVA driver so Qwen runs exclusive on GPU 0 (~16.6 GiB). LLaVA AMBER complete (37/37) locally; CHAIR mid-grid paused (checkpoint + 2 completed CHAIR cells including baseline). Marker: `evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/llava_paused_for_runai.json`.

**CHAIR_CAP** remains 512 for any resume (probe auto-bump).

**RunAI pack (not yet uploaded):** `/tmp/runai_steering_llava_2026-07-30/` — `vti_repo_working_tree.tar.gz` (includes uncommitted meandiff / `--directions_dir` code), `llava_meandiff_directions.tar.gz`, `llava_partial_results_chair.tar.gz`, `SUBMIT.md`. Helpers: `helper_scripts/runai/run_steering_visual_reasoning_llava.sh`, `pack_steering_llava_for_runai.sh`.

**Blockers for submit:** (1) `runai login` token expired on lambdab2; (2) SFTP to NFS from lambdab2 denied — upload via WinSCP ThinkPad still required.

## 2026-08-06 — matched LLaVA POPE comparison (2026-06-19 vs 2026-07-30)

**Type:** offline analysis of existing eval cells (no new generation).
**Script:** `evaluation/pope_0619_vs_0730_matched/build_comparison.py`
**Output:** `evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/`

**Frame:** LLaVA-1.5-7B, `vti_textual_additive_mlp`, all layers, β ∈ {0.2, 0.5, 0.9}, POPE random/popular/adversarial (n=200 each). Baselines identical across the two run dates.

| Arm | Source cells | Direction |
|-----|--------------|-----------|
| 06-19 | `evaluation/results/2026-06-19/llava-1.5-7b-hf/pope_*/vti_textual_additive_mlp__b{β}/` | author demos nd70, PC1+mean |
| 07-30 | `…/2026-07-30/…/vti_textual_additive_mlp__b{β}__dall__nd500__meandiff__layers_all/` | demos850 nd500, raw mean-difference |

**Artifacts:** `comparison_tables.md`; `tables/{per_split,split_averaged}_metrics.csv`; bar plots under `plots/split_averaged/` and `plots/per_split/{random,popular,adversarial}/` for accuracy, precision, recall, accuracy_gold_no, accuracy_gold_yes; `plots/direction_magnitude_per_layer.png` (+ abs-diff twin) with demos850 nd500 PC1+mean (`*_r2_partition`) as magnitude control; `cells.json`, `direction_magnitudes.json`.

**Split-averaged accuracy (%):** baseline 84.67; β=0.2 → 06-19 83.67 / 07-30 84.67; β=0.5 → 85.50 / 83.33; β=0.9 → 88.00 / 83.17.

**Update (same day):** added three pairwise per-layer cosine plots under `plots/direction_cosine/` plus `direction_cosines.json`. Mean cosine over layers: 06-19 vs 07-30 meandiff −0.0425; 06-19 vs demos850 PC1+mean −0.0386; 07-30 meandiff vs demos850 PC1+mean (same block) 0.9989.
