# Layer-Windowed Steering: Yes/No Probability Sweep (v2)

Date: 2026-07-21
Status: ready for implementation (approved by Alex in chat; sanity gates explicitly waived — exploratory sweep, degenerate outputs are informative and must be recorded, not filtered)

Supersedes: `layer_windowed_steering_yes_no_probability_plan_2026-07-21.md` (v1). Implement from THIS file.
Changes from v1: LLaVA and Qwen run in parallel on the **same** free A6000 (GPU 0); within each model, POPE-30 before AMBER-100 (smallest-first); aggregator supports partial builds (LLaVA-only, Qwen-only, or mid-AMBER). Cells, metrics, data, and artifacts are unchanged.

**v2 amendment (2026-07-21, Alex):** Stage B is **not** gated on Stage A. Colocate both 7B models on one A6000 (~14–16 GB each; ~48 GB card has headroom). Each process uses `CUDA_VISIBLE_DEVICES=<same>` + `device_map cuda:0`, unsharded.

## Exploratory question

How does the effect of VTI-style textual steering on first-token yes/no probabilities and generated responses depend on which decoder-layer band the steering hooks are applied in — per steering config, per strength, per model, under a neutral prompt vs. a gold-opposing leading prompt?

No gating hypothesis. This is a descriptive sweep; any outcome (flat across bands, band-localized effect, windowed rotation@layer avoiding the known full-depth collapse) is a legitimate result.

## What exists vs. what is new

Existing (verified against source on 2026-07-21):

- `vti_hook_ctx` (`evaluation/interventions/vti/hooks.py`) currently registers steer hooks on ALL decoder layers (`range(n_layers)`).
- Dump driver `diagnostic_experiments/perception_diag/run_dump.py`: static `CELL_SPECS` dicts `{method, site, strength, readout}`, resume via `DumpWriter.is_done(item_id, condition_id)`, per-record OOM/error status handling, `--num_demos 200` default (matches the demos_v2 `all`@nd200 direction convention, slug `demosv2_9a44f4af_all_nd200_s42_r2_prefix`).
- Scoring under steering needs NO new plumbing: `run_capture_generation` computes `score_yes_no_logits` (first-token yes/no logits/probs at the last prefill position) INSIDE `optional_steer_ctx`, so scores reflect the steered forward.
- Pins and prompt files: `data/amber/pinned_amber_disc_100.json`, `data/pope/pinned_pope_existence_yes_30.json`; augmented prompt JSONLs `data/amber/augmented_amber100.jsonl`, `data/pope/augmented_pope30.jsonl` (rows: `item_id`, `image_path`, `gold`, `qtype`, `variants[{condition_id, template_id, prompt}]`; conditions include `neutral`, `assertive_toward_yes`, `assertive_toward_no`).
- Baseline dumps to REUSE, not re-run: `data/amber/dumps/{model_short}/amber100_baseline/` and `data/pope/dumps/{model_short}/pope30_existence_yes_baseline/` for both models (max_new_tokens=128, all 5 conditions present).
- Window helper: `layer_windows(num_layers, width=10, stride=5)` in `src/prompt_spans.py` yields exactly the requested bands.

New code (all flagged as net-new):

1. `layer_indices: Optional[Sequence[int]] = None` parameter on `vti_hook_ctx`. Default `None` preserves current behavior (all layers). When provided, register hooks only on those decoder-layer indices. `directions` stays full-length `(num_layers, hidden_dim)` and is indexed by absolute layer index; keep the existing `len(directions) == n_layers` check unchanged.
2. `layer_indices` key in dump cell specs, forwarded by `optional_steer_ctx` (`capture/steered_capture.py`) into `vti_hook_ctx`.
3. Windowed-grid cell generation in `run_dump.py` (new flag, e.g. `--windowed_grid`): after model load, build cell specs programmatically from the 3 configs x 3 strengths x (`layer_windows(wrapper.num_layers)` + one all-layers entry) instead of hand-listing static entries. Record the generated grid in `run_metadata.json`.
4. Gold-conditional condition filter in `run_dump.py` (new flag, e.g. `--conditions gold_conditional`): for each item keep exactly two variants — `neutral`, plus `assertive_toward_yes` if `item["gold"] == "no"` else `assertive_toward_no`. This is a filter of the existing augmented JSONLs; no new data files are built. Record the filter in `run_metadata.json`.
5. Aggregator script for the consolidated human-readable JSON (see Artifacts). Must support PARTIAL builds: it aggregates whatever manifests exist on disk at build time (e.g. LLaVA-only, or LLaVA with AMBER cells still in progress — completed cells' manifests are valid mid-run), and records what was included under a top-level `_completeness` field (models, run tags, cell dirs found, cell dirs expected-but-missing). Re-running the aggregator later regenerates the file over the fuller tree.
6. Unit test for (1), zero GPU: assert that with `layer_indices` set, hooks are registered on exactly the requested layer set (both hook sites), and on no other layer. Included because this parameter is net-new code being written in this plan — it is a code-correctness test inside the implementation, not a pre-run sanity experiment. Add to `tests/`.
7. Minor: if `--verify_g1` is ever used with a windowed cell, the G1 probe in `verify_steered_capture_equality` must check the first layer INSIDE the window, not layer 0 (layer 0 may be unsteered). G1 is not required for this run; fine to skip entirely.

Index convention (stated to prevent off-by-one): window indices are decoder-layer indices `0..num_layers-1`, ranges INCLUSIVE at both ends, matching `directions[layer_idx]` and `dispatch.get_mlp/get_layer(wrapper, layer_idx)`. They are NOT rows of the `(num_layers+1)`-row activation stack (row 0 there is the embedding). "All layers" = `0..num_layers-1`.

## Models and windows

| HF id | model_short | decoder layers | windows (inclusive) |
|---|---|---|---|
| `llava-hf/llava-1.5-7b-hf` | `llava-1.5-7b-hf` | 32 | 0-9, 5-14, 10-19, 15-24, 20-29, 22-31, all |
| `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen2.5-vl-7b-instruct` | 28 | 0-9, 5-14, 10-19, 15-24, 18-27, all |

Qwen: `max_pixels=1003520` (dump default). Both models load unsharded on a single GPU (device policy: hooks are incompatible with sharding).

## Steering configs

Per Alex's spec, three configs:

- `additive` @ `mlp`
- `uniform_rotation` @ `mlp`
- `uniform_rotation` @ `layer`

Strengths (beta): 0.2, 0.5, 0.9. `eps_coeff` 0.1 (default). Direction: demos_v2 `all` @ nd200 (run_dump default `--num_demos 200`), same direction convention for both models from their respective `textual_v2` caches.

Known fact, recorded not gated: `uniform_rotation` @ `layer` at beta 0.9 full-depth collapses LLaVA generations to empty. Empty/degenerate responses are expected in some cells and are informative — record them (`degeneracy_flag`, `truncated`, empty strings); do not filter, do not abort.

## Cells

Per model: 3 configs x 3 betas x (7 | 6) windows = 63 (LLaVA) / 54 (Qwen). 117 steered cells total, each run on both benchmarks. Baselines are NOT re-run.

Cell/directory naming (self-describing, extends the existing dump convention):
`{rotation_mlp|rotation_layer|additive_mlp}_{beta}_layers_{start}_{end}` and `..._layers_all`.
Examples: `rotation_mlp_0.5_layers_10_19`, `rotation_layer_0.9_layers_all`, `additive_mlp_0.2_layers_22_31`.

## Data and prompt conditions

- AMBER-100 via `data/amber/augmented_amber100.jsonl` (pin `pinned_amber_disc_100.json`): 20 per stratum — existence x no, attribute x {yes,no}, relation x {yes,no}. Expected gold split after filtering: 60 no / 40 yes (cheap inline assert while filtering; log a warning on mismatch, do not abort).
- POPE-30 via `data/pope/augmented_pope30.jsonl` (pin `pinned_pope_existence_yes_30.json`): all 30 gold=yes, so every POPE item runs `neutral` + `assertive_toward_no`.
- Conditions per item: exactly two (see new code item 4).
- Records per steered cell: AMBER 200, POPE 60. Totals: LLaVA 63 x 260 = 16,380; Qwen 54 x 260 = 14,040; grand total 30,420 records.
- `max_new_tokens=128` — must match the existing baseline dumps for comparability. Greedy decoding (wrapper default).

## Metrics (primitives, per record — all already emitted by `run_capture_generation`)

- raw response string
- `parsed_outcome` (yes / no / unparseable via `_normalize_yes_no`)
- first-token scores at the last prefill position: `p_yes_raw`, `p_no_raw` (softmax over the full vocab, summed over yes/no token variants), `yes_logit_sum`, `no_logit_sum`, `logit_margin_yes_minus_no`, `answer_mass`, `p_yes_norm`, `first_token_pred`, `prefill_seq_len`
- `degeneracy_flag`, `truncated`, `status`

Primary readouts for this experiment: `p_yes_raw`, `p_no_raw`, and the raw response. `p_yes_norm` and `logit_margin_yes_minus_no` are composites of the raw fields and are carried only because the dump schema already emits them.

No aggregate metrics or plots in this plan — analysis happens in chat after the consolidated JSON is in hand.

## Execution order (LLaVA + Qwen colocated on one A6000)

Priority: LLaVA POPE results today still matter for the same-day aggregator handoff, but Qwen runs **concurrently** on the same free A6000 (GPU 0). Two independent `nohup` drivers; do not serialize Stage B behind Stage A.

Shared GPU notes: both processes set `CUDA_VISIBLE_DEVICES` to the same free index and `--device_map cuda:0`. Expect ~14–20 GB resident each; leave headroom for capture/generation spikes. If either OOMs, record `status=oom` per dump writer and resume after freeing memory — do not abort the other model's run.

Stage A — LLaVA-1.5 (`CUDA_VISIBLE_DEVICES=<free>`, unsharded, `nohup` + log):

```
# A1 — POPE-30 first (63 cells x 60 records = 3,780)
python diagnostic_experiments/perception_diag/run_dump.py \
  --model llava-hf/llava-1.5-7b-hf \
  --augmented_jsonl data/pope/augmented_pope30.jsonl \
  --windowed_grid --conditions gold_conditional \
  --run_tag pope30_windowed_steering \
  --device_map cuda:0

# A2 — AMBER-100 immediately after A1 (63 cells x 200 = 12,600)
python diagnostic_experiments/perception_diag/run_dump.py \
  --model llava-hf/llava-1.5-7b-hf \
  --augmented_jsonl data/amber/augmented_amber100.jsonl \
  --windowed_grid --conditions gold_conditional \
  --run_tag amber100_windowed_steering \
  --device_map cuda:0
```

Stage B — Qwen2.5-VL, **launched in parallel** with Stage A on the **same** GPU:

```
# B1 — POPE-30 (54 cells x 60 = 3,240), then B2 — AMBER-100 (54 cells x 200 = 10,800)
python diagnostic_experiments/perception_diag/run_dump.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --augmented_jsonl data/pope/augmented_pope30.jsonl \
  --windowed_grid --conditions gold_conditional \
  --run_tag pope30_windowed_steering \
  --max_pixels 1003520 \
  --device_map cuda:0
# then same for amber100_windowed_steering
```

Aggregator builds (partial builds are the point — see new code item 5):

- Build 1: after A1 completes — LLaVA POPE-30 + baselines (same-day handoff; Qwen cells included if present).
- Build 2 (optional): mid-AMBER over completed cells only.
- Final build: after both Stage A and Stage B finish — both models in one consolidated file.

Dumps land at `data/{amber|pope}/dumps/{model_short}/{run_tag}/{cell_dir}/` with the standard `manifest.jsonl`, `acts/`, `norms/`, `metadata.json`, plus run-level `run_metadata.json` extended with the generated window grid and the condition filter. Resume-by-(item, condition) via `DumpWriter.is_done` as today. These run tags are new and separate from the existing `amber100_baseline` / `pope30_existence_yes_baseline` trees — do not merge manifests.

Acts/norms capture stays on (default dump behavior; roughly 10 GB fp16 total across all cells on `/data`) so per-layer geometry follow-ups need no rerun.

## Consolidated human-readable JSON (requested artifact)

New aggregator `diagnostic_experiments/perception_diag/build_windowed_steering_summary.py` writing ONE file:

`diagnostic_experiments/perception_diag/windowed_steering_summary/windowed_steering_consolidated_results.json`

Grouped for human reading: model -> benchmark -> item (`item_id`, `qtype`, `gold`, `image_path`, neutral question text) -> list of run records, one per (condition x config x beta x window), each carrying: `condition_id`, full `prompt`, cell name plus `{method, site, beta, layers}`, `response`, `parsed_outcome`, `first_token_pred`, `p_yes_raw`, `p_no_raw`, `p_yes_norm`, `logit_margin_yes_minus_no`, `degeneracy_flag`, `truncated`, `status`.

Partial-build behavior: aggregates whatever manifests exist at build time; `_completeness` field records models / run tags / cell dirs included and expected-but-missing (see new code item 5). The same output path is overwritten on each rebuild over the fuller tree.

Baseline rows: pulled per item from the EXISTING baseline dumps, filtered to the same two conditions (neutral + the gold-opposing assertive condition), marked with cell name `baseline`. If the baseline manifest's prompt string for an item/condition does not exactly match the new runs' prompt, include the row anyway and record the mismatch under a top-level `_warnings` list (do not silently drop or silently include).

Size note: ~30.9k run records including baselines at full completion; expect a file in the tens of MB. Acts/norms are not embedded — manifest fields only.

## Runtime and routing

- Target: lambdab2, **one** free A6000 hosting **both** LLaVA and Qwen processes in parallel (check `nvidia-smi`; prefer an empty card, else colocate on the card already running Stage A if ≥~30 GB free). Unsharded, dual `nohup` logs; fully resumable.
- Sample scale: 30,420 records total (Stage A: 16,380; Stage B: 14,040). Purpose: exploratory sweep on pinned small subsets.
- Wall-clock: colocated runs share SMs so each is slower than solo; expect A1+B1 to finish same day, A2+B2 into the night / next morning. Aggregator Build 1 (LLaVA POPE) remains the same-day deliverable.
- RunAI fallback only if no A6000 has enough free VRAM for both residents — flag first: NFS tarball would need a re-sync with the new code.

## What the sweep will reveal / how it will be read

Per (model, benchmark, config, beta): `p_yes_raw` as a function of window position, neutral vs. leading condition, split by gold. Flat curves indicate band-insensitive steering; band-localized movement gives candidate layer bands for the layer-contribution track; windowed rotation@layer at 0.9 completing without collapse would localize the known full-depth collapse to specific bands. Purely descriptive; follow-up designs go in new plan files.

## Open questions

1. ~~Qwen start timing~~ — **resolved (Alex 2026-07-21):** colocate Stage B with Stage A on the same A6000 (GPU 0).
2. The consolidated JSON is a single file in the tens of MB per Alex's spec. If viewing tooling chokes, the aggregator can additionally emit per-model files with the same schema — not done unless asked.
3. Baseline prompt-string mismatches — aggregator reports them in `_warnings`; whether any appear determines if baseline rows are directly comparable or need a small baseline re-run (not scheduled here).
4. Colocation OOM risk: if dual-resident peaks exceed 48 GB, record per-item `oom` and/or pause one stream; do not kill the other model's completed cells.
