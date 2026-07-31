# demos_850 shuffled-image control directions (per partition block)

extraction_spec: extractions/shuffle_control_vector_diff_sample_size_07_29_26_extraction.md

Date: 2026-07-29
Status: to be implemented and run by Cursor.
Kind: extraction. Produces control primitives + gating sanity reports only — no comparison,
no number read as evidence.

Alex decisions (2026-07-29): dimension **`all` only**; geometric comparison **deferred** to a
follow-up design after sanity PASS; derangement seed **1234** (same as demos_v2 nd200 control),
**one independent derangement per block size**.

## Extraction spec reference

`extractions/shuffle_control_vector_diff_sample_size_07_29_26_extraction.md`

- Field 1: for each of the demos_850 disjoint blocks of 50 / 100 / 200 / 500, and for both
  LLaVA-1.5-7B and Qwen2.5-VL-7B, produce a control steering direction built like the matching
  deployed partition direction, except each forward uses an image from another item **in the same
  block** (captions unchanged). Pairing: same (id, value, h_values.all) as the deployed cell;
  image path overridden by derangement within the block.
- Field 2: the four block sets remain **pairwise disjoint** (inherited from
  `demos_850_partition_s42.json`). Within each block, the control set is **identical** in item
  identity and captions to the deployed `all`/nd{N} cell; only the image–caption binding is
  destroyed. Held constant across blocks: pool, caption construction, models, dimension=`all`,
  derangement seed, extractor math. Varies: block size / item identity (disjoint).
- Field 3: control directions must be computable so a later design can compare them (e.g. cosine)
  to the matching deployed demos850 partition directions. That comparison is **not** this plan.

## Purpose

Construct-and-verify image-deranged controls for the eight deployed cells
`demos850_ba05bd96_all_nd{N}_s42_r2_partition` (N ∈ {50,100,200,500} × 2 models). Same gating
primitives as
`implementation_plans/7-22-26/shuffled_control_direction_sanity_checks_plan_2026-07-22.md`.
No cosine, no magnitude comparison as a result, no behavioral runs, no `run_eval.py` wiring.

## Additivity

Leave untouched:

- `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json`
- `experiment_artifacts/vti/{model_short}/shuffled_control/all_nd200/`
- `experiment_artifacts/vti/{model_short}/shuffled_control/_act_cache/`
- all `textual_v2/demos850_*` and `textual_v2/demosv2_*` deployed directions
- `data/vti/demos_850.jsonl`, `demos_850_partition_s42.json`, `demos_v2.jsonl`

New namespace only: `shuffled_control_demos850/`.

Because `ActivationCache` keys omit the image path, pointing a shuffled run at
`textual_v2/_act_cache` or at the legacy demos_v2 `shuffled_control/_act_cache` would silently
reuse wrong-image activations. The demos850 shuffled run **must** use a fresh cache namespace.

## Grounded facts (verified 2026-07-29)

| Fact | Value |
|---|---|
| Pool | `data/vti/demos_850.jsonl`, sha256[:16] `ba05bd960cad0c18` |
| Partition | `data/vti/demos_850_partition_s42.json` — disjoint 50/100/200/500, seed 42 |
| Deployed `all` cells | Present both models; each `ids_used` equals partition `blocks[str(N)]` in order |
| Legacy module | `evaluation/interventions/vti/shuffled_control.py` — hardcoded demos_v2 `all`/nd200 |
| Derangement seed (7-22) | **1234** — reuse |

## Cells

| model_short | HF id | N | Deployed slug (control-for) |
|---|---|---|---|
| `llava-1.5-7b-hf` | `llava-hf/llava-1.5-7b-hf` | 50, 100, 200, 500 | `demos850_ba05bd96_all_nd{N}_s42_r2_partition` |
| `qwen2.5-vl-7b-instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` | 50, 100, 200, 500 | same |

8 extract cells. Dimension fixed to **`all`**. Rank 2, partition seed 42 (lineage only).

Qwen: runner exits non-zero unless `--max_pixels 1003520`.

Forward-pass count: blocks are disjoint ⇒ 850 unique ids × 2 variants (`value`, `all`) =
**1700 forwards/model** when sharing one demos850 shuffled act cache across the four N. PCA fits
are CPU-side and cheap.

## Dataset construction

For each N ∈ {50, 100, 200, 500}:

1. Take `ids_used` from the deployed cell’s `metadata.json`; assert order-equality with
   `demos_850_partition_s42.json` `blocks[str(N)]`.
2. Captions from `demos_850.jsonl`: `value` and `h_values["all"]` (unchanged).
3. Derangement over those N ids, seed **1234**, 0 fixed points; map is model-independent.
4. Override each flat demo’s `image` to the image of the mapped source-image demo id; keep `id`,
   question, and both captions.

Held constant vs the matching deployed cell: item identities, order, captions, models, token
policy, diff polarity, live PCA (PC1+mean). Destroyed: image–caption binding within the block.

## Net-new code

Do not break demos_v2 `all_nd200` reproducibility. Prefer a **new** module that reuses helpers:

| Piece | Location |
|---|---|
| Module | `evaluation/interventions/vti/shuffled_control_partition.py` |
| Runner | `evaluation/run_scripts/extract_demos850_shuffled_control_directions.py` |

Reuse from existing code (do not duplicate math):

- `generate_derangement`, `validate_derangement`, `build_shuffled_flat_demos` from
  `shuffled_control.py` (import; do not change their demos_v2 default paths)
- `ensure_variant_activation`, `obtain_textual_vti_v2_from_stacks`, `save_textual_vti_v2`-style
  `save_textual_v2_directions`, `load_textual_v2_directions`, `variant_suffix`,
  `demos_content_hash`, `_git_commit` from `directions_v2.py`
- `build_or_load_partition` / partition paths from `directions_partition.py` / `src.paths`
  (`vti_demos_850_path`, `vti_demos_850_partition_path`)

Required API shape (names may vary slightly; behavior must match):

```python
DERANGEMENT_SEED = 1234  # same as demos_v2 nd200 control
DIMENSION = "all"
PARTITION_SIZES = (50, 100, 200, 500)
SLUG_H8 = "ba05bd96"  # from demos_850 hash; recompute + assert at runtime

def derangement_path_for_block(num_demos: int) -> Path
def shuffled_demos850_act_cache_dir(model_short: str) -> Path
def shuffled_demos850_direction_dir(model_short: str, num_demos: int) -> Path
def deployed_partition_slug(demos_hash: str, num_demos: int, dimension: str = "all") -> str
def extract_shuffled_control_partition_direction(
    wrapper, model_short: str, *, num_demos: int, ...
) -> Tuple[np.ndarray, dict, dict]  # directions, meta, checks
def write_sanity_report(...) -> Path
```

## Paths / artifacts

| Path | Content |
|---|---|
| `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd{N}_s1234.json` | Per-block map + seed + n_ids + control-for slug |
| `experiment_artifacts/vti/{model_short}/shuffled_control_demos850/_act_cache/` | Shuffled-image activations only |
| `experiment_artifacts/vti/{model_short}/shuffled_control_demos850/all_nd{N}/` | `directions.npz`, `components.npz`, `metadata.json` |
| `…/shuffled_control_demos850/shuffled_control_sanity_report_{model_short}_all_nd{N}.md` | Gating primitives |
| `…/shuffled_control_demos850/shuffled_control_sanity_rollup_{model_short}_2026-07-29.md` | Optional all-N rollup |

Metadata must include: `control_for_deployed_slug`, `control_type=image_derangement`,
`selection_policy=disjoint_partition`, `demos_file=demos_850.jsonl`,
`content_hash_sha256_16`, `partition_file`, `partition_content_hash`, `block_size`, `ids_used`,
`derangement_file`, `derangement_seed`, `derangement_fixed_points`, `act_cache_dir`,
`forwards_executed`, `cache_hits`, `max_pixels`, `dimension`, `rank`, `git_commit`,
`direction_layer_norms`, PCA diagnostic fields already emitted by
`obtain_textual_vti_v2_from_stacks`.

## Procedure

### 0. Preconditions (fail loud)

- `sha256sum data/vti/demos_850.jsonl` → prefix `ba05bd960cad0c18`
- Partition check 0.5 (disjoint, sizes exact, hash match)
- For each model and each N: deployed dir
  `textual_v2/demos850_ba05bd96_all_nd{N}_s42_r2_partition/` has `directions.npz` +
  `metadata.json` with `n_pairs == N` and `ids_used == blocks[str(N)]`

### 1. Derangements (CPU, once)

For each N: `load_or_write` derangement over that block’s ids, seed 1234; validate 0 fixed points;
write the JSON artifact. Reuse across models.

### 2. Extract per model

Target: **lambdab2**. Check `nvidia-smi`; pin one free GPU via `CUDA_VISIBLE_DEVICES`.
Order: all four N for LLaVA, then all four N for Qwen (do not dual-load on one GPU unless ≥35 GiB
free). Shared demos850 shuffled act cache within a model across N.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_shuffled_control_directions.py \
  --model llava-hf/llava-1.5-7b-hf \
  --demos_path data/vti/demos_850.jsonl \
  --partition_path data/vti/demos_850_partition_s42.json \
  --num_demos 50 100 200 500 \
  --derangement_seed 1234

CUDA_VISIBLE_DEVICES=0 python evaluation/run_scripts/extract_demos850_shuffled_control_directions.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --demos_path data/vti/demos_850.jsonl \
  --partition_path data/vti/demos_850_partition_s42.json \
  --num_demos 50 100 200 500 \
  --derangement_seed 1234 \
  --max_pixels 1003520
```

### 3. Sanity reports

Emit per-(model, N) report; fail the cell (non-zero for that cell / stop trusting it) on any
gating failure.

## Checks (gating)

Same as 7-22, parameterized by N:

| Check | Expect |
|---|---|
| Derangement fixed points | 0 |
| Captions unchanged vs `demos_850.jsonl` | N of N |
| Positional id mismatches vs deployed `ids_used` | 0 |
| Direction shape | LLaVA `(32, 4096)`; Qwen `(28, 3584)` |
| Lineage slug | `demos850_ba05bd96_all_nd{N}_s42_r2_partition` |
| Cache isolation | Act files only under `shuffled_control_demos850/_act_cache/`; never write into `textual_v2/_act_cache` or legacy `shuffled_control/_act_cache` |
| Forward accounting | Across four N: `sum(forwards_executed) +` consistent cache hits ⇒ every one of 850 ids has both `value` and `all` in the demos850 shuffled cache (1700 files/model). Per-cell report records `forwards_executed` and `cache_hits` so silent reuse of the wrong namespace cannot hide. |

**Non-gating:** per-layer L2 norms of shuffled vs deployed `direction_layer_norms` (well-formedness
readout only — not the geometric comparison).

## Explicitly out of scope

- Geometric cosine / magnitude plots (follow-up **design** after all gating PASS)
- Controls for dimensions other than `all`
- Behavioral / `run_eval.py` wiring
- Modifying or regenerating demos_v2 shuffled-control artifacts
- Changing the demos_850 partition or deployed partition directions

## What confirms the controls are valid

For every (model, N): fixed points 0; captions N/N; id-order mismatches 0; correct shape; lineage
slug matches; demos850 shuffled cache holds activations for all 850 ids × 2 variants after both
models’ grids (per model). Any deviation → that cell is not a valid control for a later design.

## Docs after run

Append a factual entry to `RESEARCH_LOG.md` (commands, paths, forward counts, gate pass/fail).
Update `IMPLEMENTATION.md` with a `demos_850 shuffled-control` subsection next to the existing
shuffled-control section. No interpretation.

## Cost

- lambdab2, 1× A6000; ~1700 forwards/model + 4 PCA fits/model.
- Disk: ~850×2 act files/model under the new cache namespace plus 8 direction directories.
