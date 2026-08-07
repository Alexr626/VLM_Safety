# Artifact index

Append-only location log for experiment artifacts. Paths, approximate dates, contents, regenerable vs must-restore. **No metric readings.** Code recipes: `IMPLEMENTATION.md`. Restore packs: `ARTIFACT_ARCHIVE.md`.

Convention: **must-restore** = expensive or identity-bearing (pins, fitted directions used in published cells). **regenerable** = can be rebuilt with existing scripts if inputs exist (act caches, many plots).

---

## Generated data pins and demos

| Path | Approx date | Contents | Experiment / use | Regenerable? |
|------|-------------|----------|------------------|--------------|
| `data/vti/demos.jsonl` | early | Author VTI paired demos | Legacy nd70 textual directions | must-restore (or copy from upstream VTI) |
| `data/vti/demos_v2.jsonl` | 2026-07 | Expanded demos pool | demos_v2 prefix directions | must-restore |
| `data/vti/demos_v2_order_s42.json` | 2026-07 | Order pin | prefix selection | must-restore |
| `data/vti/demos_850.jsonl` | 2026-07-28 | 850-row pool | demos850 partitions / meandiff / PCA | must-restore (tracked in git) |
| `data/vti/demos_850_partition_s42.json` | 2026-07-28 | Disjoint 50/100/200/500 blocks | same | must-restore (tracked) |
| `data/amber/pinned_amber_disc_450.json` | 2026-06 | 450-item AMBER disc pin | overnight grids | must-restore (tracked) |
| `data/amber/pinned_amber_disc_1500.json` | 2026-08-05 | 1500-item AMBER disc pin (superset of 450) | AMBER expanded grid | must-restore (tracked) |
| `data/amber/pinned_amber_disc_100.json` | 2026-06 | 100-item pin | diagnostics | must-restore (tracked) |
| `data/chair/pinned_chair_500.json` | 2026-06 | 500-image CHAIR pin | CHAIR evals | must-restore (tracked) |
| `data/pope/pinned_pope_existence_*.json` | 2026-07 | POPE yes/no subset pins | windowed / diagnostic | must-restore (tracked) |
| `data/vti/qual_subset_*.json` | 2026-06 | Qualitative subset pins | review galleries | must-restore (tracked) |
| `data/amber/dumps/`, `data/pope/dumps/` | various | Steered-capture dumps | diagnostics | regenerable / must-restore depending on cell |
| `data/coco/` | — | COCO val2014 (+ train2014) | benchmarks / demos | regenerable via `data_scripts/download_chair.py` |

---

## Direction artifacts (`experiment_artifacts/vti/`)

Layout: `experiment_artifacts/vti/{model_short}/…`. Direction dirs typically contain `directions.npz` + `metadata.json` (+ `components.npz` for PCA).

| Path pattern | Approx date | Contents | Experiment / use | Regenerable? |
|--------------|-------------|----------|------------------|--------------|
| `{model}/textual_directions_nd70_rank1_seed42.npz` | 2026-06 | Legacy author-demo PC1 directions | 2026-06-19 POPE cells | must-restore for 06-19 comparison (some tracked) |
| `{model}/textual_v2/demosv2_*_r2_prefix/` | 2026-07 | demos_v2 PCA PC1+mean | early demos_v2 grids | regenerable if act-cache + demos present |
| `{model}/textual_v2/demos850_*_r2_partition/` | 2026-07-28+ | demos850 global PCA partition dirs | geometric + AMBER expanded PC1+mean | must-restore for published cells; else regenerable |
| `{model}/textual_v2/demos850_*_meandiff_partition/` | 2026-07-30 | Raw mean-difference dirs | 2026-07-30 overnight + AMBER expanded | must-restore for published cells |
| `{model}/textual_v2/_act_cache/` | ongoing | Last-token activation stacks | input to PCA/meandiff extract | regenerable (GPU); large (~1–2G/model) |
| `demos850_meandiff_extraction_manifest_2026-07-30.json` | 2026-07-30 | Extraction manifest | verify meandiff | tracked in git |
| `{model}/shuffled_control/all_nd200/` | 2026-07 | Shuffled-image control directions | geometric comparison | some tracked |
| `shuffled_control_*derangement*.json` | 2026-07 | Image derangement maps | shuffled control | tracked |
| `{model}/shuffled_control_demos850/` | 2026-07-29 | demos850 shuffled directions + cache | gitignored local | regenerable / must-restore |
| `{model}/textual_v2_perlayer/…` | 2026-07-29 | Per-layer PCA directions | geometric comparison | regenerable from act-cache |
| `{model}/shuffled_control_demos850_perlayer/` | 2026-07-29 | Per-layer shuffled control | gitignored | regenerable |
| `{model}/visual/patch_mask_*/` | 2026-06 | Visual-arm direction stubs | early vision experiments | tracked samples |

Models present on disk (2026-08): `llava-1.5-7b-hf`, `qwen2.5-vl-7b-instruct`, `qwen2-vl-7b-instruct`, `qwen-vl-chat`.

---

## Evaluation results (`evaluation/results/`)

Mostly gitignored. Sizes as of 2026-08-07 survey.

| Path | Approx date | Contents | Experiment / use | Regenerable? |
|------|-------------|----------|------------------|--------------|
| `2026-06-19/` (~50M) | 2026-06-19 | LLaVA POPE (+ related) cells | **06-19 arm of matched POPE comparison** | must-restore for comparison |
| `2026-07-30/` (~115M) | 2026-07-30 | Overnight steering visual-reasoning grid + analysis | **07-30 meandiff arm**; multi-benchmark | must-restore |
| `2026-08-05/` (~26M) | 2026-08-05 | AMBER-1500 expanded grid cells + analysis | steering validation continuation | must-restore |
| `2026-08-06/_analysis_pope_0619_vs_0730_matched/` (~2M under date) | 2026-08-06 | Tables + bar/cosine/magnitude plots | **primary matched comparison** | regenerable if both cell trees present |
| `2026-07-13/` (~2.4G) | 2026-07-13 | Large mid-July runs | historical | optional archive |
| Other date dirs | various | Earlier grids / smokes | historical | optional |

Key analysis builders (code in git):

- `evaluation/pope_0619_vs_0730_matched/build_comparison.py` → writes under `2026-08-06/_analysis_pope_0619_vs_0730_matched/`
- `evaluation/steering_vector_validation_continuation/build_per_configuration_and_per_item_tables.py`
- `evaluation/steering_visual_reasoning_validation/{build_result_tables,make_plots,make_organized_plots}.py`

---

## Diagnostic experiment outputs

| Path | Approx date | Contents | Regenerable? |
|------|-------------|----------|--------------|
| `diagnostic_experiments/perlayer_pca_control/verification/` | 2026-07-29 | SHA manifests, geometric reports | some tracked; plots regenerable |
| `diagnostic_experiments/*/results/` | various | Modality shift, mediation, etc. | mostly regenerable |

---

## How to extend this file

When a new direction set or result tree lands, append one row: path, date, what files are inside, which run produced it, regenerable vs must-restore. Do not paste accuracy tables here.
