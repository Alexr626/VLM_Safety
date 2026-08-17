# Context files for the 06-19 vs 07-30 comparison

Date: 2026-08-16.

The named comparison on disk is **LLaVA POPE**, not AMBER. Directory slug: `pope_0619_vs_0730_matched`. The matched frame dropped CHAIR/AMBER/rotation-strength arms (`answers/runs/august_6_2026/pope_0619_vs_0730_experimental_config.md`). AMBER appears in the **07-30 overnight grid** and in a **separate 08-05 AMBER-1500 continuation**, not as the 06-19 vs 07-30 matched comparison.

There is no `analysis/` reading for this comparison. `analysis/08_05_26/steering_vector_visual_reasoning_validation.md` is still an unfilled template.

---

## Start here

| Role | Path |
|------|------|
| Index / “current thread” | `readme.md` (Internship handoff), `REPO_MAP.md` |
| Artifact locations | `ARTIFACT_INDEX.md` |
| Code facts | `IMPLEMENTATION.md` (Matched POPE cross-date comparison, 2026-08-06) |
| Run record (numbers + commands) | `RESEARCH_LOG.md` § `2026-08-06 — matched LLaVA POPE comparison` |
| Config inventory (what differed) | `answers/runs/august_6_2026/pope_0619_vs_0730_experimental_config.md` |
| Builder | `evaluation/pope_0619_vs_0730_matched/build_comparison.py` |
| Analysis output (on disk) | `evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/` |

## Analysis output (present on this workstation)

- `README.md`, `comparison_tables.md`
- `cells.json`, `direction_cosines.json`, `direction_magnitudes.json`
- `tables/{per_split,split_averaged}_metrics.csv`
- `plots/split_averaged/*_vs_beta_bars.png`
- `plots/direction_magnitude_per_layer.png`, `direction_magnitude_abs_diff_per_layer.png`
- `plots/direction_cosine/cosine_0619_pc1mean_vs_0730_meandiff.png` (and two other pairwise cosine plots)

## Source cells

- 06-19 arm: `evaluation/results/2026-06-19/`
- 07-30 arm: `evaluation/results/2026-07-30/`
- Directions: `ARTIFACT_INDEX.md` rows for `{model}/textual_directions_nd70_rank1_seed42.npz` (06-19) and `{model}/textual_v2/demos850_*_meandiff_partition/` (07-30)

## Related notes (not the matched comparison itself)

| Path | What it is |
|------|------------|
| `answers/concepts/august_6_2026/direction_magnitude_reshape_0619_vs_0730.md` | Magnitude / reshape note on the two direction constructions |
| `answers/concepts/august_6_2026/pca_mean_vs_raw_mean_difference.md` | How 06-19 PC1+mean relates to 07-30 meandiff |
| `designs/07_30_26/steering_vector_visual_reasoning_validation.md` | Spec behind the 07-30 overnight grid (AMBER + POPE + CHAIR) |
| `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md` | Plan for that grid |
| `answers/runs/july_31_2026/comparable_vti_pc1_plus_mean_amber_prior_runs.md` | Whether a PC1+mean AMBER cell matches the 07-30 meandiff AMBER grid |
| `evaluation/results/2026-08-05/` | Later AMBER-1500 continuation (meandiff + PC1+mean), not 06-19 vs 07-30 |
