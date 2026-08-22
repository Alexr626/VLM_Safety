# Where are the 06-19 vs 07-30 POPE comparison plots?

Date: 2026-08-17  
Question: Where are the plots that compare the runs inventoried in `answers/runs/august_6_2026/pope_0619_vs_0730_experimental_config.md`?

---

They are on this workstation under:

`evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/plots/`

That tree was written by `evaluation/pope_0619_vs_0730_matched/build_comparison.py` (logged in `RESEARCH_LOG.md` § 2026-08-06). The config inventory itself does not contain plots; it only names the two cell trees.

The parent directory is gitignored (`evaluation/results`). It is present locally (listed 2026-08-17).

## Plot inventory (25 PNGs)

**Metric bars vs β** (accuracy, precision, recall, accuracy_gold_no, accuracy_gold_yes):

- `plots/split_averaged/{metric}_vs_beta_bars.png` (5 files)
- `plots/per_split/{random,popular,adversarial}/{metric}_vs_beta_bars.png` (15 files)

**Direction geometry:**

- `plots/direction_magnitude_per_layer.png`
- `plots/direction_magnitude_abs_diff_per_layer.png`
- `plots/direction_cosine/cosine_0619_pc1mean_vs_0730_meandiff.png`
- `plots/direction_cosine/cosine_0619_pc1mean_vs_demos850_pc1mean.png`
- `plots/direction_cosine/cosine_0730_meandiff_vs_demos850_pc1mean.png`

Sibling tables/JSON live next to `plots/`: `README.md`, `comparison_tables.md`, `cells.json`, `direction_cosines.json`, `direction_magnitudes.json`, `tables/`.

## What those plots compare

Matched LLaVA POPE frame only (`vti_textual_additive_mlp`, all layers, β ∈ {0.2, 0.5, 0.9}):

- 06-19: author demos nd70, PC1+mean (`evaluation/results/2026-06-19/`)
- 07-30: demos850 nd500 meandiff `layers_all` (`evaluation/results/2026-07-30/`)

Magnitude/cosine plots also overlay a demos850 nd500 PC1+mean control direction (`*_r2_partition`); that control is not a POPE eval cell in this comparison.

There is no `analysis/` reading of this comparison. AMBER is not in this plot set.
