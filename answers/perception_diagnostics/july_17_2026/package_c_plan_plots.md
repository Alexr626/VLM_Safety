# Package C — plan-suggested plots

**Date:** 2026-07-17

Yes. Plan v2 §5 names these visual/readout products (beyond JSON tables):

| ID | Plan figure language | Status on LLaVA S1 |
|----|----------------------|--------------------|
| C1 | agreement / mass / unp vs β (dose-response) | `c1_readout_validity.png`, `c1_dose_response_vs_beta.png` |
| C2 | AUC; transfer margin + fire vs β; cross-tone | `c2_auc_by_layer.png`, `c2_transfer_vs_beta.png`, `c2_fire_fraction_vs_beta.png`, `c2_cross_tone_auc.png` |
| C3 | supervised-frame scatters | `c3_plots/c3_L{00–31}.png` |
| C4 | translation-vs-separability; gold AUC vs β | `c4_translation_vs_separability.png`, `c4_gold_auc_vs_beta.png`, `c4_gold_auc_by_layer.png` |
| C5 | Δ(yes−no) distributions / mean±disp | `c5_mean_delta_by_cell.png`, `c5_delta_distributions.png` |
| C6 | norms by condition/outcome; pos-0 / per-pos | `c6_plots/c6_B0_{by_condition,by_outcome,pos0_vs_layer,per_position}.png` |
| C7 | HD QC projection AUCs | skipped (no HD dump yet) |
| C8 | projection AUCs + fixed-frame scatters | `c8_projection_auc_by_layer.png`, `c8_scatters/*.png` |

**Entry point:** `diagnostic_experiments/perception_diag/llava-1.5-7b-hf/dumps/s1_full_cell_smoke/package_c_plots/index.html`

**Scripts:** `analyze/plot_package_c.py`, `analyze/c6_norm_profiles.py` (C3 via `c3_supervised_frame.py`); driver `run_scripts/run_package_c.sh`.
