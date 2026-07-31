# Package C — implemented and run on LLaVA AMBER-25

**Date:** 2026-07-17

## Verdict

Package C data scripts **are implemented** (C1–C8). C3/C6 were fixed to the current `acts/*.npy` / `norms/*.npy` dump layout; a plotter + driver were added. They have been run on the LLaVA S1 AMBER-25 dump. **C7 is the only gap for review** — it needs an H-D QC dump (`hd_qc/run_hd_qc.py`) that has not been produced yet.

## How to run

```bash
bash diagnostic_experiments/perception_diag/run_scripts/run_package_c.sh \
  diagnostic_experiments/perception_diag/llava-1.5-7b-hf/dumps/s1_full_cell_smoke

# After Qwen smokes finish:
# bash …/run_package_c.sh …/qwen2.5-vl-7b-instruct/dumps/s3b_qwen25_amber25
# bash …/run_package_c.sh …/qwen2-vl-7b-instruct/dumps/s3a_qwen2_amber25

# Optional H-D:
# HD_DIR=…/hd_qc/<tag> bash …/run_package_c.sh <dump_dir>
```

## Review entry point (LLaVA S1)

Open:

`diagnostic_experiments/perception_diag/llava-1.5-7b-hf/dumps/s1_full_cell_smoke/package_c_plots/index.html`

Also linked from that page: `../c3_plots/`, `../c6_plots/`.

| ID | What to look at |
|----|-----------------|
| C1 | `c1_readout_validity.png` + JSON `by_condition` |
| C2 | `c2_auc_by_layer.png`, `c2_transfer_vs_beta.png` |
| C3 | `c3_plots/c3_L*.png` (d_lead × y_steer⊥) |
| C4 | `c4_translation_vs_separability.png`, `c4_gold_auc_by_layer.png` |
| C5 | `c5_mean_delta_by_cell.png`, `c5_delta_distributions.png` |
| C6 | `c6_plots/c6_B0_by_condition.png` |
| C7 | skipped until HD QC dump exists |
| C8 | `c8_scatter_auc_bundle.json` |

## Notes for Qwen smokes

Same command once `s3b_qwen25_amber25` / `s3a_qwen2_amber25` dumps complete. C7 remains shared per model (HD QC is direction/demo holdout, not dump-cell-specific) once `run_hd_qc.py` has been run for that model.
