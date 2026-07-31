# perlayer_pca_control directory cleanup

Reorganized the formerly flat `diagnostic_experiments/perlayer_pca_control/` tree:

```
perlayer_pca_control/
  scripts/        # compare_*.py, plot_cosine_pc1_sign_aligned_*.py
  verification/   # sha256 manifests, byte-identity / cache reports, extraction manifest
  plots/          # all PNGs (primary + sign-aligned diagnostic)
  tables/         # all CSVs
  summaries/      # perlayer_vs_global_pca_control_summary.md
```

Scripts and the verify helper now default to these subdirs. Primary plot paths for viewing:

- `plots/cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_{model}.png`
- `plots/cosine_deployed_vs_control_by_layer_pc1_sign_aligned_between_arms_{model}.png`
