# Sign-aligned cosine plots: drop as-is per-layer overlay

Updated `scripts/plot_cosine_pc1_sign_aligned_between_arms.py` so each panel
shows only:

1. Per-layer PCA with PC1 signs aligned (green)
2. Global PCA as-is (orange dashed), as reference

As-is per-layer is no longer overlaid here — it remains in the primary
`plots/cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_*.png`
figures. Orange vertical bands still mark layers where control PC1 was flipped.
Plots regenerated in place under `plots/`.
