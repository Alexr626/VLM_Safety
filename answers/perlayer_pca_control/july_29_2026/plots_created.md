# Were plots created for the per-layer PCA vs global PCA run?

Yes. PNGs live under `diagnostic_experiments/perlayer_pca_control/plots/` (reorganized 2026-07-29):

**Primary (deployed vs shuffled-image control cosine)**
- `plots/cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_llava-1.5-7b-hf.png`
- `plots/cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_qwen2.5-vl-7b-instruct.png`

**Direction L2 / PC1 share / PC1 sign agreement** — same `plots/` directory.

**PC1 sign-aligned diagnostic**
- `plots/cosine_deployed_vs_control_by_layer_pc1_sign_aligned_between_arms_{model}.png`

CSVs: `tables/`. Summary: `summaries/`. Verification: `verification/`. Scripts: `scripts/`.
