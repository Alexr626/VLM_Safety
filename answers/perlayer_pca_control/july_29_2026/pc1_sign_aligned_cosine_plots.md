# PC1 sign-aligned cosine replot

Yes. Additive diagnostic plots (primary as-is plots unchanged).

**Rule used:** at each decoder layer, if `sign(cos(PC1, mean))` differs between deployed and control, multiply the **control** PC1 by −1; leave layers that already agree unchanged (including both −1). Rebuild directions as `PC1 + mean`, then recompute deployed-vs-control cosine. Applied independently per (model, sample size). Global as-is and per-layer as-is curves are drawn on the same figure for reference.

**Artifacts** (under `diagnostic_experiments/perlayer_pca_control/`):
- `plots/cosine_deployed_vs_control_by_layer_pc1_sign_aligned_between_arms_llava-1.5-7b-hf.png`
- `plots/cosine_deployed_vs_control_by_layer_pc1_sign_aligned_between_arms_qwen2.5-vl-7b-instruct.png`
- matching CSVs under `tables/`
- script: `scripts/plot_cosine_pc1_sign_aligned_between_arms.py`

This is a diagnostic readout of one deferred correction from the plan; it does not replace the primary measurement or rewrite direction artifacts on disk.
