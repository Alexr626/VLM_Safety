# What are the vertical orange bars?

They are layer highlights, not a plotted metric.

**Where they actually appear**
1. `plots/cosine_deployed_vs_control_by_layer_pc1_sign_aligned_between_arms_*.png` — sign-aligned cosine figures
2. `plots/pc1_sign_agreement_by_layer_per_layer_and_global_pca_*.png` — Check 9 sign-agreement figures

They do **not** appear on the primary raw cosine comparison
(`plots/cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_*.png`). That
figure’s orange is a **dashed line** (global PCA cosine), not vertical bars.

**Meaning**
- **Sign-agreement plots:** a bar at layer `l` means
  `sign(cos(PC1_l, mean_l))` differs between the deployed arm and the
  shuffled-image control arm at that layer (Check 9 disagreement). Orange
  markers on the step traces are the same condition.
- **Sign-aligned cosine plots:** a bar at layer `l` means the diagnostic
  replot **multiplied the control PC1 by −1** at that layer so the signs
  would agree, then rebuilt `direction = PC1 + mean` before computing
  cosine. Those are exactly the disagreement layers from Check 9.

So the bars mark the same underlying fact: per-layer PC1 orientation
disagreement between arms (via `sign(cos(PC1, mean))`). On the sign-aligned
cosine figure they also record where that disagreement was corrected for the
diagnostic curve.
