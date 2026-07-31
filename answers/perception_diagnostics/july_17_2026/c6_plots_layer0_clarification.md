# C6 plots — not layer-0-only

**Date:** 2026-07-17

**Q:** Is there a reason the C6 plots only include the layer 0 last-token norms?

**A:** No — they do not. `c6_B0_by_condition.png` plots **mean last-prefill-token ℓ2 vs layer index** for every layer (`0` = embedding … `32` = final decoder). The axis label `(0=embed)` only explains what index 0 is.

What *is* limited vs the plan: only the **last-token** depth profile is plotted today. The plan’s second C6 panel — **per-position profile / position-0 sink** — is computed only as a scalar `pos0_mean_over_layers` in `c6_norm_profiles_B0.json` and is **not** plotted yet.
