# POPE-30-no mean-probability plots use P(no)

For the POPE-30-no windowed-steering probability plots, mean first-token probability is now **P(no)** (`score_p_no_raw`), not P(yes).

- Files: `…/plots/<model>/pope30_no/pope30_mean_p_no_raw_by_layer_window_{rotation_mlp,rotation_layer,additive_mlp}.png`
- POPE-30-yes plots are unchanged (still mean P(yes) / `score_p_yes_raw`)
- LLaVA plots under `plots/llava-1.5-7b-hf/pope30_no/` were regenerated 2026-07-21
