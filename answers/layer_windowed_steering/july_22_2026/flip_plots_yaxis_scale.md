# Flip-from-baseline plot y-axis scale

**Date:** 2026-07-22

## Request

Tighten y-axes on flips-from-baseline plots; old 0…n+2 (~0–32 for POPE-30) was hard to read. Prefer something like `round(n/4)` going forward.

## What changed

In `diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py`:

- `flips_ylim_max(n_items, observed_max) = max(round(n_items/4), ceil(observed_max))`
- Used by per-method flip plots and joint LLaVA×Qwen flip plots
- Integer tick locator: `MaxNLocator(integer=True, nbins=8)` (avoids one tick per count when the axis is tall)

Examples: POPE n=30 → preferred ymax **8**; AMBER subsets n=20 → preferred ymax **5**. If a figure’s tallest bar exceeds that, ymax expands so bars are not clipped (e.g. some Qwen POPE-30-no rotation@layer cells).

## Regenerated

All `*flips_from_baseline*` PNGs for LLaVA (POPE-30 yes/no + AMBER-100 attribute/relation yes/no), Qwen (POPE-30 yes/no), and merged joint flip plots under `…/plots/merged_plots/`.
