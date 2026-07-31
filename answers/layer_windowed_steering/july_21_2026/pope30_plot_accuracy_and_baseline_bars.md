# POPE-30 plot revisions (accuracy, legend, baseline bars) — 2026-07-21

## Changes

1. **Accuracy naming** — titles, y-axis, captions, and filenames now use `accuracy` (not “parsed yes-rate”). On this all-gold=yes set it is the parseable-yes fraction.

2. **Legend** — β colors on one row; no-intervention baseline (dashed line or hatched bar) on a second row below, so the keys do not sit on top of each other.

3. **Second plot set** — `plots/with_baseline_bars/`: same 9 plots, but no-intervention is drawn as a grey hatched bar at every layer window next to the β bars (same baseline height within a panel; for flips the baseline bar is 0).

## Paths

- Line baseline: `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/`
- Bar baseline: `…/plots/with_baseline_bars/`
