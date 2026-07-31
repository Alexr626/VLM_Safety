# Split gallery filenames (no more review.html)

**Date:** 2026-07-14

Files under the split subdirectory tree are now named:

`{model}_{benchmark}_{intervention}_{dimension}_nd{N}.html`

Example:
`evaluation/results/2026-07-13/_samples/llava-1.5-7b-hf/amber/additive_layer/counting/100/llava-1.5-7b-hf_amber_additive_layer_counting_nd100.html`

All 320 existing `review.html` files were renamed in place; the renderer
(`helper_scripts/render_demosv2_qual_split_review.py`) writes this pattern going forward.
