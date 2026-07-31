# Split gallery β column order

**Date:** 2026-07-14

Per-example response columns are now **ascending β**: baseline → 0.2 → 0.5 → 0.9.

Previously they followed the overnight eval order (0.5, 0.2, 0.9). All 320 split
HTMLs were regenerated with `render_demosv2_qual_split_review.py`.
