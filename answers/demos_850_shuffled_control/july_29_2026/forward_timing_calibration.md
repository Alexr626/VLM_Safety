# Calibration: is ~10 min for 1700 LLaVA forwards plausible?

Date: 2026-07-29

Yes. Measured from shuffled LLaVA act-cache write times: **1700 files in 598 s (10.0 min) → ~0.35 s/forward**.

Same machine earlier: demos850 partition LLaVA run_meta `wall_clock_sec=967` (~16 min) for load + fidelity + extracting up to 2100 new forwards across five dimensions (plus cache hits / PCA). Order of magnitude matches.

Feeling of “too fast” is mostly comparing to jobs with **many more** forwards (full grids, multi-dim × multi-N with less cache reuse, Qwen at 1M pixels, overnight mining, etc.), not a sign that shuffled LLaVA skipped work.
