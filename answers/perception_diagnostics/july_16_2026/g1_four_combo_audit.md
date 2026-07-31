# G1 steered-capture audit (analyst tightening)

**Done 2026-07-16.**

1. **Provenance:** `g1_steered_capture_check.json` now includes model, method, cell_id, site, strength, layer, n_items, pass counts, max_abs_diff stats.
2. **Coverage:** all four method×site at strength 0.5 over **25** AMBER-25 items — B2 (rot/mlp), B5 (rot/layer), B8 (add/mlp), B11 (add/layer).

**Result:** `live_all_pass=True`; worst `max_abs_diff_max` = 4.88e-4 (tol 1e-3).

Paths:
- `artifact_checks/g1_steered_capture_audit.json` (full)
- `llava-1.5-7b-hf/dumps/s0_smoke/g1_steered_capture_check.json` (condensed, supersedes old 1-point artifact)
