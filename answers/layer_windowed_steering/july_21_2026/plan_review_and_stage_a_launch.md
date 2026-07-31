# Layer-windowed steering plan review (2026-07-21)

## Verdict

Plan is implementable as written against current code. No blocking clarifications; Stage A launched on lambdab2 GPU 0.

## Clarifications checked (no open blockers)

- Identifiers match source: `vti_hook_ctx`, `optional_steer_ctx`, `run_dump.py`, `layer_windows`, augmented JSONLs, baseline dump trees.
- Windows match `layer_windows(32|28)`: LLaVA 6 bands + all = 7; Qwen 5 + all = 6 → 63 / 54 cells.
- Qwen stays after Stage A (plan default); parallel only if Alex says so on a second free GPU.
- Degenerate / empty outputs recorded, not filtered (sanity gates waived).
- Aggregator supports partial builds for same-day LLaVA POPE deliverable.

## Operational note (not blocking)

Stage A alone is roughly overnight on one A6000 (~4–6 h POPE-30 + ~14–21 h AMBER-100). GPU 0 was free at launch; GPUs 1–3 were occupied. Proceeding per approved plan.

## What was done

1. `layer_indices` on `vti_hook_ctx` + forward via `optional_steer_ctx`; G1 probe uses first layer in window.
2. `run_dump.py`: `--windowed_grid`, `--conditions gold_conditional`.
3. Unit test `tests/test_vti_layer_indices.py` (3 passed).
4. Aggregator `build_windowed_steering_summary.py`.
5. Stage A driver launched under `nohup` (POPE-30 → aggregator → AMBER-100 → aggregator).
