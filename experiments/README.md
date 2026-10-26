# experiments/

Home for the project's experiment code. It is empty on purpose: nothing from
the proposal (decoding loop with reruns/forks, cone correction, probes, signal
ladder, BRACS / VGD reimplementations, image pools) is implemented yet.

## Conventions

One directory per experiment, named by what it does (e.g. `stage1_detection/`):

```
experiments/{experiment}/
├── run_*.py / *.sh                  # entry points; take --model <hf-id>
└── outputs/{model_short}/results/   # small JSON summaries + plots/ (committed)
artifacts/{experiment}/{model_short}/  # cached states, probe weights (gitignored)
```

Use the helpers in `src/paths.py` (`experiment_results_dir`, `experiment_plots_dir`,
`artifacts_dir`) rather than hardcoding paths, and `src/model.py:_normalize_model_name`
for `model_short` (`llava-1.5-7b-hf`, `qwen2.5-vl-7b-instruct`).

Shared building blocks:

- `src/model.py` — `create_wrapper(model_id, pad_to_square=..., max_pixels=...)`
- `src/model_internals.py` — decoder layers, RMSNorm gain/eps, q/k projections,
  rotary embedding, lm_head, image-token span, BOS position
- `src/hooks.py` — `capture_residual` / `edit_residual` context managers
- `src/liseco.py` — imports the cloned LiSeCo code (`llm-control/src`)
- `src/dataset.py`, `evaluation/` — POPE / AMBER / CHAIR loaders and scorers
