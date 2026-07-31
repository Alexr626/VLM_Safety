# Why the rotation-strength checkpoint JSON does not render, and how to check progress

**Date:** 2026-06-23

## Why the checkpoint file looks broken in the editor

The file `sweep_uniform_rotation_layer_n500.checkpoint.json` **is valid JSON**, but it is written as **one minified line** (~300KB+) containing every sample's full baseline caption and all steered responses (9 betas + 2 probes). Most IDEs cannot syntax-highlight or fold that usefully.

Do **not** try to open the checkpoint for progress monitoring.

## Current Qwen CHAIR run status (as of manual inspect)

- **75 / 500 samples** complete (**15%**)
- **0** OOM failures
- **~900 / ~6000** generations (12 per sample: 1 baseline + 9 betas + 2 probes)
- Fine beta grid confirmed: `0.6 … 0.1`
- Process still running under nohup

## How to check progress

### Option A — open the progress sidecar (recommended)

```
evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/chair_rotation_strength/
  sweep_uniform_rotation_layer_n500.progress.json
```

Small, pretty-printed, no caption text. Regenerated from the checkpoint on 2026-06-23; the running job will write this automatically on future runs (every 25 samples) after the driver update.

### Option B — inspect script (works on checkpoint or directory)

```bash
python evaluation/chair_amber_diagnostics/inspect_rotation_progress.py \
  evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/chair_rotation_strength/
```

### Option C — one-liner

```bash
python3 -c "import json; ck=json.load(open('evaluation/results/2026-06-22/qwen2.5-vl-7b-instruct/chair_rotation_strength/sweep_uniform_rotation_layer_n500.checkpoint.json')); print(len(ck['per_sample']), '/ 500')"
```

## Why the log file is quiet

`nohup` buffers stdout. Progress prints only every **25 samples**, and the first line would appear at sample 25 — before the driver update those lines may not have flushed. The log will stay sparse until the next milestone unless you relaunch with `python -u` or `stdbuf -oL`.
