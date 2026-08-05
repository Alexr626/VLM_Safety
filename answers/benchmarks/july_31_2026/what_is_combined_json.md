# What is `data/*/combined.json`?

Date: 2026-07-31

Per-benchmark flattened sample manifest written by the download scripts
(`data_scripts/download_*.py` → `write_combined` in `src/dataset.py`).

Each row is one eval sample: `id`, `image_path`, `text` (prompt), `label` /
task fields, plus `raw` metadata. Loaders (`load_chair`, `load_pope`, …) prefer
this file when present so eval does not re-parse upstream formats every time.

For CHAIR: ~40k COCO val2014 captioning rows. The pinned subset
(`pinned_chair_500.json`) selects 500 ids from this list; images are opened from
`image_path` (often absolute paths that remap on RunAI).
