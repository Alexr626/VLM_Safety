# Reviewing model responses side-by-side with the input image

**Question:** When reviewing raw model responses in result files (e.g.
`evaluation/vti_rotation_strength/results/.../sweep_uniform_rotation_layer_random_n200.json`),
how do I quickly look up the image the model was fed for a POPE id like
`pope_random_00166`, and see the responses next to it?

## TL;DR

Use `helper_scripts/review_responses.py`. It resolves the image + question +
ground truth for any benchmark sample id and pulls the model response(s) out of
one or more results JSON files. It can print a terminal summary, open the image
in the OS viewer, or build a self-contained HTML page with the image
**side-by-side** with the responses.

```bash
# Terminal summary (image path + question + GT + all responses):
python helper_scripts/review_responses.py pope_random_00166 \
  evaluation/vti_rotation_strength/results/2026-06-19/qwen2.5-vl-7b-instruct/\
sweep_uniform_rotation_layer_random_n200.json

# Side-by-side image + responses in the browser:
python helper_scripts/review_responses.py pope_random_00166 \
  evaluation/vti_rotation_strength/results/2026-06-19/qwen2.5-vl-7b-instruct/\
sweep_uniform_rotation_layer_random_n200.json --html

# Just open the input image the model was fed (no results file needed):
python helper_scripts/review_responses.py pope_random_00166 --open
```

## Why this works (data model)

- POPE sample ids map **directly** to entries in `data/pope/combined.json`. Each
  entry carries `id`, `image_path` (absolute COCO val2014 path), `text` (the
  yes/no question) and `label` (ground truth). The script looks the id up there.
  `image_path` for `pope_random_00166` is
  `data/coco/val2014/COCO_val2014_000000140583.jpg` (`Is there a cow in the image?`, GT `yes`).
- The `vti_rotation_strength` sweep JSON stores a full `per_sample` list keyed by
  `id`, each record holding `baseline`, `by_beta[beta].response` for every swept
  beta, plus the `decode_only_beta_max` / `skip_pos0_beta_max` probe outputs. The
  script extracts all of these for the requested id (it also falls back to the
  capped `changed_examples_by_beta` block, and handles standard `run_eval`
  `responses.json` files — a list of `{id, response, ...}` records).

## Features

- `sample_id` (positional, required), then **zero or more** results JSON files
  (positional). Multiple files render as separate sections so you can compare
  models/variants for one id.
- `--benchmark KEY` overrides the benchmark (default: inferred from the id
  prefix, e.g. `pope`). Needed for ids whose prefix isn't the registry key.
- `--open` opens the input image in the system viewer (`xdg-open` on Linux).
- `--html` builds a portable HTML page (image base64-embedded, so it works over
  `file://`) with the image pinned on the left and the responses on the right;
  it auto-opens unless `--no-open` is given. `--html-out PATH` chooses where to
  write it (default: a temp file).
- `--max-chars N` truncates long responses in the terminal view (0 = no limit).
- Each response is auto-tagged with the parsed yes/no decision (mirroring the
  POPE `_normalize_yes_no` leading-token-priority rule) so flips across betas are
  obvious at a glance.

The script is pure stdlib (no torch/PIL import), so it runs instantly and
outside the conda env.

## Caveat

Older sweep JSONs under `results/2026-06-18/` use the pre-relocation
`metrics_by_alpha` schema and may not include `per_sample`; for those the script
falls back to the `changed_examples_by_beta` examples (capped to 5 ids/beta), so
an arbitrary id may not be present. The 2026-06-19+ files include full
`per_sample` and work for any of the N ids in the sweep.
