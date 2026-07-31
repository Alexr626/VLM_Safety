# Suggested commit split for current uncommitted changes

**Date:** 2026-07-22  
**Branch:** `activation_steering` (4 commits ahead of origin)  
**Question:** How to split current uncommitted work into 2–4 commits.

## Themes present in the working tree

1. **POPE-30 yes/no pin rebuild + augment plumbing** — rebuild script, updated yes pin, new no pin (currently gitignored), augment registry entries.
2. **Perception-diag tooling** — in-grid baseline in `run_dump.py`, yes/no + AMBER-100 dataset specs in summary/plot builders, concurrent run scripts, galleries helper, reproducibility check.
3. **Summary artifacts** — reorganized per-dataset plots, merged comparison plots, MLP 2×2 JSON, consolidated results, HTML galleries; deletion of old flat POPE-30-only plot paths.
4. **Attention knockout** — new `src/attention_knockout.py` + unit tests; unrelated `tests/test_vti_layer_indices.py`.
5. **Vendored upstream clones** — `VTI/` (~45M) and `llava-interp/` (~1.1M), each with a nested `.git/`. **Do not include in these commits as ordinary trees** without an explicit submodule/vendor decision.

## Recommended: 4 commits

### Commit 1 — POPE yes/no-30 pins + augment allowlist

**Why first:** Dataset identity must land before scripts/results that assume `pope30_yes` / `pope30_no`.

**Include:**
- `data_scripts/rebuild_pope_yes_no_30_pins.py`
- `data/pope/pinned_pope_existence_yes_30.json` (modified)
- `data/pope/pinned_pope_existence_no_30.json` (force-add; currently ignored by `data/pope/*`)
- `data/pope/augmented_pope30_yes.jsonl`, `data/pope/augmented_pope30_no.jsonl` (force-add)
- `.gitignore` allowlist lines for the new pin + augmented JSONLs (mirror existing `yes_30` / `augmented_pope30` exceptions)
- `diagnostic_experiments/perception_diag/augment/build_augmented_jsonl.py`
- `diagnostic_experiments/perception_diag/augment/outputs/augmentation_meta.json`
- `diagnostic_experiments/perception_diag/augment/outputs/leading_clause_gallery.html`
- `data/amber/augmented_amber100.jsonl` (already tracked via allowlist; regenerates with augment changes)

**Suggested message:**
```
Rebuild POPE-30 yes/no pins and wire them into augment.

```

### Commit 2 — Windowed-steering scripts for yes/no + AMBER-100

**Why separate:** Reviewable code change without 50MB+ of plots/JSON.

**Include:**
- `diagnostic_experiments/perception_diag/run_dump.py` (in-grid `baseline` cell)
- `diagnostic_experiments/perception_diag/build_windowed_steering_summary.py`
- `diagnostic_experiments/perception_diag/build_pope30_mlp_2x2_summary.py`
- `diagnostic_experiments/perception_diag/plot_pope30_windowed_steering.py`
- `diagnostic_experiments/perception_diag/build_amber100_llava_response_galleries.py`
- `diagnostic_experiments/perception_diag/check_pope30_yes_reproducibility.py`
- `diagnostic_experiments/perception_diag/run_scripts/run_llava_pope30_yes_no_concurrent.sh`
- `diagnostic_experiments/perception_diag/run_scripts/run_qwen_pope30_yes_no_concurrent.sh`
- `diagnostic_experiments/perception_diag/run_scripts/run_pope30_yes_no_orchestrator.sh`
- `diagnostic_experiments/perception_diag/run_scripts/wait_gpu0_then_qwen_pope30_yes_no.sh`

**Suggested message:**
```
Extend perception-diag windowed steering for POPE yes/no and AMBER-100.

```

### Commit 3 — Summary plots, JSON, and galleries

**Why last among diag work:** Pure artifacts produced by commit 2; matches recent “results” commits on this branch.

**Include:**
- All under `diagnostic_experiments/perception_diag/windowed_steering_summary/` that changed:
  - `windowed_steering_consolidated_results.json`
  - `pope30_{yes,no}_mlp_2x2_*.json` (add); delete old unscoped `pope30_mlp_2x2_*.json`
  - plot tree moves: delete flat `plots/{llava,qwen}/pope30_*.png`; add `pope30_yes/`, `pope30_no/`, AMBER-100 subsets, `plots/merged_plots/`
  - new AMBER HTML galleries; delete obsolete POPE gallery HTMLs

**Suggested message:**
```
Refresh windowed-steering summary plots and galleries for yes/no splits.

```

### Commit 4 — Attention knockout + small unit tests

**Why separate:** Orthogonal to the perception-diag yes/no control work.

**Include:**
- `src/attention_knockout.py`
- `tests/test_attention_knockout.py`
- `tests/test_vti_layer_indices.py` (same “new tests” theme; optional to fold here)

**Suggested message:**
```
Add attention-knockout mask helpers and unit tests.

```

## If you prefer only 3 commits

Merge **1+2** into one “data + tooling” commit, keep **3** (artifacts) and **4** (knockout). That is fine if you want fewer stops; the 4-way split is cleaner for bisect/review.

## Leave out (for now)

| Path | Reason |
|------|--------|
| `VTI/` | Nested git repo (~45M); decide submodule vs copy-without-`.git` first |
| `llava-interp/` | Same nested-`.git` issue; reference for knockout literature, not needed for current diag commits |
| `data/pope/dumps/` | Already gitignored heavy activations |

## Force-add note

`pinned_pope_existence_no_30.json` and the new `augmented_pope30_{yes,no}.jsonl` will not appear in `git status` until `.gitignore` allowlist entries are added (or until `git add -f`). Commit 1 should update `.gitignore` in the same change.
