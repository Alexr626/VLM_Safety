# Audit: git-archive gaps for the triple CHAIR/POPE run

Date: 2026-07-31

## Path the job uses

`hal-steer-triple` → per-model helpers → `run_steering_visual_reasoning_validation.sh`
→ `evaluation/run_eval.py` → `load_chair` / `load_pope` + meandiff dirs.

## Required inputs vs git archive

| Asset | Needed for | In `git archive`? | How it gets on NFS |
|-------|------------|-------------------|--------------------|
| Code / helpers | all | yes (+ overlay) | repo tarball |
| Meandiff `directions.npz` ×4×2 models | steered cells | **no** | `llava_` / `qwen_meandiff_*.tar.gz` |
| `data/chair/pinned_chair_500.json` | CHAIR subset | **no** (gitignore) | pack overlay (added) |
| `data/chair/combined.json` (~16MB) | CHAIR sample list + image paths | **no** (gitignore) | pack overlay (**just added**) |
| `data/pope/combined.json` + `output/coco/coco_pope_*.json` | POPE | **yes** | repo tarball |
| `data/amber/pinned_amber_disc_450.json` | AMBER only | **yes** | repo tarball (not used by triple default) |
| `data/vti/demos_850*.json(l)` | run manifest hashes only | **yes** | repo tarball |
| `data/coco/val2014/*.jpg` | CHAIR + POPE images | **no** (`.gitkeep` only) | `setup_vlm` / download on NFS |
| `data/coco/annotations/instances_val2014.json` | CHAIR metrics | **no** | same setup download |
| HF LLaVA / Qwen weights | models | **no** | `hf_cache` + smoke/triple prefetch |
| Absolute paths in combined.json | image open | n/a | `remap_lambdab2_paths.py --apply` in helpers |

## Next failure if only the pin was synced

After fixing the pin alone, the next hard error would have been missing
`data/chair/combined.json` (`load_chair` requires it). That file is now in the
pack overlay; verify fails closed if it or COCO val2014 / instances are absent.

## POPE-specific

No separate pin file for this grid (`--limit 200`). Question JSONs are tracked.
Image paths inside `data/pope/combined.json` are lambdab2 absolute paths —
remap rewrites them; images themselves must exist under NFS
`data/coco/val2014/` (from setup).

## Not a mid-run surprise if verify passes

Expanded `verify_steering_nfs_layout.sh` now checks chair pin + combined,
POPE JSONs, demos_850 files, coco val2014 nonempty, and instances_val2014.json
before workers start.
