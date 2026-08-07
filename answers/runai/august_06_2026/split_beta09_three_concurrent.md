# Split remaining β=0.9 across three concurrent Qwen instances?

Date: 2026-08-06 ~13:52 ET  
Question: once β=0.5 finishes, can we run three concurrent instances on the remaining β=0.9 work to finish in ~⅓ the time?

## Short answer

**Yes in spirit — but split by remaining cells (layer window × reconstruction), not by carving one cell’s 1500 samples into three groups.** Cell-level split is a small launcher change using existing env knobs. Sample-level sharding is not a slight change: three processes would race the same result directory / checkpoint unless you add shard outputs and a merge step.

Expect **closer to ~2× wall-clock than a clean 3×**, because three Qwen forwards on one H100 contend for the same GPU. VRAM is the easy part.

## Why not “split the remaining samples into three groups”?

`run_eval.py` has `--subset_ids_file` (you could make three ID lists), but each steered config writes one stem under

`evaluation/results/2026-08-05/qwen2.5-vl-7b-instruct/amber/<cell_stem>/`

with a single `responses.checkpoint.json`. Three writers on one stem corrupt or clobber each other. Fixing that means shard dirs + merge (or new runner flags). That is real engineering, not a slight script tweak.

## What *is* a slight change

The grid script already accepts per-process:

- `BETAS=0.9`
- `LAYER_SETS=...`
- `STEER_RECONSTRUCTIONS=...`
- `--skip_if_exists` (checkpoint resume)

When only β=0.9 remains, **kill/replace the single β=0.9 worker** with three workers that own **disjoint cells**, e.g.:

| Worker | Owns |
|--------|------|
| A | `raw_mean_difference` + `LAYER_SETS=5-14` (resumes in-flight checkpoint) |
| B | `raw_mean_difference` + `LAYER_SETS=15-24` |
| C | `live_pc1_plus_mean` + `LAYER_SETS=all 5-14 15-24` |

Must stop the current β=0.9 process first: it loops all remaining cells and would race the new workers on the same stems.

Do **not** leave the old β=0.9 worker running while attaching two more for “the rest.”

## VRAM / rate (facts from the live job)

At check: two Qwen `run_eval` processes ≈ **17 GiB and 20 GiB**. Three at that footprint ≈ **51–60 GiB** on an 80 GiB H100 — fits with `MAX_PIXELS=1003520`, with the usual load stagger.

Observed shared-GPU rate with two processes: ~0.58–0.64 samples/s each. Three-way will be slower per process; aggregate throughput goes up, but not linearly to 3×.

## Timing intuition (order-of-magnitude only)

After β=0.5 exits, β=0.9 still has on the order of **~5 cell-equivalents** (one partial meandiff `5-14` + meandiff `15-24` + three PCA cells). One process at ~0.7–0.9/s solo ≈ a few hours. Three cell-partitioned processes might cut that to roughly **~1.5–2 h** rather than a strict third — still a real win vs leaving one worker, not a guarantee of finishing before a meeting that is already inside the next 90 minutes unless you cut over immediately and rates stay healthy.

## Operational sequence (if you want this)

1. Wait until β=0.5 `WORKER_OK` (or kill it only if you accept abandoning its last minutes — it was nearly done at last check).
2. Delete `amber-qwen-beta-triple` (or otherwise stop both workers).
3. Resubmit a β=0.9-only triple launcher: three concurrent workers, disjoint `LAYER_SETS` / `STEER_RECONSTRUCTIONS`, same `RUN_DATE`, `--skip_if_exists`, load stagger, quote-safe NFS patch if needed.
4. Confirm three PIDs + three growing logs + three python processes on the GPU.

No change to direction artifacts or the 1500 pin; only scheduling of which process owns which remaining cell.
