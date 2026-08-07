# RunAI: finish AMBER-1500 Qwen grid as 3 concurrent β-workers on one H100

Date: 2026-08-06  
Context: overnight lambdab2 run in progress; Qwen too slow for meeting deadline.  
Prior chat lessons: [Steering vector eval watcher](46a16eec-8ea2-4f73-9687-d19256b09419).

## Do not kill lambdab2 Qwen until RunAI smoke prints `SMOKE_OK`

Keep LLaVA on lambdab2 GPU 0. Kill Qwen only after the RunAI smoke succeeds, then immediately re-pack/sync the latest Qwen partial tree (including any `responses.checkpoint.json`) so `--skip_if_exists` resumes the in-flight cell.

## Recommended split: by β (not by layer window)

Remaining Qwen work (as of morning 2026-08-05 tree): **15 cells** — meandiff `5-14`/`15-24` × 3 β, plus full PCA block (3 windows × 3 β). Mean-diff `layers_all` and baseline are already done.

Three workers on **one** H100, each with a single β:

| Worker | Env | Cells it owns (after skip) |
|--------|-----|----------------------------|
| β=0.2 | `BETAS=0.2` | meandiff 5-14, 15-24; PCA all / 5-14 / 15-24 |
| β=0.5 | `BETAS=0.5` | same pattern |
| β=0.9 | `BETAS=0.9` | same pattern |

Each worker still loops both `STEER_RECONSTRUCTIONS` and all three `LAYER_SETS`; `--skip_if_exists` drops completed cells (e.g. meandiff `layers_all`). Work is ~5 cells/worker — balanced.

Splitting by layer window is worse: the `all` worker only has PCA left while the others carry meandiff + PCA.

This matches the prior RunAI pattern you insisted on in the watcher chat: **one training job, one GPU, three concurrent processes** (`run_steering_triple_one_h100.sh`), not three separate GPU jobs.

## What worked last time (carry forward)

From `helper_scripts/runai/SUBMIT_STEERING_LLAVA.md` + the watcher chat:

1. **Image:** `llm_image14:0.1` (not `dspy_image2`).
2. **Always** `--gpu-devices-request 1 --node-pools h100-pool`.
3. **Submit `bash -c` with `run_bash_lf.py` on one line** — a `\` break caused `IndexError` / empty argv.
4. **Sequence:** pack on lambdab2 → WinSCP binary tarballs to NFS → **sync** → **smoke** → full job.
5. **`SKIP_CONDA_ACTIVATE=1`** + micromamba `envs/vlm_hal` (pods have no conda).
6. **`LOAD_STAGGER_SEC` ~45** between process starts so three weight loads do not spike VRAM together.
7. **Durable logs:** `/home/datalake/romanus/logs/runai/` (NFS).
8. **Gitignored overlays must be packed:** pins / `combined.json` / partial results — `git archive` alone is not enough (this bit you last time on CHAIR).
9. **Partial results tarball** so `--skip_if_exists` resumes.
10. **Qwen:** always `MAX_PIXELS=1003520`.

VRAM: prior triple was LLaVA+Qwen+Qwen on one H100 and finished (`TRIPLE_OK`). Three Qwen-7B (~15–16 GiB each) should fit 80 GiB with headroom; stagger loads. Host RAM: three eager AMBER-1500 loads ≈ ~38 GB — usually fine on the node; watch smoke.

## New pieces needed (old triple scripts are CHAIR/POPE-shaped)

Do **not** reuse `run_steering_triple_one_h100.sh` as-is. Add thin wrappers that call the existing driver:

- `helper_scripts/runai/run_amber_expanded_qwen_worker.sh` — sets `SKIP_CONDA_ACTIVATE=1`, remaps paths once if needed, runs  
  `evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh Qwen/Qwen2.5-VL-7B-Instruct`  
  with env `BETAS`, `RUN_DATE=2026-08-05`, `MAX_PIXELS=1003520`, `AMBER_SUBSET=data/amber/pinned_amber_disc_1500.json`.
- `helper_scripts/runai/run_amber_expanded_qwen_beta_triple_one_h100.sh` — parent: verify layout → remap once → prefetch Qwen weights once → start three workers with `BETAS=0.2|0.5|0.9`, staggered, wait, print `AMBER_QWEN_TRIPLE_OK` / `FAIL`.
- `helper_scripts/runai/run_amber_expanded_qwen_smoke.sh` — AMBER discriminative, 5-id smoke pin (or `pinned_amber_disc_1500_smoke5.json`), baseline + one meandiff + one `pc1_plus_mean` cell, `SMOKE_OK`.
- Pack script sibling to `pack_steering_llava_for_runai.sh` that overlays at least:
  - updated `evaluation/runners/eval_runner.py` (recon stem),
  - `evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh`,
  - new RunAI helpers above,
  - `data/amber/pinned_amber_disc_1500.json` (+ smoke5),
  - `data/amber/images/` (~408 MB),
  - Qwen **both** `…_meandiff_partition` and `…_r2_partition` nd500 dirs (old pack was meandiff-only),
  - `evaluation/results/2026-08-05/qwen2.5-vl-7b-instruct/` partial tree.

Verify script for this job should require the 1500 pin, amber images, and the r2 direction dir — not only meandiff.

## Submit skeleton (after pack + WinSCP)

```bash
export PATH="$HOME/.runai/bin:$PATH"
IMG=blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1
NFS='path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite'

# 1) sync (extract tarballs onto NFS)
runai training delete sync-amber-qwen -p nlm-mh 2>/dev/null || true
runai training submit sync-amber-qwen -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p $REPO; tar -xzf $BASE/vti_repo_amber_qwen.tar.gz -C $REPO; tar -xzf $BASE/qwen_amber_directions.tar.gz -C $REPO; tar -xzf $BASE/qwen_amber_partial_results.tar.gz -C $REPO; python3 $REPO/helper_scripts/runai/run_bash_lf.py $REPO/helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh'

# 2) smoke — wait for SMOKE_OK in logs/runai/ before killing lambdab2 Qwen
runai training delete amber-qwen-smoke -p nlm-mh 2>/dev/null || true
runai training submit amber-qwen-smoke -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_amber_expanded_qwen_smoke.sh'

# 3) after SMOKE_OK: kill lambdab2 Qwen only; re-pack+sync partial; then
runai training delete amber-qwen-beta-triple -p nlm-mh 2>/dev/null || true
runai training submit amber-qwen-beta-triple -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_amber_expanded_qwen_beta_triple_one_h100.sh'
```

Monitor: `runai workload list -p nlm-mh`; NFS logs under `/airl-datalake/romanus/logs/runai/`.

## Kill protocol (only after smoke)

```bash
# on lambdab2 — Qwen driver + its run_eval only; leave LLaVA + babysitter
kill 189449   # Qwen bash driver (children die with it)
# then re-pack qwen partial including checkpoint, WinSCP, quick sync overlay, submit triple
```

Leave babysitter: it may try to restart Qwen. Either stop babysitter first, or let it restart and immediately kill again — cleaner to stop babysitter when killing Qwen.

## ETA sketch (H100, 3 concurrent Qwen)

Lambdab2 contended Qwen ≈ 0.29 samples/s. Uncontended H100 was faster in the prior triple. Rough: ~5 cells × 1500 / (~0.5–0.8/s concurrent share) ≈ a few hours for the triple after smoke — order-of-magnitude only; smoke rate is the real calibration.

## Status 2026-08-06 afternoon — launcher concurrency fix + resubmit

**Bug:** first `amber-qwen-beta-triple` run used `PID=$(launch_worker …)`. Bash command
substitution waits until the background worker closes stdout, so the three β-workers
ran **serially** (~3h between β=0.2 `WORKER_OK` and β=0.5 start). GPU sat empty between
workers.

**Fix:** `helper_scripts/runai/run_amber_expanded_qwen_beta_triple_one_h100.sh` now starts
each worker with its own redirected log, takes `$!` in the parent, unsets inherited
`RUNAI_JOB_LOG` in a subshell so each β gets its own durable log, then `wait`s all three.

**Actions taken:**
1. Patched launcher on lambdab2.
2. Deleted serial `amber-qwen-beta-triple`.
3. First resubmit used `python3 -c` + quoted paths; RunAI stripped the quotes → patch
   failed; job fell back to the old serial NFS launcher. Deleted again.
4. Resubmitted with quote-safe `echo <b64> | base64 -d > launcher.sh && …` (no string
   literals). Confirmed `LAUNCHER_PATCHED` + concurrency note in the patched file.
5. Tiny optional overlay also at `/tmp/runai_amber_qwen_launcher_fix_2026-08-06/`.

**Verified concurrent (pod exec ~13:47 ET):**
- Parent: all three workers launched (`pids` 1489 / 1912 / 2597), then `waiting`.
- β=0.2: `WORKER_OK` (skip_if_exists cleared remaining cells).
- β=0.5 + β=0.9: two `run_eval.py` on the H100 (~17 GiB each, ~34 GiB / 99% util).
- Per-β logs: `logs/runai/amber-qwen-beta-{0.2,0.5,0.9}_20260806T173926Z.log`.

Helpers + pack remain at `/tmp/runai_amber_qwen_2026-08-06/`.
Canonical submit sheet: `helper_scripts/runai/SUBMIT_AMBER_QWEN_RUNAI.md`.
