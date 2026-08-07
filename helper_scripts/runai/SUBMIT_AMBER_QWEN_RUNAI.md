# RunAI: AMBER-1500 Qwen β-triple on one H100 (lambdab2 submit)

Project: `nlm-mh`. Always: `--gpu-devices-request 1 --node-pools h100-pool`.

**Important:** keep each `python3 …/run_bash_lf.py …/script.sh` on **one line**
inside `bash -c` (no `\` line break).

Do **not** kill lambdab2 Qwen until step 4 (`SMOKE_OK`).

```bash
export PATH="$HOME/.runai/bin:$PATH"
IMG=blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1
NFS='path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite'
```

## 0. On lambdab2 — pack (agent can run this)

```bash
bash helper_scripts/runai/pack_amber_expanded_qwen_for_runai.sh
# → /tmp/runai_amber_qwen_2026-08-06/
```

## 1. WinSCP (ThinkPad) — binary mode → NFS `/airl-datalake/romanus/`

From `/tmp/runai_amber_qwen_2026-08-06/`:

- `vti_repo_amber_qwen.tar.gz`
- `qwen_amber_directions.tar.gz`
- `amber_images.tar.gz`
- `qwen_amber_partial_results.tar.gz`

## 2. Login + quota check

```bash
runai login
runai project set nlm-mh
runai workload list -p nlm-mh
runai project list
```

Need **1 free GPU** in `nlm-mh`.

## 3. Sync

```bash
runai training delete sync-amber-qwen -p nlm-mh 2>/dev/null || true
runai training submit sync-amber-qwen -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/sync_amber_expanded_qwen.sh'
```

Wait for `SYNC_OK` / `VERIFY_AMBER_QWEN_OK` in `runai training logs sync-amber-qwen -p nlm-mh` or NFS `logs/runai/sync-amber-qwen_*.log`.

**First sync chicken-egg:** if `vlm_hallucination/` lacks the new sync script, extract once manually in the submit command:

```bash
runai training submit sync-amber-qwen -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; mkdir -p $REPO; tar -xzf $BASE/vti_repo_amber_qwen.tar.gz -C $REPO; tar -xzf $BASE/qwen_amber_directions.tar.gz -C $REPO; tar -xzf $BASE/amber_images.tar.gz -C $REPO; [[ -f $BASE/qwen_amber_partial_results.tar.gz ]] && tar -xzf $BASE/qwen_amber_partial_results.tar.gz -C $REPO; python3 $REPO/helper_scripts/runai/run_bash_lf.py $REPO/helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh'
```

## 4. Smoke — do not kill lambdab2 Qwen until `SMOKE_OK`

```bash
runai training delete amber-qwen-smoke -p nlm-mh 2>/dev/null || true
runai training submit amber-qwen-smoke -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_amber_expanded_qwen_smoke.sh'
```

Expect `SMOKE_OK` and both `meandiff` and `pc1_plus_mean` result dirs.

## 5. Kill lambdab2 Qwen + babysitter; re-pack partial; re-sync overlay

```bash
# stop babysitter first so it does not restart Qwen
kill 189455 2>/dev/null || true
# Qwen driver (and its run_eval child)
pkill -f 'run_amber_expanded_steering_direction_grid.sh Qwen' || true

# re-pack latest partial (includes checkpoint) and WinSCP only:
#   qwen_amber_partial_results.tar.gz
bash helper_scripts/runai/pack_amber_expanded_qwen_for_runai.sh
# WinSCP the updated partial tarball, then:
runai training delete sync-amber-qwen-partial -p nlm-mh 2>/dev/null || true
runai training submit sync-amber-qwen-partial -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; BASE=/home/datalake/romanus; REPO=$BASE/vlm_hallucination; tar -xzf $BASE/qwen_amber_partial_results.tar.gz -C $REPO; echo PARTIAL_SYNC_OK'
```

Leave **LLaVA** running on GPU 0.

## 6. β-triple grid

**Concurrency bug (fixed 2026-08-06):** do not capture worker PIDs with `PID=$(launch …)`.
Command substitution waits until the child's stdout closes, which serializes the three
β-workers. The launcher now starts each worker with its own redirected log and takes `$!`
in the parent shell. Per-β durable logs: `logs/runai/amber-qwen-beta-{0.2,0.5,0.9}_*.log`.

```bash
runai training delete amber-qwen-beta-triple -p nlm-mh 2>/dev/null || true
runai training submit amber-qwen-beta-triple -p nlm-mh \
  --nfs "$NFS" -i "$IMG" \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; python3 /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_bash_lf.py /home/datalake/romanus/vlm_hallucination/helper_scripts/runai/run_amber_expanded_qwen_beta_triple_one_h100.sh'
```

If NFS still has the pre-fix launcher, either WinSCP the updated script (or
`/tmp/runai_amber_qwen_launcher_fix_2026-08-06/amber_qwen_launcher_fix.tar.gz` → extract into
`vlm_hallucination/helper_scripts/runai/`), or patch at submit time by writing the script
bytes before `run_bash_lf.py` (base64 decode into that path).

Expect `AMBER_QWEN_TRIPLE_OK`. Parent log: `logs/runai/amber-qwen-beta-triple_*.log`.
Confirm concurrency: three `start Qwen β=` lines within ~2×`LOAD_STAGGER_SEC`, then three
worker logs growing in parallel.

## 6b. Auto cutover → β=0.9 cell triple (after β=0.5 finishes)

When β=0.5’s six steered cells all have `responses.json`, delete the β-split job and
submit a **cell-partitioned** β=0.9 triple (three Qwen processes on one H100):

| Worker | Env |
|--------|-----|
| meandiff-5-14 | `BETAS=0.9` `STEER_RECONSTRUCTIONS=raw_mean_difference` `LAYER_SETS=5-14` |
| meandiff-15-24 | `BETAS=0.9` `STEER_RECONSTRUCTIONS=raw_mean_difference` `LAYER_SETS=15-24` |
| pca-windows | `BETAS=0.9` `STEER_RECONSTRUCTIONS=live_pc1_plus_mean` `LAYER_SETS="all 5-14 15-24"` |

Launcher: `helper_scripts/runai/run_amber_expanded_qwen_beta09_cell_triple_one_h100.sh`  
Babysitter (lambdab2): `helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py`

```bash
# already started under nohup; re-start if needed:
nohup python3 helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py \
  --poll_sec 45 --log logs/babysit_amber_qwen_cutover_beta09_triple.log \
  >> logs/babysit_amber_qwen_cutover_beta09_triple.stdout 2>&1 &

# status only:
python3 helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py --check_once

# emergency immediate cutover (abandons unfinished β=0.5 cells):
python3 helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py --force_cutover_now
```

Expect `LAUNCHER_PATCHED_BETA09` then `AMBER_QWEN_BETA09_TRIPLE_OK`. Job name:
`amber-qwen-beta09-triple`. Marker: `logs/amber_qwen_cutover_beta09_done.json`.

## 7. Pull results (after finish)

WinSCP from NFS:

`vlm_hallucination/evaluation/results/2026-08-05/qwen2.5-vl-7b-instruct/`

→ lambdab2 same path (merge into the existing tree). Then on lambdab2:

```bash
python evaluation/steering_vector_validation_continuation/build_per_configuration_and_per_item_tables.py --run_date 2026-08-05
python helper_scripts/verify_amber_expanded_steering_direction_grid_run.py --run_date 2026-08-05 --pre_hashes evaluation/results/2026-08-05/_analysis_steering_vector_validation_continuation/pre_launch_hashes.json
```
