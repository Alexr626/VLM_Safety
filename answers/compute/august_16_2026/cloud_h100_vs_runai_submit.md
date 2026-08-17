# Cloud H100 site vs Nokia RunAI submit

Date: 2026-08-16. Infrastructure only. `config/sites.md` `cloud-h100` row is still a stub.

## What RunAI actually was (three pieces)

From `IMPLEMENTATION.md` § RunAI and `readme.md` § RunAI:

1. **Persistent storage (NFS)** — micromamba env `envs/vlm_hal/`, `hf_cache/`, extracted repo, COCO, results. Survives jobs.
2. **Disposable GPU** — one H100 pod, image `llm_image14:0.1`, NFS mounted at `/home/datalake`.
3. **Submit CLI on a login node** — `runai training submit` from lambdab2. Code reached the cluster as `git archive` → `vti_repo.tar.gz` → extract in the job (`helper_scripts/runai/sync_steering_llava.sh`, `update_repo_from_tarball.sh`).

Eval drivers (`run_vti.sh`, `run_steering_visual_reasoning_*.sh`, `SKIP_CONDA_ACTIVATE=1`) are mostly “activate env, run `run_eval.py`”. The Nokia-specific part is the submit command, the NFS paths, and the container image.

## What already exists in this repo for a second site

- Site table: `config/sites.md` (`personal-workstation` filled; `cloud-h100` stub).
- Path remap: `helper_scripts/runai/remap_lambdab2_paths.py` (`--relative` for tracked JSON; absolute `--apply` for gitignored manifests).
- Pack/sync scripts under `helper_scripts/runai/` that assume RunAI CLI + Nokia NFS. Do not retarget those in place until a vendor is chosen; they still encode `nlm-mh`, `h100-pool`, and `/home/datalake/romanus`.

## Two shapes that map onto that stack

**Always-on VM (SSH).** Closest to lambdab2: persistent disk holds env, weights, COCO, clone; you rsync or `git pull` and run the existing shell drivers. No `runai training submit`. Vendors that sell this: Lambda Cloud instances, RunPod/Vast GPU pods, a CoreWeave VM.

**Job + persistent volume (submit).** Closest to RunAI: you keep a volume for env/weights/results and submit a container that mounts it. Needs a scheduler API (RunPod serverless, Lambda clusters, GCP/AWS batch, etc.) and a replacement for the Nokia image (public `nvidia/cuda` + micromamba, or the NFS env if the volume is attached).

Either way the 5080 stays the qualitative-gate GPU; `cloud-h100` is for pinned-subset / multi-cell grids.

## What would be written in the repo after a vendor is chosen

1. Fill `config/sites.md` `cloud-h100` (vendor, SSH/API, disk paths, `HF_HOME`, GPU type).
2. Update `IMPLEMENTATION.md` compute section with the live submit command (same shape as the RunAI table).
3. Add a **new** helper directory (e.g. `helper_scripts/cloud_h100/`) rather than rewriting `helper_scripts/runai/` — keep Nokia scripts as historical.
4. One-time bootstrap: env from `environment.yml`, COCO at the site path, LLaVA/Qwen weights in `HF_HOME`.
5. Code ship: `git archive` or `rsync` of this checkout; extract must not wipe `data/coco/` or the env (same constraint as RunAI).
6. Remap gitignored manifests to the new prefix; do not re-`--relative` tracked files.

## Choice that is still Alex’s

Vendor logo is secondary to: always-on VM vs submit-a-job, and whether the disk is persistent across stop/start. Until that is picked, `cloud-h100` stays a stub and jobs stay on the 5080 at small N.
