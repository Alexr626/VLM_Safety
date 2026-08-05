# What is the point of the RunAI sync step?

Date: 2026-07-31

WinSCP only drops tarballs onto NFS (e.g. `/airl-datalake/romanus/*.tar.gz`).
Pods do not run from those archives as-is.

The sync job is a short 1-GPU training workload that mounts the same NFS and
**extracts** the tarballs into the live tree pods use:

- `vti_repo_working_tree.tar.gz` → `/home/datalake/romanus/vlm_hallucination/`
- `llava_meandiff_directions.tar.gz` → meandiff direction dirs under that repo
- optional partial CHAIR results → so `--skip_if_exists` can skip finished cells

It is non-destructive relative to wiping the whole NFS repo: it overlays code and
artifacts, and leaves `envs/vlm_hal`, `hf_cache`, and prior results alone
(unless a path in the tarball overwrites the same relative path).

Without sync, smoke/full jobs would run against whatever was already extracted
on NFS (stale or missing meandiff / `SKIP_CONDA_ACTIVATE` / smoke script).
