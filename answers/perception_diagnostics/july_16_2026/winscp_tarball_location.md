# Why WinSCP doesn’t show `vti_repo.tar.gz`

The file **exists on lambdab2** at:

`/home/romanus/dev/vti_repo.tar.gz` (~103 MB, owned by `romanus`)

It is **not** inside the git repo folder. WinSCP often opens:

- `…/vlm_hallucination_mitigation_summer_2026/` ← wrong (one level too deep)
- or the **NFS** session (`gpustorage-1` / `air-datalake`) ← different machine; tarball is not there until you upload it

## Fix

1. Open WinSCP session to **lambdab2** (not NFS).
2. Navigate to `/home/romanus/dev/` (parent of the repo).
3. You should see `vti_repo.tar.gz` next to `vlm_hallucination_mitigation_summer_2026/`.
4. Drag that file to NFS `/airl-datalake/romanus/vti_repo.tar.gz`.

On lambdab2 shell: `ls -lh ~/dev/vti_repo.tar.gz`
