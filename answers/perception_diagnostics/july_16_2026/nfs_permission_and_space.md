# NFS permission denied on AMBER upload + how to check space

## Why WinSCP “Permission denied” is likely

Not usually “out of disk” (that volume shows ~32T free on related mounts). Common causes for `/airl-datalake/romanus/vlm_hallucination/…`:

1. **Wrong path** — must be under `/airl-datalake/romanus/`. Outside that home → denied.
2. **Dirs owned by root / pod UID** — extract jobs write as the container user; SFTP user `air-datalake` may not be able to create/overwrite under `vlm_hallucination/data/amber/`.
3. **Missing parent** — `data/amber/` may not exist yet in the extracted tree (AMBER images were never in git).
4. Less often: Synology ACL / write lock on a file you’re overwriting.

**Workaround (recommended):** upload to a path you *can* write (your NFS home), then install via a RunAI job:

- WinSCP → `/airl-datalake/romanus/amber_images.tar` (or a folder `staging/amber_images/`)
- WinSCP → `/airl-datalake/romanus/llava_nd200_dirs.tar` (small)
- Submit a 1-GPU job that `mkdir -p` + `tar -xf` / `cp -a` into  
  `$REPO/data/amber/images/` and  
  `$REPO/experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/`

## How to check NFS free space

**WinSCP:** Session → Server/Protocol Information, or right-click a folder → Properties (capacity if Synology exposes it).

**From a RunAI pod** (authoritative for the mount pods use):

```bash
runai training submit pd-df -p nlm-mh \
  --nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite \
  -i blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1 \
  --gpu-devices-request 1 --node-pools h100-pool \
  --command -- bash -c 'export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; df -h /home/datalake; df -h /home/datalake/romanus; ls -la /home/datalake/romanus; ls -la /home/datalake/romanus/vlm_hallucination/data 2>&1 | head'
```

Then: `runai training logs pd-df -p nlm-mh`

**lambdab2 note:** mounts under `/nfs/datalake/*` are a *different* share and **read-only**; they are not the RunAI `airl-datalake` tree. Don’t use those `df` numbers as a substitute for the pod check above (though they currently show ~32T avail on that Synology).
