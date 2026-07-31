# NFS placement: AMBER images + LLaVA demos_v2 directions

Stage **into the extracted repo tree** on NFS (WinSCP binary mode).

| What | Local (lambdab2) | NFS (SFTP) | In-pod path |
|------|------------------|------------|-------------|
| AMBER images | `data/amber/images/` (~408MB) | `/airl-datalake/romanus/vlm_hallucination/data/amber/images/` | `/home/datalake/romanus/vlm_hallucination/data/amber/images/` |
| LLaVA nd200 dirs | `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/` | `/airl-datalake/romanus/vlm_hallucination/experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demosv2_9a44f4af_all_nd200_s42_r2_prefix/` | same under `/home/datalake/romanus/…` |

**Order:** extract `vti_repo.tar.gz` into `vlm_hallucination/` first, then upload these (or upload after extract). Do **not** `rm -rf` the repo afterward or you wipe the staged images/dirs.

`run_perception_s3_smoke.sh` checks exactly those relative paths under `$REPO`.
