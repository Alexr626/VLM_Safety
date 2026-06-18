#!/usr/bin/env bash
# RunAI run, executed inside a pod that mounts the NFS at /home/datalake.
# Activates the NFS-resident env (built by setup_vlm.sh) and runs the VTI POPE
# eval on LLaVA-1.5-7B. Results land under the repo's evaluation/results/ on the
# NFS, visible in WinSCP.
#
# Invoked by a RunAI job after setup_vlm.sh (see readme.md, RunAI section).
# Or locally: python3 helper_scripts/runai/run_bash_lf.py helper_scripts/runai/run_vti.sh
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BASE=/home/datalake/romanus
REPO=$BASE/vlm_hallucination
ENV_PREFIX=$BASE/envs/vlm_hal
MM=$BASE/bin/micromamba
export MAMBA_ROOT_PREFIX=$BASE/mamba
export HF_HOME=$BASE/hf_cache
export CUDA_VISIBLE_DEVICES=0   # RunAI exposes the single allocated GPU as device 0

cd "$REPO"
"$MM" run -p "$ENV_PREFIX" bash evaluation/run_scripts/run_vti_pope_llava.sh
