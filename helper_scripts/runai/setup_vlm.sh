#!/usr/bin/env bash
# One-time RunAI setup. Invoked from a RunAI job after tarball extract:
#   bash helper_scripts/runai/setup_vlm.sh
# dspy_image2:0.1 has no Python — pre-upload bin/micromamba to NFS (see HANDOFF.md).
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Fallback only if micromamba was not pre-staged on NFS:
PYTHON="${PYTHON:-$(command -v python3.8 || command -v python3 || command -v python)}"

BASE=/home/datalake/romanus
REPO=$BASE/vlm_hallucination
TARBALL=$BASE/vti_repo.tar.gz
ENV_PREFIX=$BASE/envs/vlm_hal
MM=$BASE/bin/micromamba
export MAMBA_ROOT_PREFIX=$BASE/mamba
export HF_HOME=$BASE/hf_cache
mkdir -p "$BASE/bin" "$HF_HOME"

echo "=== [1/4] extract repo ==="
if [ ! -d "$REPO/src" ]; then
  mkdir -p "$REPO"
  tar -xzf "$TARBALL" -C "$REPO"
fi
ls -la "$REPO" | head -n 20

echo "=== [2/4] bootstrap micromamba ==="
if [ ! -x "$MM" ]; then
  if [ -z "$PYTHON" ]; then
    echo "ERROR: $MM missing and no Python in container to bootstrap." >&2
    echo "Pre-upload micromamba to NFS: romanus/bin/micromamba (see HANDOFF.md)." >&2
    exit 1
  fi
  "$PYTHON" "$REPO/helper_scripts/runai/bootstrap_micromamba.py" "$BASE"
fi
"$MM" --version

echo "=== [3/4] build env from environment.yml (on NFS) ==="
if [ ! -d "$ENV_PREFIX" ]; then
  "$MM" create -y -p "$ENV_PREFIX" -f "$REPO/environment.yml"
fi
"$MM" run -p "$ENV_PREFIX" python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

echo "=== [4/4] download POPE images (COCO val2014) + LLaVA-1.5-7B weights ==="
"$MM" run -p "$ENV_PREFIX" bash -c "
  set -e
  cd '$REPO'
  export HF_HOME='$HF_HOME'
  python data_scripts/download_chair.py
  python -c \"from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llava-1.5-7b-hf')\"
"
echo "=== SETUP COMPLETE ==="
