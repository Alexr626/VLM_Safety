#!/usr/bin/env bash
# setup_env.sh — reproducible environment setup for VLM Safety experiments
#
# Usage:
#   bash setup_env.sh            # creates or recreates the vlm_safety env
#   bash setup_env.sh --update   # updates an existing env in-place
#
# Why this script instead of plain `conda env create`:
#   flash-attn requires compilation against the installed torch headers.
#   It cannot reliably be built inside `conda env create` because the
#   CUDA/torch headers are not yet on PATH during that phase.
#   This script installs everything else first, then builds flash-attn
#   against the finalized torch install.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="vlm_safety"
FLASH_ATTN_VERSION="2.8.3"
CUDA_WHEEL="cu124"   # change to cu128 for CUDA 12.8 wheels

# ── Parse args ────────────────────────────────────────────────────────────────
UPDATE_MODE=false
if [[ "${1:-}" == "--update" ]]; then
    UPDATE_MODE=true
fi

# ── Step 1: Create or update the conda environment ───────────────────────────
if $UPDATE_MODE; then
    echo "[1/3] Updating existing environment: $ENV_NAME"
    conda env update --name "$ENV_NAME" --file environment.yml --prune
else
    echo "[1/3] Creating environment: $ENV_NAME"
    # Remove existing env if present (ensures a clean, reproducible build)
    if conda info --envs | grep -q "^${ENV_NAME}\b"; then
        echo "      Removing existing environment first..."
        conda env remove --name "$ENV_NAME" --yes
    fi
    conda env create --file environment.yml
fi

# ── Step 2: Verify torch can see CUDA before building flash-attn ──────────────
echo "[2/3] Verifying CUDA visibility..."
CUDA_OK=$(conda run -n "$ENV_NAME" python -c "
import torch
print(torch.cuda.is_available())
print(torch.__version__)
" 2>&1)
echo "      torch output: $CUDA_OK"
if echo "$CUDA_OK" | grep -q "False"; then
    echo ""
    echo "WARNING: torch.cuda.is_available() returned False."
    echo "  This likely means the CUDA driver is not visible in this shell."
    echo "  flash-attn will still be installed (it only needs the toolkit,"
    echo "  not a live GPU), but verify CUDA availability before running"
    echo "  experiments."
fi

# ── Step 3: Install flash-attn against the finalized torch install ────────────
echo "[3/3] Installing flash-attn==$FLASH_ATTN_VERSION..."
echo "      This step compiles C++ extensions and may take 5–15 minutes."
echo "      If a prebuilt wheel is available for your torch/CUDA combo it"
echo "      will be used automatically and take ~30 seconds."
echo ""

conda run -n "$ENV_NAME" pip install \
    flash-attn=="${FLASH_ATTN_VERSION}" \
    --no-build-isolation \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# ── Verify flash-attn import ──────────────────────────────────────────────────
echo ""
echo "Verifying flash-attn import..."
conda run -n "$ENV_NAME" python -c "
import flash_attn
print(f'flash-attn {flash_attn.__version__} imported successfully')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================="
echo " Environment '$ENV_NAME' is ready."
echo " Activate with: conda activate $ENV_NAME"
echo ""
echo " Package summary:"
conda run -n "$ENV_NAME" python -c "
import torch, transformers, flash_attn, accelerate, einops, timm
print(f'  python:        $(python --version 2>&1 || true)')
print(f'  torch:         {torch.__version__}')
print(f'  transformers:  {transformers.__version__}')
print(f'  flash-attn:    {flash_attn.__version__}')
print(f'  accelerate:    {accelerate.__version__}')
print(f'  einops:        {einops.__version__}')
print(f'  timm:          {timm.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:  {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // 1024**3} GB)')
"
echo "========================================================="
