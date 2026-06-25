#!/usr/bin/env bash
# Minimal RunAI end-to-end smoke test: 5 POPE samples, no intervention, one split.
# Verifies NFS env, model weights, eval harness, and result writes.
#
# Invoked by a RunAI job after setup_vlm.sh (see readme.md, RunAI section).
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BASE=/home/datalake/romanus
REPO=$BASE/vlm_hallucination
ENV_PREFIX=$BASE/envs/vlm_hal
MM=$BASE/bin/micromamba
export MAMBA_ROOT_PREFIX=$BASE/mamba
export HF_HOME=$BASE/hf_cache
export CUDA_VISIBLE_DEVICES=0

RUN_TAG="${RUN_TAG:-runai_smoke}"
LIMIT="${LIMIT:-5}"

cd "$REPO"
echo "=== RunAI smoke test ==="
echo "  limit          : $LIMIT"
echo "  output_tag     : $RUN_TAG"
echo "  intervention   : no_intervention"
echo "  pope_split     : random"
echo "  model          : llava-hf/llava-1.5-7b-hf"

"$MM" run -p "$ENV_PREFIX" python evaluation/run_eval.py \
  --model llava-hf/llava-1.5-7b-hf \
  --benchmarks pope \
  --pope_split random \
  --interventions no_intervention \
  --limit "$LIMIT" \
  --output_dir "evaluation/results/$RUN_TAG"

OUT="$REPO/evaluation/results/$RUN_TAG/llava-1.5-7b-hf/pope/no_intervention"
echo ""
echo "=== Smoke test complete ==="
echo "Results:"
ls -la "$OUT/" 2>&1 || true
if [[ -f "$OUT/metric_summary.json" ]]; then
  echo "--- metric_summary.json ---"
  cat "$OUT/metric_summary.json"
fi
