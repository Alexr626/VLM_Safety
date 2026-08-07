#!/usr/bin/env bash
# RunAI smoke: Qwen AMBER discriminative — baseline + meandiff + pc1_plus_mean
# on the 5-id smoke pin. Prints SMOKE_OK on success.
set -eu
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

BASE=/home/datalake/romanus
REPO=$BASE/vlm_hallucination
ENV_PREFIX=$BASE/envs/vlm_hal
MM=$BASE/bin/micromamba
export MAMBA_ROOT_PREFIX=$BASE/mamba
export HF_HOME=$BASE/hf_cache
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export SKIP_CONDA_ACTIVATE=1
export MODEL_SHORTS="${MODEL_SHORTS:-qwen2.5-vl-7b-instruct}"

JOB_NAME="${JOB_NAME:-amber-qwen-smoke}"
# shellcheck source=runai_job_logging.sh
source "$REPO/helper_scripts/runai/runai_job_logging.sh"
runai_job_logging_start

VERIFY="$REPO/helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh"
python3 - "$VERIFY" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
PY
bash "$VERIFY"

SMOKE_TAG="${SMOKE_TAG:-runai_amber_expanded_qwen_smoke_2026-08-06}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SUBSET="data/amber/pinned_amber_disc_1500_smoke5.json"
DIR_MD="experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd500_s42_meandiff_partition"
DIR_PCA="experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd500_s42_r2_partition"

cd "$REPO"

echo "=== remap ==="
if [[ -f helper_scripts/runai/remap_lambdab2_paths.py ]]; then
  "$MM" run -p "$ENV_PREFIX" python helper_scripts/runai/remap_lambdab2_paths.py --apply || true
fi

echo "=== prefetch HF weights ==="
"$MM" run -p "$ENV_PREFIX" python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
nvidia-smi || true

echo "=== baseline ==="
"$MM" run -p "$ENV_PREFIX" python evaluation/run_eval.py \
  --model "$MODEL" --benchmarks amber --amber_task discriminative \
  --interventions no_intervention \
  --subset_ids_file "$SUBSET" \
  --max_new_tokens 256 --max_pixels "$MAX_PIXELS" \
  --run_date "$SMOKE_TAG" --output_dir evaluation/results --skip_if_exists

echo "=== meandiff β=0.2 layers=all ==="
"$MM" run -p "$ENV_PREFIX" python evaluation/run_eval.py \
  --model "$MODEL" --benchmarks amber --amber_task discriminative \
  --interventions vti_textual_additive_mlp \
  --beta 0.2 --layer_set all --directions_dir "$DIR_MD" \
  --subset_ids_file "$SUBSET" \
  --max_new_tokens 256 --max_pixels "$MAX_PIXELS" \
  --run_date "$SMOKE_TAG" --output_dir evaluation/results --skip_if_exists

echo "=== pc1_plus_mean β=0.2 layers=all ==="
"$MM" run -p "$ENV_PREFIX" python evaluation/run_eval.py \
  --model "$MODEL" --benchmarks amber --amber_task discriminative \
  --interventions vti_textual_additive_mlp \
  --beta 0.2 --layer_set all --directions_dir "$DIR_PCA" \
  --subset_ids_file "$SUBSET" \
  --max_new_tokens 256 --max_pixels "$MAX_PIXELS" \
  --run_date "$SMOKE_TAG" --output_dir evaluation/results --skip_if_exists

echo "=== smoke outputs ==="
find "evaluation/results/$SMOKE_TAG" -name metric_summary.json -print
# Require distinct meandiff vs pc1_plus_mean dirs (gap-1 naming).
test -f "evaluation/results/$SMOKE_TAG/qwen2.5-vl-7b-instruct/amber/vti_textual_additive_mlp__b0.2__dall__nd500__meandiff__layers_all/metric_summary.json"
test -f "evaluation/results/$SMOKE_TAG/qwen2.5-vl-7b-instruct/amber/vti_textual_additive_mlp__b0.2__dall__nd500__pc1_plus_mean__layers_all/metric_summary.json"

echo "SMOKE_OK"
echo "  durable_log=$RUNAI_JOB_LOG"
