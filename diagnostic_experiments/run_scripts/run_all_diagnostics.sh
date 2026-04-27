#!/bin/bash
# ============================================================
# Run full diagnostic pipeline for any supported model
# ============================================================
# Usage:
#   MODEL="OpenGVLab/InternVL2-8B"           bash run_all_diagnostics.sh
#   MODEL="OpenGVLab/InternVL2_5-8B-MPO"     bash run_all_diagnostics.sh
#   MODEL="Qwen/Qwen2-VL-7B"                 bash run_all_diagnostics.sh
#   MODEL="Qwen/Qwen2-VL-7B-Instruct"        bash run_all_diagnostics.sh
#   MODEL="llava-hf/llava-1.5-7b-hf"         bash run_all_diagnostics.sh
# ============================================================
set -e

MODEL="${MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
export MODEL
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"

echo "=== Running all diagnostics for: $MODEL ==="

echo "=== Phase 0: Data extraction ==="
python "$PROJECT_ROOT/data_scripts/extract_vl.py"  --model "$MODEL" --dataset holisafe
python "$PROJECT_ROOT/data_scripts/extract_tt.py"  --model "$MODEL" --dataset holisafe
python "$PROJECT_ROOT/data_scripts/extract_ref_activations.py" --model "$MODEL" \
    --safe_ref catqa-harmless --unsafe_ref catqa-harmful

echo "=== Phase 1: ShiftDC diagnostic ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/vl_activation_shift.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/experiment_scripts/sanity_check_tt_baseline.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_vl_activation_shift_projections.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_tt_baseline_projections.py" --model "$MODEL"

echo "=== Phase 2: Behavioral ground truth ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/generate_responses.py" --model "$MODEL" --skip_if_exists
python "$DIAGNOSTIC_ROOT/experiment_scripts/classify_responses.py" --model "$MODEL" \
    --method keyword --provider anthropic
python "$DIAGNOSTIC_ROOT/experiment_scripts/catqa_behavioral_baseline.py" --model "$MODEL" \
    --method keyword --provider anthropic --skip_if_exists
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_behavioral_ground_truth.py" --model "$MODEL"

echo "=== Phase 3: Combinatorial safety ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/combinatorial_direction.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/experiment_scripts/safety_probes.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_direction_comparison.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_probe_results.py" --model "$MODEL"

echo "=== Phase 4: ShiftDC with combinatorial direction ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/vl_activation_shift.py" --model "$MODEL" --combinatorial_dir
python "$DIAGNOSTIC_ROOT/experiment_scripts/sanity_check_tt_baseline.py" --model "$MODEL" --combinatorial_dir
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_combinatorial_shift_projections.py" --model "$MODEL"

echo "=== Done: $MODEL ==="
