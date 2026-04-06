#!/bin/bash
# ============================================================
# Diagnostic experiments for a 16 GB GPU (e.g. RTX 5080)
# ============================================================
# Runs:
#   - ALL phases for LLaVA-1.5-7B (fits comfortably in 16 GB)
#   - Phases 0, 1, 3, 4 for Qwen2.5-VL-7B (skips Phase 2:
#     generation requires more VRAM due to KV cache + CPU offload)
#
# Phase 2 for Qwen and ALL phases for InternVL2/2.5 should be
# run on a larger GPU — see run_diag_24gb.sh.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Helper: run a full diagnostic pipeline for a given model ──
run_all_phases() {
    local MODEL="$1"
    echo ""
    echo "============================================================"
    echo "  Running ALL phases for: $MODEL"
    echo "============================================================"
    MODEL="$MODEL" bash "$SCRIPT_DIR/run_all_diagnostics.sh"
}

# ── Helper: run phases 0, 1, 3, 4 (skip Phase 2) ─────────────
run_skip_phase2() {
    local MODEL="$1"
    echo ""
    echo "============================================================"
    echo "  Running Phases 0, 1, 3, 4 (skipping Phase 2) for: $MODEL"
    echo "============================================================"

    echo "=== Phase 0: Data extraction ==="
    python "$PROJECT_ROOT/data_scripts/extract_vl.py"  --model "$MODEL" --dataset holisafe
    python "$PROJECT_ROOT/data_scripts/extract_tt.py"  --model "$MODEL" --dataset holisafe
    python "$PROJECT_ROOT/data_scripts/extract_ref_activations.py" --model "$MODEL" \
        --safe_ref catqa-harmless --unsafe_ref catqa-harmful

    echo "=== Phase 1: ShiftDC diagnostic ==="
    python "$SCRIPT_DIR/experiment_scripts/vl_activation_shift.py" --model "$MODEL"
    python "$SCRIPT_DIR/experiment_scripts/sanity_check_tt_baseline.py" --model "$MODEL"
    python "$SCRIPT_DIR/plotting_scripts/plot_vl_activation_shift_projections.py" --model "$MODEL"
    python "$SCRIPT_DIR/plotting_scripts/plot_tt_baseline_projections.py" --model "$MODEL"

    echo "=== Skipping Phase 2 (generation) — run on larger GPU ==="

    echo "=== Phase 3: Combinatorial safety ==="
    python "$SCRIPT_DIR/experiment_scripts/combinatorial_direction.py" --model "$MODEL"
    python "$SCRIPT_DIR/experiment_scripts/safety_probes.py" --model "$MODEL"
    python "$SCRIPT_DIR/plotting_scripts/plot_direction_comparison.py" --model "$MODEL"
    python "$SCRIPT_DIR/plotting_scripts/plot_probe_results.py" --model "$MODEL"

    echo "=== Phase 4: ShiftDC with combinatorial direction ==="
    python "$SCRIPT_DIR/experiment_scripts/vl_activation_shift.py" --model "$MODEL" --combinatorial_dir
    python "$SCRIPT_DIR/experiment_scripts/sanity_check_tt_baseline.py" --model "$MODEL" --combinatorial_dir
    python "$SCRIPT_DIR/plotting_scripts/plot_combinatorial_shift_projections.py" --model "$MODEL"

    echo "=== Done (Phases 0,1,3,4): $MODEL ==="
}

# ── LLaVA 1.5 — all phases ───────────────────────────────────
run_all_phases "llava-hf/llava-1.5-7b-hf"

# ── Qwen 2.5 VL — skip Phase 2 ──────────────────────────────
run_skip_phase2 "Qwen/Qwen2.5-VL-7B-Instruct"

echo ""
echo "============================================================"
echo "  16 GB run complete."
echo "  Remaining: run run_diag_24gb.sh on a larger GPU."
echo "============================================================"
