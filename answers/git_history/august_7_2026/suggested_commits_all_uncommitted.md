# Suggested commits — all currently uncommitted work (2026-08-07)

Scope: every modified and untracked path on `new_research_workflow`. Includes items previously left local (session transcripts, incomplete design stub, large SHA manifests, scan PDF).

Also force-adds `data/amber/pinned_amber_disc_1500.json` by extending `.gitignore` (pin is required by the AMBER expanded grid but was still ignored under `data/amber/*`).

## Suggested commits (12)

### 1. Tutor `/transcribe` harness
- `.claude/commands/transcribe.md`
- `.claude/agents/tutor.md`

### 2. Result-dir stem from `steer_reconstruction`
- `evaluation/runners/eval_runner.py`

### 3. AMBER-1500 pin + tooling
- `.gitignore` (`!data/amber/pinned_amber_disc_1500.json`)
- `data/amber/pinned_amber_disc_1500.json`
- `data_scripts/draw_amber_discriminative_1500.py`
- `helper_scripts/verify_amber_discriminative_1500_pin.py`

### 4. Continuation extraction, plan, reading
- `extractions/08_05_26/steering_vector_validation_continuation.md`
- `implementation_plans/8-5-26/amber_expanded_steering_direction_grid_plan_2026-08-05.md`
- `analysis/08_05_26/steering_vector_visual_reasoning_validation.md`

### 5. AMBER expanded grid drivers + analysis
- `evaluation/run_scripts/run_amber_expanded_steering_direction_grid.sh`
- `evaluation/run_scripts/babysit_amber_expanded_steering_direction_grid.py`
- `evaluation/steering_vector_validation_continuation/`
- `helper_scripts/verify_amber_expanded_steering_direction_grid_run.py`

### 6. Matched POPE 06-19 vs 07-30 comparison
- `evaluation/pope_0619_vs_0730_matched/`

### 7. Organized overnight plot trees
- `evaluation/steering_visual_reasoning_validation/make_organized_plots.py`

### 8. RunAI AMBER expanded Qwen helpers
- `helper_scripts/runai/SUBMIT_AMBER_QWEN_RUNAI.md`
- `helper_scripts/runai/pack_amber_expanded_qwen_for_runai.sh`
- `helper_scripts/runai/sync_amber_expanded_qwen.sh`
- `helper_scripts/runai/verify_amber_expanded_qwen_nfs_layout.sh`
- `helper_scripts/runai/run_amber_expanded_qwen_*.sh`
- `helper_scripts/runai/babysit_amber_qwen_cutover_beta09_triple.py`

### 9. Learning queue update + Agresti scan
- `learning/review_queue.md`
- `learning/scans/Statistical Inference for Discrete Data Review_08_05_26.pdf`
- `learning/scans/transcripts/Statistical Inference for Discrete Data Review_08_05_26.md`

### 10. Chat answers (concepts / runai / runs / git_history / transcripts)
- all currently untracked under `answers/`

### 11. Leftover design stub + perlayer SHA manifests
- `designs/PCA_visual_reasoning_validation_07_30_26.md`
- `diagnostic_experiments/perlayer_pca_control/verification/global_fit_directions_sha256_manifest_{pre,post}_run.json`

### 12. IMPLEMENTATION.md + RESEARCH_LOG.md
- `IMPLEMENTATION.md`
- `RESEARCH_LOG.md`
