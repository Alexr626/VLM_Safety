# Suggested commits for current unstaged work

Date: 2026-08-05. Working tree on `new_research_workflow` after `f351b45`. Nothing staged.

## Exclude (do not commit as-is)

| Path | Why |
|---|---|
| `hal-steer-triple_20260731T151200Z.log` | Run log at repo root (~136K); `logs/` is ignored — leave local or move under ignored `logs/` |
| `designs/PCA_visual_reasoning_validation_07_30_26.md` | Still mostly template placeholders (`<Explanation A>`, etc.); not a finished design |
| `answers/**/session_transcript_*.json` | Optional session dumps; skip unless you want chat archives in git |
| `diagnostic_experiments/perlayer_pca_control/verification/global_fit_directions_sha256_manifest_{pre,post}_run.json` | Large regenerable byte-identity manifests (~2M each); same class as prior exclusion |

Direction `.npz` trees under `experiment_artifacts/` are not in the untracked list (good). The meandiff **manifest** JSON is small and worth tracking.

## Suggested commits (9)

Order is dependency-friendly: harness → pins → extraction → eval wiring → CHAIR default → plan → overnight/analysis → RunAI → logs/answers.

### 1. Harness: no minted shorthand + last-updated stamps

Tighten planner/implementer naming rules; stamp harness files; allow `Edit(IMPLEMENTATION.md)` again; document stamp convention.

- `.claude/agents/{examiner,planner,tutor}.md`
- `.claude/commands/{examine-design,examine-extraction,examine-results,plan,tutor}.md`
- `.claude/hooks/{require_design_spec.py,verify_harness.sh}`
- `.claude/settings.json` (also has session-local allowlist noise — consider stripping ephemeral `/tmp/claude-…` allow entries before commit)
- `.cursor/rules/{providing_answers,research_workflow}.mdc`
- `CLAUDE.md` (two-logs section + naming section)
- `WORKFLOW.md`, `WORKFLOW_MAP.md` (stamps section), `TOOLING.md`
- `templates/{abstract,bypass_log_starter,design,extraction,reading}_template.md`
- `answers/README.md`
- `helper_scripts/export_agentic_workflow_harness.sh`
- `helper_scripts/stamp_harness_dates.py` (new)

Draft message:

```
Tighten no-shorthand naming across planner/implementer harness and stamp last-updated dates.
```

### 2. Track CHAIR pin; stop ignoring project logs

- `.gitignore` — un-ignore `IMPLEMENTATION.md` / `RESEARCH_LOG.md`; allow `data/chair/pinned_chair_500.json`
- `data/chair/pinned_chair_500.json` (new)

Draft message:

```
Track pinned_chair_500 and stop gitignoring IMPLEMENTATION.md and RESEARCH_LOG.md.
```

### 3. Mean-difference demos_850 extraction

CPU meandiff over act-cache partitions; extract + verify; extraction manifest.

- `evaluation/interventions/vti/directions_meandiff.py` (new)
- `evaluation/run_scripts/extract_demos850_meandiff_directions.py` (new)
- `helper_scripts/verify_demos850_meandiff_extraction.py` (new)
- `experiment_artifacts/vti/demos850_meandiff_extraction_manifest_2026-07-30.json` (new)

Draft message:

```
Add demos_850 raw mean-difference textual direction extraction and verify.
```

### 4. Eval: load precomputed directions and layer windows

Wire `--directions_dir` / `--layer_set` through CLI → runner → `VTITextualIntervention`; meandiff result-dir naming.

- `evaluation/interventions/vti/intervention.py`
- `evaluation/interventions/__init__.py`
- `evaluation/run_eval.py` (layer-set parser + directions_dir; CHAIR default change is commit 5)
- `evaluation/runners/eval_runner.py`

**Note:** `run_eval.py` also changes `--chair_max_new_tokens` default 64→256. Prefer splitting that hunk into commit 5 (or commit the whole file here and leave scripts/readme for 5). Cleanest: stage `run_eval.py` with both changes in **5** if you do not want to split hunks, and keep this commit to intervention + runner only — but then runner/CLI must land together. Practical split:

- This commit: intervention + `__init__` + runner + the `--directions_dir` / `--layer_set` parts of `run_eval.py`
- Next commit: CHAIR default in `run_eval.py` + callers

If hunk-splitting is annoying, merge 4+5.

Draft message:

```
Wire --directions_dir and --layer_set into textual VTI eval.
```

### 5. Raise standing CHAIR caption cap default to 256

- `evaluation/run_eval.py` (`--chair_max_new_tokens` default)
- `evaluation/chair_amber_diagnostics/run_scripts/run_exp1_repro_grid.sh`
- `evaluation/chair_amber_diagnostics/step0_chair_token_cap.py` (`--max_pixels`, model-suffixed outputs, truncation readouts)
- `readme.md` (aligned defaults)

Draft message:

```
Raise standing CHAIR caption token cap default from 64 to 256.
```

### 6. Steering visual-reasoning validation plan (folded sanity checks)

- modify `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md`
- delete `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_sanity_checks_plan_2026-07-30.md`

Draft message:

```
Fold steering visual-reasoning sanity checks into the main validation plan.
```

### 7. Overnight grid drivers, babysitter, and offline analysis

- `evaluation/run_scripts/run_steering_visual_reasoning_validation.sh` (new)
- `evaluation/run_scripts/launch_steering_visual_reasoning_overnight.sh` (new)
- `evaluation/run_scripts/babysit_steering_visual_reasoning_grids.py` (new)
- `evaluation/steering_visual_reasoning_validation/{__init__,build_result_tables,make_plots}.py` (new)
- `helper_scripts/build_amber_plot_qualitative_html.py` (new; AMBER response galleries for overnight plots)

Draft message:

```
Add steering visual-reasoning overnight drivers, babysitter, and analysis plots.
```

### 8. RunAI pack/sync/smoke/triple launchers for the grid

- `helper_scripts/runai/SUBMIT_STEERING_LLAVA.md` (new)
- `helper_scripts/runai/pack_steering_llava_for_runai.sh` (new)
- `helper_scripts/runai/sync_steering_llava.sh` (new)
- `helper_scripts/runai/verify_steering_nfs_layout.sh` (new)
- `helper_scripts/runai/run_steering_visual_reasoning_{llava,qwen}.sh` (new)
- `helper_scripts/runai/run_steering_{llava,qwen}_smoke.sh` (new)
- `helper_scripts/runai/run_steering_triple_one_h100.sh` (new)
- `helper_scripts/runai/runai_job_logging.sh` (new)
- `helper_scripts/runai/update_repo_from_tarball.sh` (new)
- `helper_scripts/runai/run_bash_lf.py` (argv guard)

Draft message:

```
Add RunAI sync, smoke, and one-H100 triple launchers for steering grids.
```

### 9. Record code/run facts + chat answers

Matches recent style (`e92e9a8`).

- `IMPLEMENTATION.md`
- `RESEARCH_LOG.md`
- new `answers/` markdown under `benchmarks/`, `chair_amber_diagnostics/`, `codebase_records/`, `cursor_remote_ssh/`, `runai/`, `runs/` (july_30 / july_31 trees)
- optionally also the small `answers/README.md` stamp if not in commit 1

Draft message:

```
Record meandiff/steering-grid implementation notes and july 30–31 answer notes.
```

## Optional thinner/fatter variants

- **Merge 4+5** if you do not want to hunk-split `run_eval.py`.
- **Split 7**: analysis package vs overnight orchestration vs qualitative HTML — only if reviewability matters more than fewer commits.
- **Split 9**: `IMPLEMENTATION.md`/`RESEARCH_LOG.md` alone, then `answers/` alone — useful if answers are noisy.
- **Harness settings hygiene:** before commit 1, drop ephemeral `Bash(python3 "/tmp/claude-…")` allowlist entries from `.claude/settings.json` if those paths are machine-session-specific.

## Rough `git add` map (commits 1–9)

```text
1  .claude/ .cursor/ CLAUDE.md WORKFLOW.md WORKFLOW_MAP.md TOOLING.md templates/ \
   answers/README.md helper_scripts/export_agentic_workflow_harness.sh \
   helper_scripts/stamp_harness_dates.py

2  .gitignore data/chair/pinned_chair_500.json

3  evaluation/interventions/vti/directions_meandiff.py \
   evaluation/run_scripts/extract_demos850_meandiff_directions.py \
   helper_scripts/verify_demos850_meandiff_extraction.py \
   experiment_artifacts/vti/demos850_meandiff_extraction_manifest_2026-07-30.json

4  evaluation/interventions/__init__.py evaluation/interventions/vti/intervention.py \
   evaluation/runners/eval_runner.py  (+ directions/layer pieces of run_eval.py)

5  evaluation/run_eval.py evaluation/chair_amber_diagnostics/ readme.md

6  implementation_plans/7-30-26/

7  evaluation/run_scripts/{run,launch,babysit}_steering* \
   evaluation/steering_visual_reasoning_validation/ \
   helper_scripts/build_amber_plot_qualitative_html.py

8  helper_scripts/runai/

9  IMPLEMENTATION.md RESEARCH_LOG.md answers/{benchmarks,chair_amber_diagnostics,\
   codebase_records,cursor_remote_ssh,runai,runs}/
```
