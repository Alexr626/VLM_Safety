# Repository map

Orientation for this codebase. Metrics and readings live in `RESEARCH_LOG.md` / analysis notes; code facts in `IMPLEMENTATION.md`. Artifact locations: `ARTIFACT_INDEX.md`. Archive/restore: `ARTIFACT_ARCHIVE.md`.

**Two remotes, two trees**

| Remote | Branch (typical) | Contents |
|--------|------------------|----------|
| Personal GitHub (`Alexr626/VLM_Safety`) | `VLM_hallucination_mitigation` | Full tree including agentic research-workflow harness |
| Nokia GitLab | `manager_handoff` | Experimental / eval code only — no agentic workflow paths |

Paths labeled **(personal GitHub only)** below are absent from the GitLab handoff tree.

---

## Start here (post-internship focus)

Matched LLaVA POPE comparison of differently constructed steering vectors:

- Analysis output: `evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/`
- Builder: `evaluation/pope_0619_vs_0730_matched/build_comparison.py`
- Arms: 2026-06-19 author-demo nd70 PC1+mean vs 2026-07-30 demos850 nd500 meandiff (`layers_all`)

Related continuation (AMBER-1500, meandiff + PC1+mean): see `evaluation/steering_vector_validation_continuation/` and `evaluation/results/2026-08-05/`.

---

## Core library

| Path | Role | Notes |
|------|------|-------|
| `src/` | Model wrappers, benchmark loaders, activation cache, paths | Entry: `create_wrapper`, `BENCHMARK_REGISTRY`, `project_root()` |
| `evaluation/run_eval.py` | CLI for benchmark × intervention runs | `--directions_dir`, `--layer_set`, `--chair_max_new_tokens` |
| `evaluation/runners/` | Shared eval loop, result-dir naming | |
| `evaluation/interventions/` | `no_intervention`, VTI textual/visual | Meandiff + PCA directions under `vti/` |
| `evaluation/classifiers/` | Metrics / scorers | |
| `tests/` | Unit tests | |
| `environment.yml` | Conda env source of truth | |

## Evaluation packages (primary)

| Path | Role | Primary / historical |
|------|------|----------------------|
| `evaluation/pope_0619_vs_0730_matched/` | Offline POPE cross-date comparison | **Primary** |
| `evaluation/steering_vector_validation_continuation/` | AMBER-1500 per-config / per-item tables | **Primary** |
| `evaluation/steering_visual_reasoning_validation/` | 2026-07-30 multi-benchmark grid analysis + plots | Primary (prior overnight grid) |
| `evaluation/run_scripts/` | Drivers: extract directions, overnight grids, babysitters | Primary |
| `evaluation/chair_amber_diagnostics/` | Earlier CHAIR+AMBER VTI grids | Historical but still used |
| `evaluation/vti_rotation_strength/` | Rotation-strength sweeps | Historical |
| `evaluation/results/` | Run outputs (mostly gitignored) | Restore from Drive — see `ARTIFACT_ARCHIVE.md` |

## Data and artifacts

| Path | Role | Notes |
|------|------|-------|
| `data/` | Benchmarks, COCO, pins, VTI demos | Raw benchmarks re-downloadable; pins/demos often must-restore |
| `data_scripts/` | Download + prep scripts | |
| `experiment_artifacts/` | Directions, act caches, derangements | Mostly gitignored; see `ARTIFACT_INDEX.md` |
| `diagnostic_experiments/` | Modality shift, mediation, perception diag, per-layer PCA geometry | Mixed: some active, some historical |

## Ops / RunAI

| Path | Role |
|------|------|
| `helper_scripts/` | Verify scripts, qualitative HTML, remaps |
| `helper_scripts/runai/` | NFS sync, smoke, one-H100 triples, AMBER Qwen pack |

## Logs (code vs measurement)

| Path | Role |
|------|------|
| `IMPLEMENTATION.md` | How the codebase works; artifact production recipes |
| `RESEARCH_LOG.md` | What runs produced (settings + raw outcomes) |
| `ABSTRACT.md` | High-level project abstract |
| `STEERING_MATH_REFERENCE.md` | Math notes for steering geometry |
| `readme.md` | Setup + handoff entry |

## Agentic research workflow (personal GitHub only)

Absent from Nokia `manager_handoff`. Do not rely on these for manager onboarding.

| Path | Role |
|------|------|
| `answers/` | Agent scratchpad / Q&A markdown |
| `.claude/`, `.cursor/` | Agent harness, hooks, rules |
| `templates/` | Design / extraction / reading templates |
| `WORKFLOW.md`, `WORKFLOW_MAP.md`, `TOOLING.md`, `CLAUDE.md` | Workflow docs |
| `learning/` | Personal review queue + scans |
| `designs/`, `extractions/`, `analysis/`, `implementation_plans/` | Specs, plans, readings |

## Other

| Path | Role |
|------|------|
| `VTI/`, `llava-interp/` | Nested / reference trees — not required for the main eval path |
| `logs/`, `wandb/`, `exports/` | Local / generated; gitignored |
| `.env` | Secrets — **never** commit or upload |
