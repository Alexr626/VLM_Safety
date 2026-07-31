# Commit split — shuffled cosine comparison vs research workflow

Date: 2026-07-28  
Branch: `new_research_workflow`  
Question: how to group untracked work into commits that separate the deployed-vs-shuffled n=200 cosine comparison from the new agent research harness, without committing shuffled activation caches.

## Inventory (what `git status` currently shows)

Everything below is untracked. `.claude/`, `.cursor/`, and `answers/` are present on disk but ignored by `.gitignore`.

| Bucket | Paths | Approx size |
|---|---|---|
| Shuffled-control code | `evaluation/interventions/vti/shuffled_control.py`, `evaluation/run_scripts/extract_shuffled_control_directions.py` | small |
| Shuffled directions (keep) | `experiment_artifacts/vti/{llava,qwen…}/shuffled_control/all_nd200/{directions,components,metadata}.*`, sanity reports, `shuffled_control_image_derangement_nd200_s1234.json` | ~4 MB |
| Shuffled activations (do not commit) | `…/shuffled_control/_act_cache/` (400 files × 2 models) | ~215 MB |
| Cosine comparison results | `diagnostic_experiments/perception_diag/control/geometric_comparison/` (script + csv + png + summary) | ~384 KB |
| Research workflow | root docs, `templates/`, `analysis/bypass_log.md`, `extractions/`, and (currently ignored) `.claude/` | small |
| Nested vendor clones (leave out) | `VTI/`, `llava-interp/` (each has its own `.git` / remotes) | large / not yours |

## Recommended commits (4)

Do these in order. Fold the `.gitignore` edits into the commit that first needs them.

### 1. Ignore heavy caches / vendor trees (prerequisite)

Update `.gitignore` so a later `git add` cannot accidentally suck in activations or nested repos:

```gitignore
# Shuffled-control (and similar) activation dumps — keep directions/metadata only
experiment_artifacts/**/_act_cache/

# Nested upstream clones — use remotes/submodules, do not vendor into this repo
/VTI/
/llava-interp/

# Claude Code harness: track shared config; keep local overrides private
# (replace the bare `.claude/` ignore with the two lines below)
.claude/settings.local.json
# remove or comment out: .claude/
```

Suggested message:

> Ignore shuffled activation caches and nested vendor clones; allow tracking shared `.claude/` harness files.

Without this, commit 2/3 are easy to contaminate with `_act_cache`.

### 2. Shuffled-image control extraction tooling

**Include only code:**

- `evaluation/interventions/vti/shuffled_control.py`
- `evaluation/run_scripts/extract_shuffled_control_directions.py`

Suggested message:

> Add shuffled-image control direction extraction for VTI textual demos.

This is the producer; no result numbers yet. Keeps the method reviewable independently of the cosine plots.

### 3. Deployed vs shuffled geometric comparison (n=200) — results commit

**Include small reproducible artifacts + comparison outputs:**

- `experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json`
- `experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/all_nd200/`  
  (`directions.npz`, `components.npz`, `metadata.json`)
- `experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/shuffled_control_sanity_report_llava-1.5-7b-hf.md`
- `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/all_nd200/`  
  (same three files)
- `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/shuffled_control_sanity_report_qwen2.5-vl-7b-instruct.md`
- `diagnostic_experiments/perception_diag/control/geometric_comparison/`  
  (comparison script, CSVs, PNGs, `geometric_comparison_summary.md`)

**Explicitly exclude:**

- `experiment_artifacts/vti/*/shuffled_control/_act_cache/`

Safe add pattern after commit 1’s gitignore:

```bash
git add \
  experiment_artifacts/vti/shuffled_control_image_derangement_nd200_s1234.json \
  experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/all_nd200 \
  experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/shuffled_control_sanity_report_llava-1.5-7b-hf.md \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/all_nd200 \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/shuffled_control_sanity_report_qwen2.5-vl-7b-instruct.md \
  diagnostic_experiments/perception_diag/control/geometric_comparison
git status   # confirm no _act_cache paths staged
```

Suggested message:

> Record deployed-vs-shuffled-control geometric comparison at nd200 (directions, plots, summary).

Precedent: existing tracked visual VTI dirs keep `directions.npz` + `metadata.json`; including `components.npz` here is consistent with the comparison needing PC1-fraction readouts (~3 MB total, fine).

### 4. Agent research workflow harness

**Include:**

- `CLAUDE.md`
- `WORKFLOW.md`
- `WORKFLOW_MAP.md`
- `TOOLING.md`
- `ABSTRACT.md`
- `STEERING_MATH_REFERENCE.md` (project math reference the workflow agents are told to respect; belongs with the harness, not with the cosine cell)
- `templates/` (`design_template.md`, `bypass_log_starter.md`, `reading_template.md`, `abstract_template.md`)
- `analysis/bypass_log.md`
- `extractions/steering_vector_diff_sample_sizes_07_28_26_extraction.md` (first extraction-gate artifact under the new process)
- empty `designs/` only if you want the directory scaffold tracked (use `.gitkeep` if so)
- `.claude/agents/`, `.claude/commands/`, `.claude/hooks/`, `.claude/settings.json`  
  (requires commit 1’s gitignore change; do **not** add `settings.local.json`)

Suggested message:

> Add agent research workflow docs, templates, and Claude Code examiner/planner/tutor harness.

## Leave uncommitted

| Path | Why |
|---|---|
| `…/shuffled_control/_act_cache/` (~215 MB, 800 npz) | Regenerable from the derangement + extract script; too large for git |
| `VTI/` | Upstream clone (`shengliu66/VTI`); nested `.git` |
| `llava-interp/` | Upstream clone (`clemneo/llava-interp`); nested `.git` |
| `answers/`, `.cursor/` | Already ignored; local agent/IDE scratch, not the shared workflow surface |
| `IMPLEMENTATION.md`, `RESEARCH_LOG.md` | Already ignored by design |

## Optional thinner split

If you want commit 3 even tighter:

- **3a** directions + derangement + sanity reports only  
- **3b** `geometric_comparison/` plots/CSVs/summary (+ comparison script)

Not necessary; the geometric cell is one experiment and ~4 MB total without caches.

## Order rationale

1. Gitignore first → hard to accidentally commit activations.  
2. Extraction code before results → bisect/revert of method vs evidence stays clean.  
3. Cosine results alone → history of the science cell is one commit.  
4. Workflow last → process harness does not interleave with experimental evidence, and `.claude/` tracking is an intentional policy change.
