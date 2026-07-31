# Proposed harness commit after path-layout revisions

Date: 2026-07-30. Baseline: `5eff7fe` (previous Claude research-workflow harness commit).

## What changed (symptoms / facts from the diff)

Three themes, all since `5eff7fe`:

1. **Spec/reading path convention.** Commands, agents, docs, templates, and the hook messaging no longer assume `designs/<id>_design.md` / `extractions/<id>_extraction.md` / `analysis/<id>_reading.md` at directory root. Paths may be dated subdirectories and may omit the old suffixes. Slash commands take a repo-relative path pasted from the editor; bare ids fall back to recursive glob and ask on ambiguity. Plan second-line declarations must copy the resolved path verbatim.

2. **Prediction-table / sample-size examiner guidance.** Examiner + `/examine-design` + design template: a prediction-table row is a condition, not a factorial arm; do not demand one entry per arm/beta/layer set/sample size. Sample-size questions target evaluation-n resolution (instrument property), not forecasts for swept direction/demo sizes.

3. **Export helper.** New `helper_scripts/export_agentic_workflow_harness.sh` packages the harness tree for offline / other-web-UI iteration. `.gitignore` gained an `exports/` rule for that output (same file also has unrelated `demos_850` allowlist lines — see below).

Unchanged vs `5eff7fe` among harness files: `CLAUDE.md`, `TOOLING.md`, tutor agent/command, `verify_harness.sh`, `settings.json`, `abstract_template.md`, `bypass_log_starter.md`.

## Proposed commit file list (harness only)

Include:

- `.claude/agents/examiner.md`
- `.claude/agents/planner.md`
- `.claude/commands/examine-design.md`
- `.claude/commands/examine-extraction.md`
- `.claude/commands/examine-results.md`
- `.claude/commands/plan.md`
- `.claude/hooks/require_design_spec.py`
- `WORKFLOW.md`
- `WORKFLOW_MAP.md`
- `templates/design_template.md`
- `templates/extraction_template.md`
- `templates/reading_template.md`
- `helper_scripts/export_agentic_workflow_harness.sh` (new; companion to this harness work)

`.gitignore`: **partial.** The working-tree file mixes (a) `exports/` for the export script and (b) `!data/vti/demos_850*.json(l)` allowlist lines that belong with the demos_850 experiment commit, not this one. Prefer staging only the `exports/` hunk, or leave `.gitignore` out and handle `exports/` in a later cleanup.

Exclude (same as last harness commit): experiment code/data (`src/paths.py`, `directions_v2.py`, demos_850 scripts/artifacts, `designs/`, extractions churn, etc.).

## Draft commit message

```
Relax harness path conventions to dated subdirs and drop required _design/_reading suffixes.

Commands resolve a pasted repo-relative path (bare-id glob as fallback); plans must declare
that exact path. Examiner/design guidance treats prediction-table rows as conditions, not
factorial arms. Add helper_scripts/export_agentic_workflow_harness.sh for offline bundle export.

Files included:
- .claude/agents/examiner.md
- .claude/agents/planner.md
- .claude/commands/examine-design.md
- .claude/commands/examine-extraction.md
- .claude/commands/examine-results.md
- .claude/commands/plan.md
- .claude/hooks/require_design_spec.py
- WORKFLOW.md
- WORKFLOW_MAP.md
- templates/design_template.md
- templates/extraction_template.md
- templates/reading_template.md
- helper_scripts/export_agentic_workflow_harness.sh
```

(Add `.gitignore` to the list only if the `exports/` hunk is staged alone.)

## One doc inconsistency to be aware of

`WORKFLOW.md` still says slash commands “take the bare id, search … recursively”; the command files now primarily take a full path and only glob on bare ids. The install/flow diagrams were updated to `<path>`. Worth a one-line WORKFLOW edit in this commit or the next — not blocking.

## Status

Proposal only — no commit created. Say if you want this commit made (and whether to include the `exports/` gitignore hunk and/or the export script).
