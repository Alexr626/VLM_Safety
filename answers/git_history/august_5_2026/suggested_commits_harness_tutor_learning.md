# Suggested commits — newest harness-only changes

Date: 2026-08-05. Scope: research agentic workflow harness artifacts only (no `learning/` content, no analysis, no results).

## In scope (modified)

| File | What changed |
|---|---|
| `.claude/agents/tutor.md` | Edit tool; drop `STEERING_MATH_REFERENCE.md`; allow `learning/{review_queue,lit_review_queue,scans}`; textbook-sourced practice problems; scan-reading protocol; compressed-first pacing; paired-binary inference topic; mid-task queue logging |
| `.claude/commands/tutor.md` | Brief/read rules aligned with the agent |
| `WORKFLOW_MAP.md` | Map `learning/*` rows; drop `STEERING_MATH_REFERENCE.md` from stamp exclusions |
| `helper_scripts/export_agentic_workflow_harness.sh` | Drop `STEERING_MATH_REFERENCE.md` from export; copy `answers/README.md`; `mkdir learning` |

## Out of scope (left alone)

- `learning/` (untracked content — not harness definition)
- `analysis/`, `answers/concepts/`, `answers/runai/`, designs, diagnostic manifests, plots
- Session transcripts
- Hook scripts (none dirty)

## Suggested commits (1)

One coherent tutor/learning harness retarget; splitting agent vs map/export would orphan half the change.

### 1. Tutor harness + map/export alignment

- `.claude/agents/tutor.md`
- `.claude/commands/tutor.md`
- `WORKFLOW_MAP.md`
- `helper_scripts/export_agentic_workflow_harness.sh`

Draft message:

```
Retarget tutor harness to learning queues and textbook-sourced practice.
```

Optional 2-commit split if preferred:

1. `.claude/agents/tutor.md` + `.claude/commands/tutor.md` — same message focused on the role
2. `WORKFLOW_MAP.md` + `helper_scripts/export_agentic_workflow_harness.sh` — "Map and export learning/ paths; drop STEERING_MATH_REFERENCE from harness bundle."
