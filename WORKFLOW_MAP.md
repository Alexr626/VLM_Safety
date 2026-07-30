# Workflow map — what lives where and what enforces what

Reference for the agentic research workflow. Written 2026-07-28, revised 2026-07-28 to add the
extraction branch.

Four documents describe this workflow and they divide as follows. Keep them from drifting by
sending detail to whichever one owns it rather than restating.

| Document | Owns |
|---|---|
| `WORKFLOW.md` | The loop. Install steps, the order of operations, the two spec kinds, the gates, the bypass protocol, what the harness does not fix. |
| `TOOLING.md` | Client split. Which role runs in Cursor, which in Claude Code, what ports and what does not, migration after the internship. |
| `WORKFLOW_MAP.md` | This file. Where every file lives, who writes it, who reads it, and what enforces each rule. |
| `HANDOFF_<date>.md` | State at a point in time. What is verified, what is outstanding, where the research stands. Goes stale by design; date it and write a new one. |

`CLAUDE.md` is not documentation of the workflow — it is shared project context loaded by every
agent, carrying infrastructure facts and the current evidence state.

---

## Directory map

Repo root: `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/`

```
CLAUDE.md                     shared context, loaded by every agent session
WORKFLOW.md                   the loop
TOOLING.md                    client split and migration
WORKFLOW_MAP.md               this file
ABSTRACT.md                   living draft, rewritten every run          [Alex only]
HANDOFF_<date>.md             point-in-time state

.claude/
  settings.json               permissions and hook wiring
  agents/
    examiner.md               questions only, gated
    tutor.md                  concepts only, never project results
    planner.md                spec to plan, gated, both spec kinds
  commands/
    examine-results.md        /examine-results <run_id>
    examine-design.md         /examine-design <exp_id>
    examine-extraction.md     /examine-extraction <ext_id>
    plan.md                   /plan <id>, routes on whichever spec exists
  hooks/
    require_design_spec.py    PreToolUse gate on plan writes, both spec kinds
    verify_harness.sh         deterministic test of the gate, 14 cases

.cursor/rules/                Cursor-side rules; implementation agent only

designs/<exp_id>_design.md         experiment spec, before a run         [Alex only]
extractions/<ext_id>_extraction.md extraction spec, before primitives    [Alex only]
analysis/<run_id>_reading.md       reading, before any agent sees it     [Alex only]
analysis/bypass_log.md             record of every gate lift             [Alex only]
answers/                      shared scratchpad; see answers/README.md
  README.md                   what may and may not be persisted here
  concepts/<date>/            tutor derivations, LaTeX, read in preview
  <topic>/<date>/             mechanism write-ups and factual lookups
implementation_plans/         planner output, gated by the hook
templates/                    copy from these; do not edit in place
  design_template.md
  extraction_template.md
  reading_template.md
  abstract_template.md
  bypass_log_starter.md
```

The repo's own code — `src/`, `evaluation/`, `diagnostic_experiments/`, `data/`, `scripts/` — is
unchanged by any of this and is denied to every agent except through the implementation agent
working from a plan.

---

## The two spec kinds

The single most consequential distinction in the workflow, because it determines which gate a
piece of work passes through.

| | Experiment | Extraction |
|---|---|---|
| Spec | `designs/<exp_id>_design.md` | `extractions/<ext_id>_extraction.md` |
| Produces | a number read as evidence | primitives |
| Examples | cosine between two directions, split-half reliability, benchmark scores under steering, any contrast between arms | activation caches, per-layer stacks, attention weights, per-head values, residual streams, extracted directions |
| Load-bearing field | the prediction table | how the sets relate, checked against what it must support later |
| Examined by | `/examine-design` | `/examine-extraction`, capped at three questions |
| Post-run stage | `analysis/<run_id>_reading.md`, then `/examine-results` | verification checks written by the plan |

**The test:** would a different value change what you believe? Producing a tensor cannot come
out wrong in a way that changes a belief; measuring something can.

**Two things that are not the test.** Artifact type — a derived direction is as extractable as a
raw activation. And compute type — most comparisons this project needs are CPU-only numpy over
the existing cache, and being cheap does not make them extractions.

---

## File ownership

| Path | Written by | Read by | Purpose |
|---|---|---|---|
| `designs/*_design.md` | Alex | examiner, planner | Hypothesis, competing explanations, prediction table, cells, item sets, n, falsifier. Gates the planner. |
| `extractions/*_extraction.md` | Alex | examiner, planner | Three fields: what data, how the sets relate, what it must support later. Everything an implementer can read from the repo is deliberately absent. Gates the planner. |
| `analysis/*_reading.md` | Alex | examiner | Prediction copied forward unedited, the numbers, the reading, rejected alternatives, evidence tier. Gates the examiner. |
| `analysis/bypass_log.md` | Alex | examiner | Every gate lift, dated. The rate is the instrument. |
| `ABSTRACT.md` | Alex | — | Every claim names the file supporting it. Unsupported wants go in the parked section. |
| `implementation_plans/*.md` | planner | Alex, implementation agent | Spec expanded with real identifiers, paths, signatures, sanity checks. Must declare exactly one of `design_spec:` or `extraction_spec:` on line 2. |
| `answers/concepts/<date>/*.md` | tutor | Alex, agents | Derivations in LaTeX, read in a preview pane rather than a terminal. |
| `answers/<topic>/<date>/*.md` | implementation agent | Alex, agents | Mechanism write-ups and factual lookups. Facts, never readings. |
| `CLAUDE.md` | Alex | every agent | Project facts, evidence state, competencies not to delegate. |

---

## Enforcement inventory

The most useful thing in this document. Rules fall into three tiers, and knowing which tier a
rule is in tells you how much to trust it.

### Deterministic — enforced by config or code

| Rule | Mechanism | Status |
|---|---|---|
| No plan lands without an existing, filled spec | `require_design_spec.py`, PreToolUse on Write/Edit/MultiEdit | verified, 14/14 |
| Exactly one spec declaration per plan; combined extraction-and-experiment plans refused | same hook | verified |
| A declaration must point inside its own directory | same hook | verified |
| Agents cannot write `analysis/`, `designs/`, `extractions/`, `ABSTRACT.md`, `answers/interpretations/` | `settings.json` deny | **unverified** |
| Agents cannot write repo source, data, results | `settings.json` deny | inherited from prior setup |
| No execution, no destructive shell, no git mutation | `settings.json` deny | inherited |
| Examiner cannot write anything at all | agent frontmatter: `tools: Read, Grep, Glob` | by construction |

The hook resolves its own path via `$CLAUDE_PROJECT_DIR`. If that is ever changed back to a
relative path, every write in the repo breaks the moment an agent changes directory, and the
failure is indistinguishable from a deliberate block because `python3` exits 2 on a missing file.
Run `verify_harness.sh` after any change to the hook or its wiring.

`Bash(mkdir*)` was removed from the deny list on 2026-07-28. It was blocking agents from
creating dated subdirectories under `answers/`, which presented as an unexplained write failure.
The destructive concern was always `rm` and `mv`; both remain denied.

### Instruction-only — enforced by model compliance

| Rule | Where |
|---|---|
| Examiner never interprets, ranks, or proposes | `.claude/agents/examiner.md` |
| Examiner refuses without a reading file | same, plus `.claude/commands/examine-results.md` |
| Tutor never reads results directories | `.claude/agents/tutor.md` |
| Tutor closes with a comprehension check | same — observed to fail twice; cause not yet isolated |
| Tutor writes derivations to disk, summary to terminal | same |
| Planner adds nothing beyond the spec | `.claude/agents/planner.md` |
| Planner refuses combined extraction-and-experiment requests | same, backed by the hook on the plan write |
| `answers/` holds facts, never readings | `answers/README.md` |

These hold as long as the model follows its file. Subagent files load **only when the subagent is
invoked** — a plain session gets `CLAUDE.md` and nothing else, which is the leading explanation
for the tutor compliance failures.

### Gate weight

The two gates are deliberately unequal. A bad design costs a belief — numbers get read, the
reading is wrong, and it survives into a paper. A bad extraction costs compute — you notice and
rerun. Gate weight scales with what the mistake costs.

The one exception is why the extraction gate exists at all: an error in how the sets relate to
each other does not announce itself. It produces artifacts that look correct, get used, and
confound everything downstream. That failure has design-sized cost in extraction clothing, and
it is the only thing the extraction examiner is really looking for.

### Unenforceable

- **Writing an experiment into `extractions/`.** The hook checks that a spec exists and is
  filled; it cannot check that the work described is really extraction. This is the obvious route
  around the prediction table and the one most likely to be taken under deadline, because the
  work genuinely is offline and the hypothesis genuinely is not ready. Self-policed by the
  "would a different value change what you believe" test.
- Opening a session with no configuration and asking for the answer.
- Discussing results in a Claude Desktop project chat, which has no gates and no log.
- An examiner question pitched so precisely that it hands over the answer.
- Whether Alex understood a derivation or nodded at it.

The bypass log is the only instrument covering any of these, and it is self-report. That is the
known ceiling on the whole design.

---

## Maintenance

When something changes, update the owning document and nothing else. When the enforcement tier of
a rule changes — an instruction becomes a hook, or a hook is removed — update the inventory
above, because that table is the one people will trust.

After any change to `settings.json` or the hooks directory:

```bash
bash .claude/hooks/verify_harness.sh
```

The unverified denies in the deterministic table need an agent to attempt a write to `analysis/`,
`designs/`, and `extractions/` and be refused. Until that is done, treat them as instruction-only.
