# Research workflow

Four roles, three gates, and a small set of files only you write. This document is how to run it.

The design principle behind all of it: **anything whose output is a belief you will have to
defend is yours; anything whose output is an artifact belongs to an agent.** Every rule below
is that line, applied somewhere specific.

## Install

Copy into the repo root on lambdab2:

```
CLAUDE.md                                # shared context, replaces the analyst CLAUDE.md
WORKFLOW.md                              # this file
TOOLING.md                               # client split and migration
WORKFLOW_MAP.md                          # file map and enforcement inventory
.claude/settings.json                    # permissions + hook wiring
.claude/agents/examiner.md
.claude/agents/tutor.md
.claude/agents/planner.md
.claude/hooks/require_design_spec.py     # chmod +x
.claude/hooks/verify_harness.sh          # chmod +x
.claude/commands/examine-results.md
.claude/commands/examine-design.md
.claude/commands/examine-extraction.md
.claude/commands/plan.md
answers/README.md
templates/                               # copy from these, do not edit in place
```

Then create the directories you write into, and seed two files:

```bash
mkdir -p analysis designs extractions answers/concepts
cp templates/bypass_log_starter.md analysis/bypass_log.md   # then delete the example entry
cp templates/abstract_template.md ABSTRACT.md
chmod +x .claude/hooks/require_design_spec.py .claude/hooks/verify_harness.sh
```

Verify before trusting any of it:

```bash
bash .claude/hooks/verify_harness.sh
```

It must print `passed=14 failed=0`. The script exercises the gate directly with synthetic
tool-call events, so it does not depend on talking an agent into attempting a write. If it
fails, every gate below is decoration.

Two things the script cannot check, because Claude Code rather than the hook enforces them: ask
an agent to write to `analysis/` and to `designs/` and confirm both are refused. However, editing existing experiment or extracting designs is okay in some cases. See the rules below for more details. 

## The four roles

**Examiner** — interrogates what you wrote. Reads your design, your extraction spec, or your
reading of a run, and asks questions. Outputs only questions, facts read from files, and factual
discrepancies between the two. Cannot tell you what a result means. Gated on your file existing.

**Tutor** — teaches concepts. Statistics, linear algebra, steering geometry, the literature.
Explains, derives, works synthetic examples. Cannot read your results or apply a concept to
them; when you ask it to, it gives you a toy case with the same structure and you carry it back
yourself. Writes full derivations to `answers/concepts/<date>/` and returns a plain-notation
summary, because terminals do not render LaTeX.

**Planner** — turns a spec you wrote into a plan Cursor can implement. Verifies identifiers
against the repo, finds what already exists on disk, specifies signatures and paths and artifact
names. Adds nothing that is not in your spec. Gated on the spec existing.

**Cursor** — implements and runs. Works from plans that carry a spec reference, so every
artifact traces back to something you wrote.

## Two kinds of spec

The workflow has two entry points because two different things get produced.

**Experiments** produce a number you read as evidence. Comparisons, measurements, benchmark
scores, reliability curves, cosines between directions. These need `designs/<exp_id>_design.md`.

**Extractions** produce primitives. Activations, per-layer stacks, attention weights, per-head
values, residual streams, extracted directions, caches. These need
`extractions/<ext_id>_extraction.md`.

The dividing test, when something sits near the line: **would a different value change what you
believe?** Producing a tensor cannot come out wrong in a way that changes a belief. Measuring
something can.

Two clarifications that matter in practice. The artifact type is not the constraint — a
direction is as extractable as a raw activation, and attention weights are as extractable as
residual streams. And **compute type is not the constraint either.** Most of the comparisons
this project needs are CPU-only numpy over the existing cache; being cheap and offline does not
make something an extraction. Descriptive checks on artifacts themselves — file counts, shapes,
the norm of a single extracted object — stay on the extraction side. The moment two arms are
contrasted, it is an experiment.

## The files you write

Check the `settings.json` for allowed capabilitites. While you're not able to write to any of the files below when they don't already exist, you are able to edit some of them once they exist (ex. when Alex has created an initial version, given it to you for review, and, only after advising Alex through questioning and feedback, have you both agreed on edits you can make to improve them).

`designs/<exp_id>_design.md` — before an experiment. From `templates/design_template.md`. The
prediction table is the part that matters: one row per condition, one column per competing
explanation, what each predicts. If two columns come out identical across every row, the design
cannot separate those explanations and should not be run. That check costs ten minutes.

`extractions/<ext_id>_extraction.md` — before an extraction. From
`templates/extraction_template.md`. Two fields carry the weight. **Independence structure** —
which id sets are disjoint, nested, or identical, and why — because that is the one choice no
later analysis can undo. And **comparisons this must support later**, which does the job the
prediction table does: it needs no hypothesis, only a statement of what you must be *able* to
compute, and it is checkable against the artifacts afterwards.

`analysis/<run_id>_reading.md` — after a run, before any agent sees the numbers. From
`templates/reading_template.md`. Write it from the result files directly. Asking an agent what
the files show before you have written this defeats the gate, because a factual summary arrives
with an implied reading attached.

`ABSTRACT.md` — rewritten after every run. Every claim names the file that supports it. Claims
you want but cannot support go in the parked section. The cheapest early warning you have: when
a sentence stops matching a number, one of them has to change.

`analysis/bypass_log.md` — every gate lift, dated, in your own words.

## The loop

```
              need primitives?                    need an answer?
                     |                                   |
                     v                                   v
   extractions/<ext_id>_extraction.md     designs/<exp_id>_design.md      [you]
                     |                                   |
                     v                                   v
   /examine-extraction <ext_id>           /examine-design <exp_id>        [examiner]
                     |                                   |
                     v  revise until it holds            v  revise until it holds
   /plan <ext_id>                         /plan <exp_id>                  [planner]
                     |                                   |
                     v  you review the plan              v  you review the plan
   Cursor runs the extraction             Cursor implements and runs
                     |                                   |
                     v                                   v
   artifacts on disk ---------- feed ---->  analysis/<run_id>_reading.md  [you]
                                                         |
                                                         v
                                            /examine-results <run_id>     [examiner]
                                                         |
                                                         v  revise your reading
                                            ABSTRACT.md updated           [you]
                                                         |
                                                         v
                                            next question                 [you]
```

The extraction branch has no reading stage, because nothing was measured. Its output is
artifacts, and the check on it is the verification section of its own spec — did the files land,
in the shapes and counts claimed, keyed as intended.

Two places the loop exits sideways. Any point where you do not understand a concept well enough
to defend it goes to the Tutor, not the Planner — "I do not know how to design a test that
separates a criterion shift from sycophancy" is a teaching request, and the Planner will happily
answer it in a way that costs you the competency. Any point where you need to know what is in a
paper, a file, or a function is a plain agent request with no ceremony attached.

## The gates

**Design gate.** The Planner refuses without `designs/<exp_id>_design.md`, and the pre-write
hook blocks any file landing in `implementation_plans/` that does not declare an existing,
filled spec on its second line. Deterministic — the file exists or it does not.

**Extraction gate.** The same hook, accepting
`extraction_spec: extractions/<ext_id>_extraction.md` instead. It enforces that exactly one
declaration is present and that it points inside its own directory.

What the hook rejects is two *declarations*, not a plan that does two things. A design spec may
carry the primitives it strictly requires — list them under *Primitives this design requires*,
declare `design_spec:` alone, and one plan covers both. See "What this does not fix" below for
why the boundary runs in only one direction.

**Reading gate.** The Examiner refuses without `analysis/<run_id>_reading.md`. Enforced by the
command file and the agent definition rather than by a hook, because the trigger is
conversational rather than a file write. It is therefore softer, and the honest statement is
that you can talk your way past it if you want to.

**Bypass.** You lift a gate by writing an entry in `analysis/bypass_log.md`, dated today, naming
the run or spec. The Examiner reads that file and honours the entry. It will never write one for
you and will never suggest it.

Do not try to get the bypass count to zero. The number is the instrument. You have been running
on self-report about how often you accept an agent's reading, and self-report is exactly what
failed. If the log shows you bypassing most of the time, that is the most important fact about
your workflow, and right now you have no way to see it.

## What this does not fix

**The extraction/experiment boundary is self-policed.** Nothing in the hook can tell whether a
spec labelled extraction is really a comparison wearing a different hat. Writing an experiment
into `extractions/` skips the prediction table entirely, and it is the obvious route around the
harder gate. It will feel reasonable at the time — the work is offline, the artifacts are cheap,
the hypothesis is not ready. The test is whether the plan produces a number you would read. If
it does, it is an experiment no matter which directory the spec sits in.

The boundary runs one way only. Producing data is not the consequential act; reading a number
off it is. So an experiment that happens to need new primitives first does **not** need a second
spec — list them under *Primitives this design requires* in the design spec and the plan covers
both. What the rule catches is the reverse: a comparison filed as an extraction, where the
prediction table never gets written. Writing two specs for one question is friction with nothing
on the other side of it.

**Leading questions leak answers.** An examiner asking exactly the right pointed question has
told you the answer with a question mark on it. The agent definition pitches questions at the
level of category rather than instance, which reduces the leak without closing it. Expect some
of the work to be done for you and try to notice when it is.

**The tutor boundary is not enforced by permissions.** It is instruction only. Path-scoped read
denial is not expressible per-subagent in the current configuration, so the Tutor's promise not
to read your results holds exactly as long as it follows its own file. The same applies to its
closing comprehension check, which has been observed to skip.

**Subagent files load only when the subagent is invoked.** A plain session gets `CLAUDE.md` and
nothing else. Asking a general session a tutor question does not get you the tutor.

**You can always open an unconstrained session.** No configuration prevents you from starting a
plain Claude Code session with no `CLAUDE.md` and asking for the answer. The bypass log is the
only defence, and it is self-report.

**The permission surface is not finished.** The general shape is right and the gates below hold,
but the read-only edges are still being tuned. Occasionally an agent is denied a call that would
have answered a factual question — listing a directory, parsing a stored `.npz` to report what
is in it — without doing any of your thinking for it. The failure mode to watch for is not the
denial; it is an agent that answers from reasoning instead of from the repo and does not say
which read it was missing. Denials should be reported by name, with the fact left marked
unverified. When one looks like an edge case rather than the rule working, it is yours to widen
or leave alone — an agent proposing a workaround for its own permissions has the wrong end of
the loop.

**None of this makes the work faster.** The gates cost you time on every loop, and the first few
will feel like ceremony. The evidence that they earn it is already on record: the gold-yes-only
POPE design would have died on the prediction table, before the run, for ten minutes of writing.

## When to revisit

After ten logged runs, read the bypass log and count. Then ask whether the readings you wrote
got sharper — whether the examiner's questions started arriving as things you had already
covered. If they did, tighten the gates. If the bypass rate is high and the readings are not
improving, the problem is not the configuration and no further configuration will fix it.
