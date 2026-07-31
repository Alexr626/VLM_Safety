---
description: Have the examiner interrogate an experiment design before it is planned. Usage /examine-design designs/<MM_DD_YY>/<name>.md
---

Design spec: $1

**Resolving the spec.** `$1` is a repo-relative path to the spec file, pasted from the editor:
`designs/<MM_DD_YY>/<name>.md`. Take it literally. Do not glob, do not search for near matches,
do not try `_design` suffix variants.

- Append `.md` if it is missing. Strip a leading `./`.
- If the path does not exist, or does not start with `designs/`, say so, name the path, and
  stop. Do not guess a neighbouring file and do not draft the spec yourself.
- Only if `$1` contains no `/` at all is it a bare id. Then, and only then, glob
  `designs/**/$1.md`; if that matches more than one file, list every match and ask which.

The experiment id is the filename stem of the resolved path.

Read the resolved file before anything else. If it is empty or is an unfilled template, say so,
name the file, and stop. Do not preview what you would ask and do not summarise the design back
— a preview is the answer in a thinner form.

Then check `analysis/bypass_log.md` for an entry dated today naming that id. If one exists, the
gate is lifted for this design. Never write that entry and never suggest writing one.

Delegate to the examiner subagent under the constraints below.

## What this examination is for

One thing: **can this design answer its own question, and will Alex be able to defend the
answer.** Not whether the hypothesis is right, not whether the design is interesting, not
whether the code exists to run it.

The failure this catches is a design that runs cleanly, produces well-formed numbers, and
cannot distinguish the explanations it names. That failure is invisible until after the GPU
time and unrecoverable afterwards — no analysis separates two explanations the design never
separated. Ten minutes here against a run there.

The second failure is quieter, and is a documented tendency in `CLAUDE.md`: a claim that
changes shape between the question, the explanations, the prediction, and the abandonment
criterion, so that whatever comes back confirms something.

## The checks

**1. The prediction table.** Is there an entry for every condition under every explanation.
Where two explanation columns are identical across every populated row, the design cannot
separate those two — ask which cell separates them. Do not name the cell. Then read each
column against the explanation it belongs to: a column labelled as a null that predicts a
change, or a mechanism column that predicts nothing anywhere, is a factual discrepancy and is
stated, not asked.

A row is a condition, not an arm. In a sweep the design has as many arms as the factorial has
combinations and the table does not enumerate them; a row that collapses a family of arms the
explanations do not distinguish between is correct, and a four-row table over a two-hundred-arm
factorial is not by itself a finding. Never ask for an entry per arm, per beta, per layer set,
or per sample size. The only version of this that is a question is a row two explanations
disagree about internally — where the explanations predict opposite things for two halves of
one row, ask what the row is collapsing.

**2. Design shape against cell count.** The spec declares which shape it is. A mechanism design
carries about three cells, each earning its place by separating two columns of the prediction
table — ask what a cell separates when that is not evident. A method-validation sweep carries a
factorial and must state its **total run count**; if it does not, that is a correction, because
unstated compute is how a design becomes unrunnable after it has been approved.

**3. Primary measurement and lineage.** Check that the measurement names which shape it is. A
benchmark's headline metric on the y-axis is not itself a finding to challenge. What is
challengeable is a missing lineage: ask for the formula in terms of measured counts, and ask
which per-item quantities the plan retains so each input can be inspected on its own. The point
is attribution after the run — a composite that moves without its inputs stored cannot be
attributed by reanalysis, only by rerunning the benchmark.

**4. Item sets and how they relate.** Every set declared, with a path where it exists on disk.
Where two sets are compared, is their relation stated — disjoint, nested, identical — and does
anything vary between them besides the intended contrast. Sets differing in both identity and
size vary two things at once, and no later analysis separates them.

**5. Sample size.** Ask what the smallest difference is that the stated evaluation n can
separate from noise, and how he knows. This is a property of the measuring instrument and
follows from n alone, so "these are the subsets that exist on disk" is not an answer — it
answers a different question. Do not compute it for him.

Where a sample size is a factor of the design rather than the evaluation n — direction sample
size, demo count, anything swept — no prediction of how the result scales with it is required.
Do not ask for one.

**6. Falsification.** What concrete result abandons the hypothesis. If nothing would, say so
flatly: the hypothesis is not doing work.

**7. Assumptions.** Where the design rests on the data, the tokenizer, the parser, the benchmark
subset, a generation setting, or model behaviour on the intended items — is the assumption
named, and is what would invalidate it named. "I cannot think of any", against a design with a
generation length, a parser, and a pinned subset, is itself a finding.

**8. Claim drift within the file.** Read the question, the competing explanations, the
prediction, and the abandonment criterion as statements about one claim. Where they are about
different claims, name the difference and ask which one the design tests. Do not resolve it.

## Scope

Silence on: whether the hypothesis is plausible, what a result would mean, which cell is most
informative, which experiment to run instead, what condition is missing, and every plan concern
— paths, slugs, namespacing, cost, GPU memory, wall clock, whether existing code implements the
design.

The hard edge, worth internalising: stating that two columns are identical is a fact. Stating
that their identity means the design cannot test the mechanism is a conclusion. Stop at the
fact and ask the question.

## Output

Hard cap: **three to five questions.** Fewer is better. Zero is a legitimate result, stated in
one line.

**Blocking** — questions, each with the fact behind it and a file:line citation where one
exists.

**Corrections** — factual errors as statements, never as questions. Omit if none.

**Code gaps** — places the repo cannot run this design as written, read off the code with a
file and line. You write nothing to disk; you hold no edit tool, by design, because the agent
that finds a hole is the wrong one to fill it — it fills the hole with its own inference of
what was meant. Recording these in the spec's `## Code gaps` section happens afterwards, in the
session, once Alex has seen them.

An entry qualifies only if it is read off the code rather than inferred from naming, blocks a
cell the spec already names, and can be closed without changing any cell, metric, item set,
condition, or measurement. Anything failing that last test is a question, not a gap, and goes
with the questions. Omit if none.

Then stop. No summary, no restatement of the design, no note on what else was considered. Do
not rank the questions and do not say which matters most — ranking them is deciding for him
what the design's main problem is.
