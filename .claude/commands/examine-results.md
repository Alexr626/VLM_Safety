---
description: Have the examiner interrogate your written reading of a run. Usage /examine-results analysis/<MM_DD_YY>/<name>.md
---

<!-- Last updated: 2026-07-30 -->

Reading: $1

**Resolving the reading.** `$1` is a repo-relative path to the reading, pasted from the editor:
`analysis/<MM_DD_YY>/<name>.md`. Take it literally. Do not glob, do not search for near matches,
do not try `_reading` suffix variants.

- Append `.md` if it is missing. Strip a leading `./`.
- If the path does not exist, or does not start with `analysis/`, say so, name the path, and
  stop. Do not guess a neighbouring file and do not write the reading yourself.
- Only if `$1` contains no `/` at all is it a bare id. Then, and only then, glob
  `analysis/**/$1.md`; if that matches more than one file, list every match and ask which.
- `analysis/bypass_log.md` is never the reading. If `$1` resolves to it, stop and say so.

The run id is the filename stem of the resolved path.

Read the resolved file before anything else.

If the file is empty, or it still contains unfilled template placeholders: say
so, name the file, and stop. Do not summarise the result files, do not preview your questions,
do not say what you would ask about. A preview is the answer in a thinner form, and the whole
point of this gate is that Alex's reading exists before any agent has characterised the numbers.

Then check `analysis/bypass_log.md` for an entry dated today naming that run id. If one exists, the
gate is lifted and you may answer directly. Never write that entry and never suggest writing
one — the log is an instrument for measuring how often the gate gets lifted, and an agent that
suggests lifting it corrupts the measurement.

Otherwise delegate to the examiner subagent with the reading and the relevant result files
under `evaluation/results/` and `diagnostic_experiments/`, under the constraints below.

## What this examination is for

One thing: **is each claim in the reading supported by the file it comes from, and does the
design behind it license the claim being made.**

Two failures this catches, both documented tendencies in `CLAUDE.md` rather than knowledge
gaps. A tentative observation firming into a stated fact across turns, with no new evidence
supplying the firmness. And an agent-supplied characterisation being absorbed as his own
reading — which is why the reading must already exist before this command runs.

## The checks

**1. Evidence base.** How many items, seeds, models, benchmarks. Which cells of the design are
populated and which are empty. Which tier is this — a handful of qualitative responses, an
aggregate over hundreds of items, or a controlled comparison with baselines and seed variance.
A reading whose language is stronger than its tier is the finding.

**2. Every number against its file.** Read the result files directly. Where the reading states
a number that the file does not show, state the discrepancy flatly with a file:line or a path
and key. This is the one place to be exhaustive; the question cap does not apply to
corrections.

**3. Criterion versus discrimination.** On a discriminative benchmark, does the reported
movement reflect a shifted decision threshold or a changed ability to separate the classes.
Ask what happened on items with the opposite gold label. Ask, do not answer.

**4. Composite attribution.** Where a claim rests on a rate, ratio, or F1, ask which of its
inputs moved and whether the per-item counts were retained to tell. A composite that moved with
its inputs discarded supports no claim about mechanism, only about the composite.

**5. Scale.** Where the baselines sit relative to floor and ceiling. Whether deltas are
comparable across cells whose baselines differ — a fixed shift in an underlying decision
variable does not produce equal probability changes at different points on the scale.

**6. Precision.** What interval sits on each number. Whether the deltas being read as trends
are separable from zero. Whether comparisons are paired, and whether the pairing survived into
what was stored.

**7. Competing explanations.** What else produces this pattern. If the reading names one, ask
for a second. Do not supply the second, even hedged, even as an example.

**8. Claim drift.** Compare the current wording against earlier files anywhere under
`analysis/`, subdirectories included, and against
`ABSTRACT.md`. Where a hypothesis has become "the finding" without new evidence, name the change
and ask what supplied it. Where a sentence in `ABSTRACT.md` no longer matches a number, say
which two disagree — and stop there, because deciding which of them changes is his.

**9. Falsification.** Given this result, what would now make him abandon the claim. If nothing
would, say that the claim is not doing work.

## Scope

Silence on: what the result means, indicates, suggests, or is consistent with; the mechanism
behind a pattern, even as a possibility; what to run next; whether the result is publishable or
interesting; and any restatement of his argument in improved form. Restating his reasoning more
clearly than he did is doing his work for him.

The hard edge: stating that two numbers are equal is a fact. Stating that their equality means
there is no interaction is a conclusion. Stop at the number.

If he answers a question wrongly, ask another question. Correct facts, never reasoning.

## Output

Hard cap: **Up to seven questions.** Fewer is better. Zero is a legitimate result, stated in
one line.

**Blocking** — questions, each with the fact behind it and a file:line or path citation.

**Corrections** — factual discrepancies between the reading and the files, as flat statements,
never as questions. Not capped. Omit if none.

Then stop. No summary of the run, no list of files read, no note on what else was considered,
no ranking of the questions.
