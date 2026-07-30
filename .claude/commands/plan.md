---
description: Expand an approved design spec into a repo-grounded implementation plan. Usage /plan <exp_id>
---

Spec id: $1

Read `designs/$1_design.md` or `extractions/$1_extraction.md`, whichever exists. If neither
exists, say so and stop — do not draft one, do not sketch what one might contain. If both
exist, ask which this plan is for rather than guessing from the id.

Delegate to the planner subagent under the constraints below.

## What this expansion is for

The spec says what to run and why. The plan says how, in enough detail that Cursor implements
it without guessing and without inventing. **The planner supplies grounding, precision, and
repo reality. It does not supply judgement.**

The value is in what the plan removes: every place an implementer would otherwise have to make
a call. A plan that says "extract directions at each sample size" has moved the decision, not
made it. A plan that names the function, its arguments, the output path, the slug, and what is
already on disk and must be reused has.

## The whitelist

The spec is a whitelist. Cells, conditions, metrics, models, benchmarks, item sets, seeds,
hyperparameters, and primitives appear in the plan only if they appear in the spec.

This holds for constructions that feel like consequences of the analysis rather than additions
to it. A flip-recovery analysis implies a set of items that flipped; a matched comparison
implies a control set; a per-stratum breakdown implies strata. None of them enter unless the
spec names them. Where the analysis genuinely needs a set the spec does not name, that is an
inadequate design: **return the question, write nothing.**

One section is exempt. `## Code gaps` is not subject to the whitelist — each entry is an
implementation step to write into the plan, not a question to return, provided closing it
changes no cell, metric, item set, or condition. An entry that would change one of those was
misfiled and goes back as a question.

If the design is flawed, underpowered, confounded, destructive in a way its own spec does not
account for, or unable to answer its own question — say so and return questions. Do not fix it,
do not write a better version. An improved design the planner wrote is a design Alex cannot
defend.

## What the planner resolves silently

None of this returns to Alex. All of it is answerable by reading the repo, and answering it is
the job:

- Namespacing, slugs, cache layout, key composition, output paths
- What already exists on disk and should be reused rather than regenerated — check
  `data/*/pinned_*.json`, existing dumps, cell manifests, and extracted direction directories
  before describing any new construction
- Whether the requested structure needs code that does not exist yet — that is a step in the
  plan, written as new, not a question
- Verification checks and manifest design
- Forward-pass counts, wall clock, disk
- Peak memory against what is actually free on the target machine, checked before writing

**It returns to Alex in one case:** when resolving one of the above would change what the spec
asked for.

## Grounding

Every identifier must be real and verified against the repo, not recalled. Model strings,
`BENCHMARK_REGISTRY` keys, function and argument names, config fields, output paths. Where the
plan needs something that does not exist, mark it explicitly as new rather than writing it as
though it were already there.

Two standing rules from `CLAUDE.md`. Extractions are additive — and additive means *reachable*,
not merely present, so a change to a content hash, an ordering file, or a slug component
strands existing artifacts without deleting a byte. Trace that chain explicitly. Where additive
is impossible, the plan stops and asks; it does not proceed with a note.

Constraints that have bitten this project before, and are worth checking every time: activation
cache keys that omit a parameter which changes their contents; `device_map` sharding silently
breaking hook-based interventions, so extraction goes to RunAI unsharded; a contrast where the
image never changes needing no new forward passes at any sample size.

## Sanity checks come first

Where the design rests on an assumption about the data, the tokenizer, the extraction, a
benchmark subset, the parser, a generation setting, or model behaviour on the intended items,
that assumption is checked **before** the main experiment runs, not alongside it.

Name each check, say what it verifies and what result would invalidate the design, and put them
in a section that gates the rest of the plan. Where the checks are substantial, write them as
their own plan file first and say in chat that the main plan waits on their results.

This project has run a full experiment on a broken data premise. Treat the sequencing as hard.

## Plan format

Filename: plain English with a date suffix, under `implementation_plans/`, for example
`toward_yes_clause_grid_plan_2026-07-28.md`. Revising means writing a new dated version and
leaving the old one in place — plan lineage is load-bearing, because artifacts trace back
through it.

The file must carry exactly one of `design_spec: designs/$1_design.md` or
`extraction_spec: extractions/$1_extraction.md` on its second line. A pre-write hook enforces
that exactly one declaration is present, that it points inside the matching directory, and that
the file it names exists and is not an unfilled template. **The hook is not an obstacle to work
around.** If it fires, the spec does not exist in usable form and the plan should not be written.

A design spec that carries its own required primitives still declares only `design_spec:`. Two
declarations are rejected.

Sections, in plain English: Question, Design spec reference, Cells, Data, Metrics, Artifacts,
Sanity checks that must pass first, What confirms or falsifies, Open questions. No gate codes,
no analysis codes. Cell shorthand is fine inside the Cells table when there are enough parallel
cells to make it useful; nothing else uses codes.

Metrics named in plain English and stated as primitives where they are primitives. Where the
spec's y-axis is a composite, carry its lineage — the formula in measured counts, and the
per-item quantities to retain — from the spec into the plan, and carry the spec's stated reason
for reporting it. Do not restate the reason in better words.

Artifact filenames self-describing and readable without the plan in hand:
`accuracy_by_gold_label_and_clause_direction.png`, not `c3_B1_grid.png`. Same for plot titles
and axis labels.

Be specific about function signatures, command invocations, output paths, and result formats.
Do not pad. End with the open questions.
