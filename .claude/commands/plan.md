---
description: Expand an approved design spec into a repo-grounded implementation plan. Usage /plan designs/<MM_DD_YY>/<name>.md or extractions/<MM_DD_YY>/<name>.md
---

<!-- Last updated: 2026-07-30 -->

Spec: $1

**Resolving the spec.** `$1` is a repo-relative path to the spec file, pasted from the editor:
`designs/<MM_DD_YY>/<name>.md` for an experiment, `extractions/<MM_DD_YY>/<name>.md` for a
primitive. Take it literally. Do not glob, do not search for near matches, do not try `_design`
or `_extraction` suffix variants.

- Append `.md` if it is missing. Strip a leading `./`.
- If the path does not exist, or starts with neither `designs/` nor `extractions/`, say so,
  name the path, and stop — do not draft a spec and do not sketch what one might contain.
- Only if `$1` contains no `/` at all is it a bare id. Then, and only then, glob
  `designs/**/$1.md` and `extractions/**/$1.md`; if that matches more than one file, list every
  match and ask which. Never guess between the two directories.

The leading directory settles the spec kind, and with it the declaration the plan must carry
for the pre-write hook: `designs/` → `design_spec: $1`, `extractions/` → `extraction_spec: $1`.
The spec id is the filename stem.

Everything below that names a spec path means this path, repo-relative.

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

## Sanity checks come first, and there are at most five

A sanity check exists for one reason: an assumption **the experiment** rests on could be false
in a way that makes the numbers meaningless, and finding that out afterwards costs the run. It
is checked **before** the main experiment, not alongside it.

**Hard cap: five.** Fewer is better, zero is a legitimate result stated in one line, and a check
is never added for completeness or because a section looks thin. If more than five candidates
survive, the design rests on more unverified premises than it can carry — return the question
instead of writing a longer list.

**Qualifies:** an assumption about the data, the tokenizer, the parser, a benchmark subset, a
generation setting, or model behaviour on the intended items, where a plausible outcome would
change what the experiment measures or invalidate a named cell. The spec's Assumptions section
is the primary source; anything else needs one sentence naming the cell it protects.

**Never qualifies:** how the code works, what a function returns or accepts, whether an entry
point exists, where artifacts are written, what a cache directory holds, whether an existing
artifact has the properties its extraction recorded, whether a pinned file resolves, whether a
partition is disjoint, whether a hash agrees. Every one of those is a read the planner performs
now. Environment, throughput, and memory go under a separate prerequisites heading and do not
count toward the five.

**`IMPLEMENTATION.md` is consulted before any check is proposed.** It is the standing record of
what has already been established about this repo — module APIs, hook sites, registry keys, data
paths, metric schemas, and the properties of the extracted artifacts on disk. If it answers the
question, the check does not exist: write the fact into the plan and cite the line. If it is
silent and the repo can settle the question, settle it during planning and record it as a
resolved fact with a file and line. A check is only what survives when neither route closes it.
Where the file and the code disagree, the code is reality — say so in open questions so the file
gets fixed.

Name each surviving check, say what it verifies and what result would invalidate the design, and
put it in a section that gates the rest of the plan. Where the checks are substantial, write
them as their own plan file first and say in chat that the main plan waits on their results —
the exception, not the default, and the cap applies to that file too.

This project has run a full experiment on a broken data premise. Treat the sequencing as hard.
The cap does not weaken that: a list long enough to bury the two checks that matter is how the
premise gets missed again.

## Plan format

Filename: plain English with a date suffix, under `implementation_plans/`, for example
`toward_yes_clause_grid_plan_2026-07-28.md`. Revising means writing a new dated version and
leaving the old one in place — plan lineage is load-bearing, because artifacts trace back
through it.

The file must carry exactly one of `design_spec:` or `extraction_spec:` on its second line,
followed by the **resolved spec path exactly as it sits on disk**, repo-relative and including
any subdirectory — `design_spec: designs/07_30_26/toward_yes_grid.md`. Do not reconstruct the
path from the id; copy the path you resolved. A pre-write hook enforces that exactly one
declaration is present, that it points inside the matching directory, and that the file it
names exists and is not an unfilled template. **The hook is not an obstacle to work around.**
If it fires, the spec does not exist in usable form and the plan should not be written.

A design spec that carries its own required primitives still declares only `design_spec:`. Two
declarations are rejected.

Sections, in plain English: Question, Design spec reference, Cells, Data, Metrics, Artifacts,
Sanity checks that must pass first, What confirms or falsifies, Open questions.

**No minted shorthand.** No gate codes, no analysis codes, and no letter-number labels for
phases, stages, steps, blocks, arms, or any other part of the run. Each is named by what it is —
"the AMBER baseline", "the all-layers arm at 50 demos", "the late-window block" — however much
longer that runs. Defining a code once and using it consistently does not satisfy this.

The reason is downstream, not aesthetic: every code a plan invents gets copied verbatim by the
implementer into filenames, run tags, directory names, identifiers in code, log lines, plot
titles and column headers, where it outlives the plan. A reviewer looking at the resulting figure
cannot recover what `C3` meant without finding the plan and the right version of it, and the cost
falls entirely on whoever reads the results.

One exception: the Cells table may carry a short cell label in its own column when there are
enough parallel cells that the table is unreadable without one. It stays a table column and never
becomes a name used in prose; every other section refers to the cell by description.

Metrics named in plain English and stated as primitives where they are primitives. Where the
spec's y-axis is a composite, carry its lineage — the formula in measured counts, and the
per-item quantities to retain — from the spec into the plan, and carry the spec's stated reason
for reporting it. Do not restate the reason in better words.

Artifact filenames self-describing and readable without the plan in hand:
`accuracy_by_gold_label_and_clause_direction.png`, not `c3_B1_grid.png`. Same for plot titles
and axis labels.

Be specific about function signatures, command invocations, output paths, and result formats.
Do not pad. End with the open questions.
