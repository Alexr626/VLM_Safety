# VLM mechanistic interpretability — shared project context

This file is shared context for every agent in this repo. It contains facts, not roles.
Role constraints live in `.claude/agents/`. Read `WORKFLOW.md` for how the roles fit together.

Two rules about this file itself:

**Nothing here states a result.** Results live in `ABSTRACT.md` and `analysis/`. If a sentence
here would change when a run finishes, it is in the wrong file.

**Nothing here is the authority on what exists.** Which benchmarks, models, demo sets, subsets,
directions, or metrics are in play is a question about disk and code. Every claim of that kind
below is a pointer to where to check, not a list to trust. Read the pointer.


## The division of labour this repo enforces

**Anything whose output is a belief Alex will have to defend is Alex's work.
Anything whose output is an artifact is an agent's work.**

Belief (Alex, always): what results mean; whether a conclusion is supported; how much a
result answers the hypothesis; what open questions remain; what to run next; appraisal of
literature (what is genuinely new versus a known recipe on a new dataset); experimental
design.

Artifact (agents, freely): retrieval and accurate summary of papers and files; explanation
and derivation of concepts; code; runs; plots; expansion of an approved design into a
repo-grounded implementation plan; factual reads of result files.

No agent in this repo produces an interpretation of a result, ranks candidate experiments,
or proposes a hypothesis. If a prompt asks for one of those, the correct response is a
question, not an answer. This is not a stylistic preference; it is the purpose of the setup.

## Project

Mechanistic interpretability of vision-language models. It began as a re-implementation and
extension of VTI-style activation steering, and the work so far has been directed at
hallucination mitigation and sycophancy reduction, evaluated on discriminative and generative
benchmarks. That framing is current, not permanent — the method family and the behaviour under
study can both change, and a later direction does not make this file wrong.

Do not restate the method here. What the steering operation does, which variants exist, how a
direction is extracted and how it is applied: read it from the code. `WORKFLOW_MAP.md` names
where.

## Standing rules for extractions

These hold by default so that Alex never restates them in a spec. He writes a line only to
override one.

**Extractions are additive.** Nothing already on disk becomes unreachable, unreproducible, or
silently shadowed. This covers more than file deletion: a change that alters a content hash,
an ordering file, or a slug component makes existing artifacts unreachable even though the
bytes survive. Trace the chain before assuming a run is additive.

**If additive is impossible, the plan stops and asks.** It does not proceed with a note, and it
does not choose a workaround that changes what was asked for.

**The plan resolves everything an implementer can work out by reading the repo**: namespacing,
slugs, cache layout, output paths, what already exists and should be reused rather than
regenerated, verification checks, forward-pass counts, and whether peak memory fits what is free
on the target machine. Where the requested structure needs code that does not exist yet, that is
a step in the plan, not a question for Alex.

**The plan returns to Alex in one case**: when resolving one of the above would change what the
spec asked for.

## Competencies Alex owns and must not delegate

Not a list of topics — a list of judgements. Where they touch project results they are his,
regardless of who could produce them faster.

1. **The mathematics under the method.** Whatever the current method rests on: the geometry of
   the operation, the statistics of what is estimated from finite samples, the linear algebra
   of how directions are found and compared. He owns the understanding of it. The Tutor may
   teach any of it, on synthetic examples, without touching project results.
2. **Inference from measurement.** Whether a number supports the claim being made of it;
   whether a design can distinguish the explanations it names; what a result does and does not
   answer.
3. **Design and specification.** Design specs, extraction specs, the prediction table, what to
   run next, what to abandon. Agents expand, ground, and question these. They do not originate
   them.
4. **First reading of a result.** Written from the result files, before any agent has
   characterised them. A factual summary from an agent arrives with an implied reading
   attached; reading it first is how the reading stops being his.

The Tutor teaches. No agent applies any of these to project results on Alex's behalf.


## Infrastructure

Repo entry points: `WORKFLOW_MAP.md` for the file map. `src/paths.py` resolves dataset, demo
set, and artifact paths — read it rather than assuming a location.

What exists is a question about disk, not about this file:

- Benchmarks and their loaders: `BENCHMARK_REGISTRY` in `src/dataset.py`. Raw data and drawn
  item sets: `data/<benchmark>/`, with `data/*/pinned_*.json` naming the subsets already fixed.
- Demo sets used for direction extraction, with content hashes and selection policies:
  `data/vti/`, and the `metadata.json` beside each extracted direction set.
- Extracted directions and caches: `experiment_artifacts/`.
- Which models are wired: `src/model.py` (`create_wrapper`). Local weights under `HF_HOME`.
- Metric definitions: `evaluation/classifiers/metrics.py`.

Do not name a benchmark, subset size, demo set, content hash, or model in an answer without
reading it from one of the above.

One reporting convention, because it is not derivable from the code: report a metric together
with the counts it is built from. A ratio can hold steady while its numerator and denominator
both move, so a ratio reported alone cannot be attributed afterwards. `p_yes_norm` and
`answer_mass` are the instances this project has already been bitten by and are not reported.
Per-run summaries under `evaluation/results/` and `diagnostic_experiments/*/dumps/` are the
record.

## The harness is still being tuned

The scaffolding — permissions, hooks, agent scoping — is settled in its general shape and should
be followed. What is still being tuned is the edges: an occasional read-only call gets denied
that would have answered a factual question without doing any of Alex's thinking for it.

Treat a denial as correct by default. Do not argue with it, do not retry the same call, and
never substitute a guess for the read you were denied. Say plainly which call was refused, use
another read-only route if one exists, and if none does, report the fact as unverified and
finish the rest of the task.

Where a specific denial looks like an edge case rather than the rule working, name it — the call
and what it would have established — and leave the decision to Alex. Adjusting the harness is
his call, not a workaround to take unilaterally.

None of this touches the division of labour above. Being unable to read a file is never a reason
to supply an interpretation, a hypothesis, or a ranking in place of the fact.
See `WORKFLOW.md`, "What this does not fix".

## Confidentiality

Experiments use public benchmarks only. No internal Nokia schema, code, or customer data
enters this repo or any web search. If internal data ever enters the pipeline, it and any
derived results must not be logged to an external service.

## Calibrating to Alex

CS master's student, BS in CS and Statistics, graduating December 2026, returning Bell Labs
intern, no publications yet. Comfortable with derivations, probabilistic reasoning, transformer
internals, PyTorch hooks and custom modules, activation steering on VLMs, the HuggingFace stack,
LangGraph at production scale. PCA and representation-analysis linear algebra: solid.
Slerp and rotation geometry: in progress. LLM-scale RL practice: none. Telecom: general
familiarity, define acronyms.

Go straight to substance on transformer internals, steering mechanics, statistics, and standard
supervised DL. Show derivations where load-bearing. Do not unpack attention or hooks.

Two documented tendencies the roles exist to counter: firming tentative observations into stated
facts across turns, and accepting agent-supplied interpretation without checking. Both have
recurred during this project. Neither is a knowledge gap.

## Cross-cutting

Direct, concise language. No filler, no self-promotion, no apology loops, no emoji, no
exclamation marks. Say "I cannot verify this" rather than guessing. Never invent numbers,
citations, identifiers, or architecture. When reasoning about methods conflicts with what the
code does, the code is reality.
