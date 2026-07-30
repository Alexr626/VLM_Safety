# Design spec — <exp_id>

Written by Alex, before any plan exists. Copy to `designs/<exp_id>_design.md` and fill in.
Delete every angle-bracket placeholder; the pre-write hook rejects plans whose design spec
still contains them.

---

## The question

<One sentence. One question. If it needs "and", split it into two designs.>

## Competing explanations

At least two. If only one explanation could produce the result you expect, you are not
testing anything.

1. <Explanation A, stated as a mechanism, not a label.>
2. <Explanation B.>
3. <Optional further explanations.>

## Prediction table

One row per condition, one column per explanation. Fill in what each explanation predicts for
the primary measurement — direction and rough magnitude, not just "changes".

This is the load-bearing part of the spec. If two columns are identical across every row, the
design cannot distinguish those explanations and should not be run. Find that out here rather
than after the GPU time.

| Condition | Explanation A predicts | Explanation B predicts |
|---|---|---|
| <condition 1> | <prediction> | <prediction> |
| <condition 2> | <prediction> | <prediction> |
| <condition 3> | <prediction> | <prediction> |

Which single cell does the most work in separating A from B, and why:

<answer>

## Cells

How many cells belong here follows from which shape this design has. Say which, then fill the
table.

**Mechanism design** — about 3 cells. Every cell exists to separate two entries in the
prediction table, and a cell that separates nothing is padding. Anything greatly beyond 3
should be a second experiment design.

**Method-validation sweep** — a factorial is expected and should not be forced into three
rows. The point is coverage of a method's own parameter space, not separation of explanations,
so the honest form is a factor table (factor, levels, where the level came from), then the
arms, then the **total run count stated explicitly** so the compute is visible before it is
spent.

The test for which shape you are in: does this cell exist to distinguish two columns of the
prediction table, or to characterise how the method behaves across its own parameters?

| Cell | Model | Benchmark / subset | Intervention | Layers | Beta |
|---|---|---|---|---|---|
| | | | | | |

## Item sets

Every item set, including any subset, filter, control group, or behaviour-conditional split.
For each: where it comes from, how many items, and one sentence on why it exists. If it already
exists on disk, give the path and call it a filter of that file rather than new construction.

- <name — source — n — purpose>

## Primitives this design requires

Anything the primary measurement needs that is not already on disk — directions, activation
caches, per-layer refits. Leave as "none" when everything exists. Listing a primitive here means
the plan produces it under this spec; no separate extraction spec is written.

The planner treats this section as a whitelist, so a primitive omitted here will not be produced.
Two limits: it must be a strict prerequisite of a measurement named below, and its scope must not
exceed what that measurement consumes. A primitive still worth producing after deleting the
measurement is its own extraction.

For each: what it is, where it will be written, and what existing artifact it must not shadow.

- <primitive — output path — what it must not overwrite, or "none">

## Code gaps

Places where the repo cannot run this design as written. Found by reading the code, not by
reasoning about it: an argument that exists at one layer and is not forwarded by the layer
above it, a metric defined nowhere, a script whose default contradicts the spec, an entry point
that does not accept a factor named in Cells.

**Filled by an agent, not by Alex.** This section exists so that a code gap does not become
Alex's writing task or Alex's patch. An examiner reports gaps in chat; whoever has edit access
records them here. The planner turns each into an implementation step and does not return it as
a question.

The boundary: a gap belongs here only if it blocks a cell already named above, and only if
closing it changes no cell, metric, item set, condition, or measurement. Anything that would
change one of those is not a gap — it goes back to Alex as a question.

- <gap — where the code stops short, with file and line — which cell it blocks, or "none">

## Primary measurement

What goes on the y-axis. Two shapes are legitimate; say which this is.

**A primitive** — probability of a token, parsed answer, layer index, count. Prefer this when
the design is about a mechanism. A primitive has one way to move, so when it moves you know
what moved.

**A benchmark's own headline metric** — accuracy, precision, recall, F1, CHAIR score. This is
correct, not a compromise, when the claim under test is about a method's end-to-end
performance. A hallucination-mitigation method cannot be validated in units of token
probability; the metric the field reports is the quantity the claim is about. The lineage
requirement below applies in full, and replaces the old blanket ban on rates.

<measurement>

### Lineage — required whenever the y-axis is a composite

A composite is anything built from other measured quantities: a ratio, a rate, a normalised
difference, an F1. The problem is not that composites are imprecise. It is that a composite
has more than one way to move, and the number does not say which one happened.

- Accuracy on a yes/no benchmark rises when the model discriminates better, and also when its
  decision criterion shifts toward whichever answer the gold labels happen to favour. These are
  different findings and produce the same number.
- `chair_i` = hallucinated objects / mentioned objects falls when the model hallucinates less,
  and also when it simply mentions fewer objects. A shorter caption improves the score.
- `p_yes_norm` = p_yes / (p_yes + p_no) rises when p_yes rises, when p_no falls, or when both
  fall at different rates — and holds steady while the model stops putting mass on either token
  at all. That is why it is banned outright in `CLAUDE.md` rather than merely flagged here.

Writing the lineage does not make a composite safe, and it is not a justification exercise. It
does one mechanical thing: it names the measured counts underneath, which obliges the plan to
store and report them per item. That is what makes it possible, after the run, to say which
input moved. Skip it and the denominator is discarded at aggregation time — at which point the
attribution cannot be recovered by reanalysis, only by rerunning the benchmark. The cost of the
rule is ten minutes; the cost of skipping it is the run.

For each composite on the y-axis:

- Its formula in terms of measured counts, with the file and line where this repo defines it.
- The per-item quantities the plan must retain so each input can be inspected on its own.
- One sentence on why the composite is on the y-axis rather than one of its inputs.

<lineage, retained inputs, and justification — or "none, the y-axis is a primitive">

## Sample size

n per cell: <n>

Why that n — what size of effect it can separate from zero, and how you know:

<answer>

## What I predict

Write this before the run. It is not scored and no one else reads it; its only job is to exist
before the numbers do, so that your reading afterwards cannot quietly become what the data
said.

<your prediction, with numbers where you can>

## What would make me abandon the hypothesis

<A concrete result. If nothing would, the hypothesis is not doing any work and this design
should not be run.>

## Assumptions that need checking first

Anything this design rests on regarding the data, the tokenizer, the extraction, the benchmark
subset, the parser, or model behaviour on the intended items. For each, what result would
invalidate the design.

- <assumption — what invalidates>
