# Design spec — VTI_visual_reasoning_validation_07_30_26

Written by Alex, before any plan exists. Copy to `designs/<exp_id>_design.md` and fill in.
Delete every angle-bracket placeholder; the pre-write hook rejects plans whose design spec
still contains them.

---

## The question

A second follow-up question would be, using a more principled application of PCA, say, uncentered SVD-style PCA performed at each layer of the VLMs tested, 

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

Three or fewer. Anything beyond that is a second design.

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

## Primary measurement

A primitive: probability of a token, parsed answer, layer index, count. Not a ratio, rate, or
normalised difference.

<measurement>

If a composite is genuinely wanted, write its lineage in terms of measured quantities and say
why it belongs on the y-axis instead of one of its inputs:

<lineage and justification, or "none">

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
