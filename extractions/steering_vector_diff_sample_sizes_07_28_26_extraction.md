# Extraction spec — steering_vector_diff_sample_sizes_07_28_26

Written by Alex, before any extraction plan exists. Copy to
`extractions/<ext_id>_extraction.md` and fill in. Delete every angle-bracket placeholder; the
pre-write hook rejects plans whose spec still contains them.

This is the lighter of the two gates. It exists because producing primitives ahead of a
hypothesis is legitimate and often the right use of waiting time — but it still carries
commitments, and two of them (independence structure, and what the run forecloses) determine
whether the artifacts are usable later.

---

## Scope check

An extraction spec covers **producing and storing primitives, and verifying they are what they
claim to be.** Activations, per-layer stacks, attention weights, per-head values, residual
streams, extracted directions, caches — the artifact type is not the constraint.

It does not cover **any comparison, or any number you would read as evidence.** Cosine between
two directions, a reliability curve, a benchmark score, a contrast between two conditions:
those are experiments and need `designs/<exp_id>_design.md`, regardless of whether they need a
GPU. Most of them do not.

The test when unsure: **would a different value change what you believe?** If yes, it is an
experiment. Producing a tensor cannot come out "wrong" in a way that changes your mind;
measuring something can.

Descriptive checks on the artifacts themselves stay here — file counts, shapes, norms of a
single extracted object, sanity assertions. The moment two arms are contrasted, it moves.

---

## What is produced

Primitives, and for each: which model, which demo id set, which variants, which derangement or
none, and at what n.

| Artifact | Model | Id set | Variants | n | Where it lands |
|---|---|---|---|---|---|
| activations | LLaVA-1.5 | 50/100/200/500 | Independent | 850 | experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2 |
| activations | Qwen-2.5 | 50/100/200/500 | Independent | 850 | experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2 |

## Independence structure

For every pair of id sets above, state whether they are disjoint, nested, or identical, and why
that choice was made.

This is the field that does the work the prediction table does in a design spec. Nested prefix
sets share their errors; two directions drawn from them cannot be treated as independent
estimates, and no later analysis can undo that. Getting it wrong here is not recoverable by
re-analysis, only by re-extraction.

They are all disjoint sets, built for comparison of steering vectors meant to capture caption truthfulness at varying sample sizes.

## Comparisons this must support later

You do not need a hypothesis to fill this in. You need to state what you must be *able* to
compute once the artifacts exist. "Cosine between two independent estimates at matched n=250."
"Attention entropy under two prefix conditions on the same items."

Each line is checkable against the artifacts afterwards. If a comparison you later want is not
supported by the independence structure above, that is visible here rather than after the run.

- Able to compute cosine similarity of steering vectors representing truthfulness at different sample sizes across independent data.

## What already exists

What is already on disk that this does not duplicate, with paths. Note in particular anything
where the primitive is invariant to the parameter being varied — for instance, a contrast where
the image never changes needs no new forward passes at any sample size.

experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2 and The equivalent for lava already exists, but the steering vectors Calculated at sample sizes lower than 500 are done so using data that are subsamples of the 500 set, meaning the data used to calculate the vectors are not independent, meaning they do not provide for meaningful comparison of the efficacy of of larger sample size and calculating truthfulness for downstream experimentation. Anot. This particular situation where a subset of the existing artifacts needed for downstream experimentation already existing should be, serve as inclination to use the existing experimental artifacts and primitives to to create the necessary independent steering vectors by simply creating 350 more such captioned images and extracting activations for them. Then with the total pool of 850 activations, create independent steering vectors of the sizes specified above. 

## What this forecloses

Every cache invalidated, file overwritten, or namespace collided. Whether the run is reversible,
and if not, what is lost permanently.

Fill this in even when the answer is "nothing." The field exists because a destructive
extraction is discovered here or in a code review, and code review is later and more expensive.

No cache is invalidated for the non-shuffled Steering vector extraction at different sample sizes. Once shuffling is considered and image-shuffled based vectors are desired for corresponding vectors of unshuffled corresponding sample sizes, then this will change. 

## Namespacing

How the new artifacts are keyed so they cannot collide with or silently shadow existing ones.
Where a cache key omits a parameter that changes its contents, say how that is fixed before the
run rather than after.

demosv2_9a44f4af_all_nd<sample_size>_independent_s42_r2_prefix

## Cost

Forward passes, GPU hours, wall clock, disk. Which machine.

lambda, b2, gpu, 0 

## Verification

How you will know the artifacts are what they claim to be. File counts, shapes, id manifests,
a spot check that a cached entry corresponds to the input it is keyed to.

- scripts to verify that the subsets of images from the 850 total examples contain no overlapping subsets among the 50, 100, 200, 500 splits. 
