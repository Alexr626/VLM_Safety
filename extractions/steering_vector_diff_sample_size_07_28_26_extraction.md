# Extraction spec — steering_vector_diff_sample_size_07_28_26

Copy to `extractions/<ext_id>_extraction.md`. Three fields. Delete every angle-bracket
placeholder; the pre-write hook rejects plans whose spec still contains them.

Short on purpose. A bad extraction costs compute; a bad design costs a belief. The gate is
correspondingly light, and everything an implementer works out by reading the repo is absent —
see "Not your job" at the bottom.

---

## Scope check

Covers **producing and storing primitives**: activations, per-layer stacks, attention weights,
per-head values, residual streams, extracted directions, caches.

Does not cover **any comparison, or any number you would read as evidence** — those need
`designs/<exp_id>_design.md`, whether or not they need a GPU.

The test: **would a different value change what you believe?** Producing a tensor cannot come
out wrong in a way that changes a belief; measuring something can.

---

## 1. What data

Plain English. What primitives, over what items, at what sizes, for which models. A paragraph
is enough.

If items are paired or grouped in a way that matters — the same image under two captions, the
same question under two prefixes, several items sharing a source object — say so. Pairing that
is not stated tends not to survive into what gets written.

Similar to the extractions under experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2 And the corresponding extractions for qwen 2.5. I want to create or extract activations at each layer of these models for 295 new examples that need to be generated to the demos v2 list of image plus caption sets. The image has a truthful caption and a set of four hallucinated captions differing only in one visual reasoning dimension, such that the hallucinated captions have one kind of of hallucination each and the all version is a set of hallucinations across all visual reasoning dimensions. I say 295 new examples because there are currently 555 examples. Together, that makes 850 examples, which can be split into sets of 50, 100, 200, and 500 to create a steering vector from each of those sets such that the data used to generate those steering vectors are completely independent of each other, and hence those vectors can be compared for meaningful downstream evaluation and experimentation. 

## 2. How the sets relate

The field that cannot be fixed later. Everything else here is a rerun; this is a rerun you do
not know you need.

For every pair of sets: disjoint, nested, or identical, and why. State it as intent — what
produces the property is the plan's problem, not your sentence to write.

Then, in one line: what is held constant across the sets, and what varies. If more than one
thing varies, say so deliberately rather than by omission.

Every pair is disjoint from each other. There is one partition of the 850 items into sets of 50, 100, 200, and 500, and that same partition is used for all five hallucination dimensions: no image appears in two different size-sets, regardless of dimension. Held constant across the sets: the item pool they are drawn from, the caption construction, and the models. What varies: the sample size, and with it the identity of the items in each set. The intent is to compare downstream the efficacy of the steering vectors produced from those data of different sample sizes.

## 3. What this must support later

No hypothesis needed. What must you be *able* to compute once these exist.

This is what field 2 gets checked against. A comparison that varies two things at once under
your stated structure will not answer what you want, and here is where that costs ten minutes
instead of a re-extraction.

- What needs to be able to be computed downstream are steering vectors from those activations.
---

## Not your job

The examiner is silent on these and the plan resolves them. Do not fill them in:

- Namespacing, slugs, cache layout, key composition, output paths
- Which repo parameter carries which of your requirements
- What already exists on disk and should be reused rather than regenerated
- Forward-pass counts, wall clock, disk, GPU memory feasibility
- Verification checks and manifest design
- Whether existing code produces the structure you asked for, or needs new code to

Two standing rules mean you never write them either. Extractions are additive: nothing already
on disk becomes unreachable or unreproducible. If that is impossible, the plan stops and asks
rather than proceeding. Write a line here only to *override* those defaults.

The plan comes back to you in one case: when resolving something above would change fields 1,
2, or 3.
