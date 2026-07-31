# Extraction spec — shuffle_control_vector_diff_sample_size_07_29_26

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

Similar to the extraction under experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control, Using the new sets of 50, 100, 200, and 500 independent samples of truthful and hallucinated captions over independent images, I want to create a control vector generated with activations from inputs of shuffled images with captions that don't pertain to the image such that I can create a steering vector that acts as a control and corresponds to each of the newly created 50, 100, 200, and 500 sample size vectors. The newly created control vectors should use the same exact data sets, data subsets that the 50, 200, and 500 vectors were created from in which the shuffled images are within sample. This should allow for a direct comparison between the vectors generated for control with the vectors generated for actual steering. 

## 2. How the sets relate

The field that cannot be fixed later. Everything else here is a rerun; this is a rerun you do
not know you need.

For every pair of sets: disjoint, nested, or identical, and why. State it as intent — what
produces the property is the plan's problem, not your sentence to write.

Then, in one line: what is held constant across the sets, and what varies. If more than one
thing varies, say so deliberately rather than by omission.

Similar to the creation of the steering vectors of each sample size each data set that creates the control vectors will be disjoint the data sets themselves are already defined from the last run. The only thing that needs changing is which image is passed at each input such that When each new set of example, truthful, and hallucinating captions are passed, the image corresponding to those captions is randomly shuffled to be an image that has no relation to the captions that is describing it. 

## 3. What this must support later

No hypothesis needed. What must you be *able* to compute once these exist.

This is what field 2 gets checked against. A comparison that varies two things at once under
your stated structure will not answer what you want, and here is where that costs ten minutes
instead of a re-extraction.

- These vectors will be used to compute cosine similarities and other comparisons against the true steering vectors to measure how well they capture the concept of truthfulness in activation space in relation to truthfully describing an image. Since each set of captions will be describing an image that has No relation to the actual image that is passed at each forward pass. These control vectors will measure the actual steering vector's abilities to capture signal in which the textual descriptions are related to the images themselves. 
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
