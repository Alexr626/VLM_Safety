# Extraction spec — steering_vector_validation_continuation

## Scope note

This spec produces evaluation metrics, which the scope check below classifies as an experiment,
not an extraction. It sits in `extractions/` deliberately and with no prediction table, because
there is no hypothesis yet about how either model behaves on the expanded AMBER set under the
VTI-derived textual steering vector. The reason for running it now is that the statistical
method for the matched-pair comparison is still under review, and generation takes long enough
that waiting for the method would cost days.

Two consequences, stated so that whoever reads the resulting files knows the terms:

- The metric files this produces are not to be read as evidence until an experiment design
  specification exists under `designs/`. That specification is to be written after the data is
  generated and will say what the comparison is.
- The comparison this data is meant to support is a matched-pair, dependent-sample comparison
  between configurations. The count of discordant pairs governs whether such a comparison can be
  run at all, but it is not what sets the item count below — see field 1.

---

## Scope check

Covers **producing and storing primitives**: activations, per-layer stacks, attention weights,
per-head values, residual streams, extracted directions, caches.

Does not cover **any comparison, or any number you would read as evidence** — those need a spec
under `designs/`, whether or not they need a GPU.

The test: **would a different value change what you believe?** Producing a tensor cannot come
out wrong in a way that changes a belief; measuring something can.

---

## 1. What data

Plain English. What primitives, over what items, at what sizes, for which models. A paragraph
is enough.

If items are paired or grouped in a way that matters — the same image under two captions, the
same question under two prefixes, several items sharing a source object — say so. Pairing that
is not stated tends not to survive into what gets written.

Answer - similar data to what was collected for the experiment defined by implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md, but with some slight modifications:

1. The data should only be collected for a steering vector generated at a sample size of 500.
2. The data that will be used for evaluation should only be an expansion of the Amber 450 set to include more examples sufficient for matched pair dependent sample comparisons. I want to skip POPE and CHAIR for now to dig deeper into the current results on AMBER, after the data has been generated.
3. The data should be collected when the VTI based textual string vector is applied for comparison with baseline and simple mean-difference vector results.

**Size of the expanded AMBER set: 1500 discriminative items in total.** I am not sizing this
draw against a target number of discordant pairs. Reviewing the contingency tables of the
existing Qwen results makes it clear that some configurations — the early and middle layer
windows — produce almost no flips at all, so no single draw size makes every configuration
testable. 1500 is simply the subset size; the discordant count falls where it falls.

[Supporting counts filled by agent, read from
`evaluation/results/2026-07-30/_analysis_steering_visual_reasoning_validation/hallucinations_induced_and_removed_vs_baseline.csv`.
At nd500 and n=450, `n_decision_flips` against baseline is below 10 in 11 of the 18
(model, coefficient, layer window) cells, and Qwen at layers 5–14 gives 0, 0, 1 across the three
coefficients.]

**Items per question type: 500 each.** 500 existence, 500 attribute, 500 relation.

**Ground-truth balance of the expanded set.** I do not want to carry forward the yes/no balance
of the existing 450. The 1500 items should come as close as possible to the proportion of
yes ground-truth answers per question type in the whole AMBER discriminative benchmark:

| question type | n in benchmark | yes | no | pct yes |
|---|---|---|---|---|
| existence | 4924 | 0 | 4924 | 0 |
| attribute | 7628 | 3814 | 3814 | 50 |
| relation | 1664 | 975 | 689 | 58.59 |

The finer annotation types under attribute (action, number, state) are ignored for this.

[Per-cell counts derived by agent, arithmetic from the two constraints above and nothing else:
existence 0 yes / 500 no; attribute 250 yes / 250 no; relation 293 yes / 207 no at 58.59%.
Overall 543 yes / 957 no, 36.2 percent yes.]

**Grouping.** The unit I treat as independent for a discordant-pair count is an image with a
unique prompt: two items may share an image so long as their prompts differ. The pinned 450
draws roughly 2 items per image, against as many as 18 available per image in the pool, and that
sparseness is much preferable. The more distinct the images across the drawn items, the better.

**Models and grid: exactly as in the 7-30 run.** LLaVA-1.5-7B and Qwen2.5-VL-7B-Instruct;
steering coefficient 0.2, 0.5, and 0.9; layer windows all, the early window, and the late
window, with each model keeping the windows it used before. [Layer indices filled by agent,
read from the run slugs under `evaluation/results/2026-07-30/*/amber/`: early is layers 5–14
for both models; late is layers 20–29 for LLaVA and layers 15–24 for Qwen.]

## 2. How the sets relate

The field that cannot be fixed later. Everything else here is a rerun; this is a rerun you do
not know you need.

For every pair of sets: disjoint, nested, or identical, and why. State it as intent — what
produces the property is the plan's problem, not your sentence to write.

Then, in one line: what is held constant across the sets, and what varies. If more than one
thing varies, say so deliberately rather than by omission.

Answer: There are two sets of relevant data to this data collection. One, the AMBER subset that I'm asking for, which is the one and only set of data that I want done for evaluation across the different configurations of models, steering coefficient and so on. and to the set of 500 demo examples used to calculate the steering vector either through a simple mean difference as per the current experiment setup or using the VTI extraction approach. Of course, the Amber subset and the 500 demo examples are completely disjoint.

**The expanded AMBER set of 1500 is a strict superset of the pinned 450**
(`data/amber/pinned_amber_disc_450.json`). The 450 keep their existing item ids. The 1050 items
added are drawn to bring the whole 1500 as close as possible to the per-question-type yes-ratio
stated in field 1, which means the added items do not have the same balance as the original 450.

**No responses are reused. Every configuration is generated fresh over all 1500 items.** The
2026-07-30 responses for the 450 are not carried forward, so no configuration mixes responses from
two runs and nothing in this run depends on those responses reproducing. That reproducibility
cannot be established from the repo: the run manifest records `git_commit: f351b45`, while the
commits introducing the `--directions_dir` / `--layer_set` code that run used — `e486a4a`,
`04ed0fc`, `97be07c` — all come after it, so the code that ran was uncommitted working-tree state
that git does not record. Regenerating removes the question rather than answering it.

The superset relation is kept for a separate reason, not for reuse: it leaves the 450 items
evaluated twice under the same settings, once on 2026-07-30 and once in this run, in the baseline
and meandiff-nd500 cells.

**Held constant across every evaluation configuration:** the 1500-item AMBER set itself, the same
items in the same order in every configuration, and the demo sample size of 500 used to produce
the steering vector.

**What varies:** model (two), steering vector condition (no intervention, simple mean-difference,
VTI), steering coefficient (0.2, 0.5, 0.9), and layer window. More than one thing varies across
the grid as a whole, deliberately — it is a factorial, and the matched-pair comparisons it is
meant to support are between configurations that differ in one of those factors at a time.

## 3. What this must support later

No hypothesis needed. What must you be *able* to compute once these exist.

This is what field 2 gets checked against. A comparison that varies two things at once under
your stated structure will not answer what you want, and here is where that costs ten minutes
instead of a re-extraction.

Answer: I must be able to compute the accuracy, precision, recall, etc. of the two models under the different applications of the steering vector and with no application of the steering vector on the amber subset that I'm talking about.

Those same quantities must also be computable **broken down by question type and by ground-truth
label**, since the expanded set is drawn against a per-question-type yes-ratio target.

And **per item**: for every configuration, whether each item was answered correctly, retained
under a stable item id, over the identical 1500 items in every configuration — so that the number
of discordant pairs between any two configurations can be counted.

And **against the 2026-07-30 run**, on the 450 items the two runs share: the same per-item
comparison between this run's baseline and meandiff-nd500 cells and that run's, so the two runs
can be checked against each other item by item.

---

## Code gaps

**Filled by an agent, not by you.** Leave it empty; an agent that reads the repo fills it in.

Places where the repo cannot produce what fields 1–3 ask for as written: an argument that
exists at one layer and is not forwarded by the layer above it, a cache key that omits a
parameter that changes its contents, an entry point that does not accept a structure named in
field 1. Found by reading the code, with file and line.

"Not your job" below already assigns *resolving* these to the plan. This section is where they
are made visible in the spec rather than surfacing only inside the plan, so that a gap is a
recorded fact rather than something you have to notice or patch yourself.

The boundary is the same one that governs the whole spec: a gap belongs here only if closing it
changes nothing in fields 1, 2, or 3. Anything that would change them comes back to you as a
question.

- (empty — for an agent to fill)

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
