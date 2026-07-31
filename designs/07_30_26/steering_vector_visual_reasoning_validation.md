# Design spec — steering_vector_visual_reasoning_validation_07_30_26_design

Written by Alex. Sections marked **[filled by agent]** were expanded from Alex's prose
elsewhere in this file; every value in them traces to a sentence he wrote or to a path in
the repo, and each such section names its source. Nothing in them is a new condition.

---

## The question

This experiment is in reference to only the textual steering method from the VTI paper, not the visual steering method. It is clear when reading the VTI manuscript versus reading the code base provided by the authors that there are both plenty of discrepancies and unwritten details between the method that is suggested in the paper and the method that was implemented in the code base. One part of the paper that is left largely undescribed and leaves plenty of questions upon first viewing of the codebase is the use of PCA to calculate the steering vector used in downstream experimentation. The codebase describes using some form of centered singular value decomposition on the differences in activations between truthful versus hallucinated captions of images. Specifically, the data that is used to perform the singular value decomposition is a flattened concatenation of the activations described of all layers of the model. Once this PCA is performed, the actual steering used during inference is the sum of the first principal component plus the mean of the data described. Finally, the Steering vector is applied using a blunt rotation-based steering approach in the forward function of the VTI layer of the code. 

The number of unexplained design choices per the codebase has made it extremely difficult to confirm the findings of the paper and to build upon the work done in it. Given the number of ambiguities and discrepancies, I believe the easiest approach to try to confirm the findings of the paper is to simplify the method by trying to apply it in its simplest form: using the mean difference in truthful versus hallucinated captions as the steering vector and applying it to examples from Amber, Pope, and Chair on similar models to those used in the paper, specifically Llava 1.5 and Qwen 2.5. However, given the runtime of trying to do this, it would be easier and still likely representative of the method itself to perform this verification Using the established subsets of Amber, Pope, and Chair that exist in the codebase, i.e. the 400-600 example subsets of these datasets that have been used previously.

All this to say, the question is, does textual steering using a steering vector calculated solely from mean differences of paired truthful versus hallucinated captions of images I actually improved the visual reasoning capabilities of VLMs? 

## Competing explanations

At least two. If only one explanation could produce the result you expect, you are not
testing anything.

1. My hypothesis and explanation A is that textual steering vector inclines the model to provide truthful answers in response to visual questions, when performing VQA, per the benchmarks objectives that were mentioned in the question pretense. As a result, the accuracy on discriminative visual reasoning benchmarks should improve. Note that this explanation does not imply that the application of a textual string vector should improve the visual reasoning capabilities of the model, but as a proxy of improving the model's quote-unquote honesty in response to visual reasoning questions in VQA-style formats, the accuracy should also improve. I also expect that if this explanation is true, that the model does not make a marked improvement on generative benchmarks like Chair, since an improvement or a stronger inclination for the model to be honest or truthful would not improve its ability to caption images accurately. 
2. My second explanation is that textual steering is an actual causal mechanism to somehow improve the visual reasoning capabilities of VLMs, and hence they actually perform better on both image captioning and visual question and answering. 
3. The null hypothesis is that textual steering has no effect on either truthfulness or visual reasoning abilities of a model and hence does not improve the ability of the model to perform captioning or visual question and answering. 

## Prediction table

One row per condition, one column per explanation. Fill in what each explanation predicts for
the primary measurement — direction and rough magnitude, not just "changes".

This is the load-bearing part of the spec. If two columns are identical across every row, the
design cannot distinguish those explanations and should not be run. Find that out here rather
than after the GPU time.

Note that I have added a column specifying a baseline performance without the intervention of textual steering as described above, just to serve as a reference for the way I predict that the model should behave under each explanation provided above. The AMBER rows are described with accuracy, and the CHAIR rows use CHAIR score.

| Condition | baseline no intervention | Explanation A predicts | Explanation B predicts | Explanation C predicts |
|---|---|---|---|---|
| LLaVA performance on discriminative benchmark (ex. AMBER) | 70 | 77 | 77 | 70 |
| Qwen performance on discriminative benchmark (ex. AMBER) | 74 | 78 | 78 | 74 |
| LLaVA performance on generative benchmark (ex. CHAIR) | 20 | 20 | 16 | 20 |
| Qwen performance on generative benchmark (ex. CHAIR) | 17 | 17 | 15 | 17 |

Note that I assume that the performance on Llava will improve greater than Qwen because Llava should have worse visual reasoning capabilities as an older model than Qwen and a less effective visual encoder. As a result, it should be easier to improve the capabilities of model with such an inference time intervention, i.e., it should show a greater impact on the performance under each hypotheses in which the steering vectors should improve performance compared to qwen, and the baseline performance of qwen should be better to begin with than lava. 

Which single cell does the most work in separating A, B, and C, and why:

Rows 3 and 4, the performance on Chair, should do most of the job in separating all the hypotheses, since Chair is a generative benchmark, and hence performance on the benchmark should differ in comparison to a discriminative benchmark like Amber. In particular, I expect row 3, the performance on Llava, to be the most telling, since I expect that any improvement in visual reasoning should be more obvious in the differences in performance in LLava on a generative benchmark. In fact, I expect the relative performance change from baseline to explanation A in row 1 compared against the same change in row 3 to reveal whether it's possible that the steering vector is not actually improving visual reasoning performance, but rather maybe changing the behavior of the model in a different way such as its desire to be truthful or sycophantic that may result in better performance on a discriminative benchmark but not on a generative benchmark.  

## Cells

Fill this in for me based on my answer below for the item sets. Beta values of 0.2, 0.5, and 0.9 are a good grid. The layer sets used should be no layers for baseline, all layers for comparison to the paper implementation, and out of curiosity, I'd like to try intervention on layers 5 through 14, i.e. the early middle layers, as well as, say, 20 to 30 or to 27 or the equivalent for qwen or lava, i.e. the mid to late layers. Make sure that the layers applied for any windows never include the very last layer of the model unless it's being applied to all layers as per the original VTI paper. 

Three or fewer. Anything beyond that is a second design.

**[filled by agent]** — expanded from the paragraph above, the Item sets section, and Alex's
message of 2026-07-30 fixing the late-layer windows. This design is a full factorial, not
three cells; the count is stated below so the conflict with the template's limit is visible.

Factors, crossed fully unless noted:

| Factor | Levels | Source |
|---|---|---|
| Model | `llava-hf/llava-1.5-7b-hf` (32 decoder layers, idx 0–31), `Qwen/Qwen2.5-VL-7B-Instruct` (28 decoder layers, idx 0–27) | The question; layer counts read from local HF configs |
| Benchmark subset | AMBER discriminative 450 (`data/amber/pinned_amber_disc_450.json`), POPE 600 (`data/pope/pinned_eval_ids.json`, 200 each random/popular/adversarial), CHAIR 500 (`data/chair/pinned_chair_500.json`) | Item sets |
| Direction sample size | nd50, nd100, nd200, nd500 | Item sets |
| Direction construction | raw mean difference only (no PCA) | Alex 2026-07-30, and the question |
| Intervention | `additive` only | Alex 2026-07-30: the paper's method is additive linear steering; rotation is the authors' code, not this design |
| Hook site | `mlp` only | Alex 2026-07-30 (see below) |
| Layer set | all layers; 5–14; late window (LLaVA 20–29, Qwen 15–24) | The paragraph above, as corrected 2026-07-30 |
| Beta | 0.2, 0.5, 0.9 | The paragraph above |

Hook site is fixed at `mlp` and is not a factor. `evaluation/interventions/vti/steer.py:11`
exposes `HOOK_SITES = ("mlp", "layer")` and `intervention.py:39` defaults to `mlp`. Under
`variant="additive"` the two sites are the same operation: `steer` returns `x + alpha * d̂`
(`steer.py:44-45`), the decoder layer's only consumer of the MLP output is the residual add, so
adding `alpha * d̂` to the MLP output and adding it to the layer output produce the same hidden
state entering the next layer. The two sites are not equivalent for the rotation variants,
which renormalise against `x.norm()` (`steer.py:47,72`) — but no rotation arm exists here.

Steered arms, one row per (model × layer set). Each row is crossed with 4 direction sample
sizes × 3 betas = 12 arms, and each arm is evaluated on 5 benchmark invocations
(AMBER 450 ×1, POPE ×3 splits, CHAIR 500 ×1).

| Cell | Model | Benchmark / subset | Intervention | Layers | Beta |
|---|---|---|---|---|---|
| S1 | llava-1.5-7b-hf | AMBER 450 / POPE 3×200 / CHAIR 500 | additive @ mlp | all (0–31) | 0.2, 0.5, 0.9 |
| S2 | llava-1.5-7b-hf | AMBER 450 / POPE 3×200 / CHAIR 500 | additive @ mlp | 5–14 | 0.2, 0.5, 0.9 |
| S3 | llava-1.5-7b-hf | AMBER 450 / POPE 3×200 / CHAIR 500 | additive @ mlp | 20–29 | 0.2, 0.5, 0.9 |
| S4 | qwen2.5-vl-7b-instruct | AMBER 450 / POPE 3×200 / CHAIR 500 | additive @ mlp | all (0–27) | 0.2, 0.5, 0.9 |
| S5 | qwen2.5-vl-7b-instruct | AMBER 450 / POPE 3×200 / CHAIR 500 | additive @ mlp | 5–14 | 0.2, 0.5, 0.9 |
| S6 | qwen2.5-vl-7b-instruct | AMBER 450 / POPE 3×200 / CHAIR 500 | additive @ mlp | 15–24 | 0.2, 0.5, 0.9 |
| B1 | llava-1.5-7b-hf | AMBER 450 / POPE 3×200 / CHAIR 500 | none | — | — |
| B2 | qwen2.5-vl-7b-instruct | AMBER 450 / POPE 3×200 / CHAIR 500 | none | — | — |

Run count, with the POPE splits counted separately. `run_eval.py:33` takes one `--pope_split`
per invocation and `eval_runner.py:284` keys the output `pope_{split}`, so POPE 600 is three
invocations, not one.

- Steered arms: 2 models × 3 layer sets × 4 direction sample sizes × 3 betas = **72 arms**
- Benchmark invocations per arm: 5 (AMBER 1 + POPE 3 + CHAIR 1)
- Steered invocations: 72 × 5 = **360**
- Baseline invocations: 2 models × 5 = **10**
- **Total: 370 benchmark invocations**, covering 450 + 600 + 500 = 1550 items each.

Layer indices are absolute decoder-layer indices, `0..num_layers-1`, matching the
`layer_indices` argument of `vti_hook_ctx` in `evaluation/interventions/vti/hooks.py`.
Both windows exclude the last layer as required; the all-layers arm includes it, as permitted.

Direction category slice: `all`. Agent's choice — the extracted directions are stored per
category (`all`, `attribute`, `counting`, `existence`, `relation`) and the spec does not name
one; `all` is the general-purpose slice matching the paper's single steering vector. Override
this line if a per-category slice was intended.

## Item sets

Every item set, including any subset, filter, control group, or behaviour-conditional split.
For each: where it comes from, how many items, and one sentence on why it exists. If it already
exists on disk, give the path and call it a filter of that file rather than new construction.

- The number of examples from the subsets of Amber, Chair, and Pope are already determined in this repo, and this experiment should use those same exact subsets. They should be between 400 to 600. As for the size of the examples used to create the steering vectors, again, the samples from the 50, 100, 200, and 500 example sets from the demos 850 set should be used to calculate the steering vectors. and should each be used on each benchmark for each model, along with of course a baseline run without any intervention on each model as well. 

## Primitives this design requires

Anything the primary measurement needs that is not already on disk — directions, activation
caches, per-layer refits. Leave as "none" when everything exists. Listing a primitive here means
the plan produces it under this spec; no separate extraction spec is written.

The planner treats this section as a whitelist, so a primitive omitted here will not be produced.
Two limits: it must be a strict prerequisite of a measurement named below, and its scope must not
exceed what that measurement consumes. A primitive still worth producing after deleting the
measurement is its own extraction.

For each: what it is, where it will be written, and what existing artifact it must not shadow.

**[filled by agent]** — rewritten 2026-07-30 on Alex's instruction. The earlier version of this
section declared **none** and pointed the design at the existing `_r2_partition` direction sets.
That was wrong: those sets are PCA reconstructions, not mean differences
(`metadata.json` records `"steer_reconstruction": "live_pc1_plus_mean"` and a sign convention
reading "no mean-diff sign-align"), and this design does not test PCA construction.

**One primitive: raw mean-difference textual directions, no PCA.**

*What it is.* For each (model, nd ∈ {50, 100, 200, 500}), the per-layer mean over that
partition block of `value_activation − h_value_activation` at the `all` category slice —
i.e. `mean_i (clean_i − hallucinated_i)`, one vector per layer, no centering, no SVD, no
component selection. Diff polarity matches the existing path
(`DIFF_POLARITY = "value_minus_h_value"`, `directions_v2.py:41`, clean − hallucinated), so the
sign convention is inherited and no sign-alignment step is needed. Array layout is the same
`(num_layers + 1, hidden_dim)` decoder-aligned form the existing directions use, row 0 being
the embedding row, with the hook consumer slicing `[1:]` (`directions_v2.py:287-288`).

*Inputs, all already on disk — this primitive requires zero forward passes.* The per-demo
last-token activation stacks are cached at
`experiment_artifacts/vti/<model_short>/textual_v2/_act_cache/`, 5100 files per model =
850 demos × 6 variants (`value` plus the five dimensions in `DIMENSIONS`,
`directions_v2.py:33`), verified present for both `llava-1.5-7b-hf` and
`qwen2.5-vl-7b-instruct`. Demo membership per nd block comes from
`data/vti/demos_850_partition_s42.json` over `data/vti/demos_850.jsonl` (content hash
`ba05bd96`), `selection_policy: disjoint_partition`, seed 42 — the same blocks the PCA sets
used. The four blocks are **disjoint**, not nested: the nd50/100/200/500 arms differ in demo
identity as well as in n.

*Where it will be written.* New sibling directories under the same model namespaces, agent's
choice of slug, overridable:

- `experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/`
- `experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_meandiff_partition/`

each holding `directions.npz` and `metadata.json`. No `components.npz` — there are no
components.

*What it must not shadow.* The `_r2_partition` slug differs from `_meandiff_partition`, so the
existing PCA direction sets, the activation cache, and the partition file are all untouched and
still reachable. Nothing is regenerated in place. Evaluation outputs are new files under
`evaluation/results/` and must not overwrite any existing baseline dump.

*One fact about how this direction is consumed.* `steer` L2-normalises the direction per layer
before applying it (`steer.py:43`), so the per-layer magnitude profile of the raw mean
difference is discarded at application time; only its per-layer orientation reaches the model,
and beta is the sole magnitude knob. This is identical to how the PCA directions are consumed.

## Code gaps

**[filled by agent]** — read off the code, not reasoned about. Each is an implementation step
for the planner, not a question for Alex; none of them changes a cell, metric, item set, or
condition. Revised 2026-07-30 after the direction construction changed from PCA to raw mean
difference; gaps 1a/1b replace the previous gap 1.

1a. **No mean-difference extractor exists.** Nothing in the repo computes a textual
    mean-difference steering vector. `obtain_textual_vti_v2_from_stacks`
    (`directions_v2.py:277-323`) forms the per-demo diffs and then immediately fits PCA
    (`_live_pca_fit`, line 302), returning `PC1 + mean`; the only other use of a mean
    difference in the tree is sign-alignment of a PC in the visual path
    (`visual_directions.py:51-73`). The inputs are all cached — 5100 activation files per
    model under `textual_v2/_act_cache/` — so this is a reduction over existing arrays and
    zero forward passes. **Blocks S1–S6.**

1b. **A mean-difference direction set is not reachable from the intervention.**
    `VTITextualIntervention` resolves directions through `compute_or_load_textual_directions_v2`
    (`intervention.py:12-20`), which is PCA-only and whose selection policy is `shuffled_prefix`
    (`directions_v2.py:36`); neither `intervention.py` nor `run_eval.py` imports
    `directions_partition`, and `run_eval.py:21-77` exposes no flag naming a direction cache
    directory. `__init__` already accepts a preloaded `_directions` array
    (`intervention.py:48`), so this may be a pass-through rather than a rewrite.
    **Blocks S1–S6.**

2. **No layer-window argument end to end.** `vti_hook_ctx` accepts `layer_indices`, absolute
   `0..num_layers-1` (`hooks.py:62-98`), but `VTITextualIntervention.__init__` has no such
   parameter and does not forward one at `intervention.py:151-160`; `run_eval.py` exposes no
   corresponding flag. **Blocks S2, S3, S5, S6** — only the all-layers arms run today.
3. **No reusable h+/h- function.** `evaluation/classifiers/metrics.py` defines no flip
   accounting. Flip counting exists only inline at `rotation_strength.py:132-161`
   (`_flip_counts`), computed against a baseline generated in the same process. Note it returns
   **four** counts — `flip_tp_to_fn`, `flip_tn_to_fp`, `flip_fn_to_tp`, `flip_fp_to_tn` — where
   Primary measurement names only the latter two. Reporting h+/h- from `run_eval.py` output
   requires a per-item join against the retained B1/B2 dumps; the join does not exist, though
   the per-item records it needs are written (`eval_runner.py:154`).
   **Blocks the h+/h- half of Primary measurement.**
4. **Two defaults contradict this spec and must be overridden on every invocation.**
   (a) `run_eval.py:50` defaults `--chair_max_new_tokens` to 64; this design requires 256 (see
   Assumptions). Same trap in `rotation_strength.py:69` and
   `rotation_strength_chair_amber.py:81`. (b) `VTITextualIntervention` defaults
   `variant="uniform_rotation"` (`intervention.py:39`); this design is additive only, which is
   selected by naming `vti_textual_additive_mlp` in `--interventions`
   (`evaluation/interventions/__init__.py:29-33`) rather than by relying on the default.
5. **POPE scoring does not emit the gold-label split now required.**
   `score_pope_records` (`metrics.py:38-113`) tracks tp/fp/fn/tn internally but returns
   `accuracy_overall`, P/R/F1, `yes_ratio`, `n_correct`, `n_total`, `n_unparsed`, and
   `by_category` — no `tn`, and no per-gold-label accuracy. AMBER already has what Primary
   measurement asks for: `_amber_disc_finalize` (`metrics.py:124-142`) returns `yes_ratio`,
   `neg_item_accuracy`, `pos_item_accuracy`, `n_neg_total`, `n_pos_total`. **Blocks the
   gold-split half of Primary measurement on POPE only.** Closable offline from the retained
   per-item responses; no rerun needed.

**Not a gap — recorded at Alex's request, 2026-07-30.** `HOOK_SITES = ("mlp", "layer")`
(`steer.py:11`) names the second site after the module it hooks rather than after what it is;
`layer` is the decoder-layer output, i.e. the residual stream. Under `variant="additive"` the
two sites are the same operation (see the note in Cells), so the choice is a naming trap rather
than a factor. Alex's view is that `layer` should be renamed to something like
`residual_stream`, and that additive textual steering should be fixed at `mlp`. This blocks no
cell in this design and changing it is not required to run it, so it is logged here rather than
in the numbered list.

## Primary measurement

A primitive: probability of a token, parsed answer, layer index, count. Not a ratio, rate, or
normalised difference.

In this case the measurements used will be CHAIR score for the CHAIR benchmark — a
hallucination rate, where lower is better, not an accuracy — as well as accuracy, precision,
recall, F1, and the total number of hallucinations induced and reduced from the set of those
run (h- and h+) for the discriminative benchmarks.

If a composite is genuinely wanted, write its lineage in terms of measured quantities and say
why it belongs on the y-axis instead of one of its inputs:

**[filled by agent — lineage only]** — read off the repo. The justification sentence is left
for Alex; nothing below argues that these belong on the y-axis.

Every quantity named above except h+ and h- is a rate over a measured count. Lineage:

- `chair_i` = (objects mentioned that are not in the image, summed over captions) / (objects
  mentioned, summed over captions). `chair_s` = (captions containing at least one hallucinated
  object) / (non-empty captions). Both defined in
  `evaluation/classifiers/metrics.py:338-343`, both lower-is-better, both computed over
  non-empty captions only. Object grounding uses the Rohrbach et al. synonym list vendored at
  `evaluation/classifiers/chair_synonyms.txt`.
- Accuracy, precision, recall, F1 on AMBER and POPE are all functions of the four parsed
  yes/no outcome counts (tp, fp, tn, fn), whose underlying primitive is the parsed answer per
  item.
- h+ and h- are already primitives — counts, not rates. h+ = items flipping tn→fp against the
  matched baseline run; h- = items flipping fp→tn. Definitions at
  `evaluation/vti_rotation_strength/rotation_strength.py:245` and
  `evaluation/chair_amber_diagnostics/rotation_strength_chair_amber.py:368`. Both require a
  per-item join against the baseline run for the same model and benchmark subset, so baseline
  cells B1 and B2 must be run and retained per item, not only in summary.
- Per `CLAUDE.md`, `p_yes_norm` and `answer_mass` are banned from reporting and appear nowhere
  in this design.

**Gold-label split reporting on the discriminative benchmarks — added 2026-07-30 by Alex.**
AMBER 450 is not balanced: `data/amber/pinned_amber_disc_450.json` `_meta.gold_counts` is
`{"no": 276, "yes": 174}`. Accuracy on an unbalanced yes/no set moves both when the model
discriminates better and when its answer rate shifts toward the majority label, and the
aggregate number does not say which happened. Therefore, for every discriminative cell
(AMBER and each POPE split), report alongside the aggregate:

- the **yes rate** — parsed-yes answers / parsed answers — for the run, and the same for its
  matched baseline;
- **accuracy on the gold-no subset and accuracy on the gold-yes subset, separately**, each with
  its item count;
- the four counts tp, fp, tn, fn from which all of the above are formed.

Alex's stated assumption, to be checked against these numbers rather than assumed: the model
should perform comparably on the gold-no and gold-yes subsets. Retaining these per item is what
makes the aggregate attributable by reanalysis rather than by rerunning the benchmark
(`templates/design_template.md:127-129`).

## Sample size

n per cell: 450 (AMBER discriminative), 600 (POPE), 500 (CHAIR). Identical in every steered
and baseline cell — the subsets are fixed, so n does not vary across models, sample sizes,
betas, or layer sets. Direction sample sizes (50 / 100 / 200 / 500 demos) are a factor, not
an n per cell.

Why that n — what size of effect it can separate from zero, and how you know:

Just use these sample sizes because I've already constructed well selected examples into subsets of each benchmark. And it's not necessarily important to select different ends for this use case. I know that these sample sizes are sufficient for detecting effect of the steering on the accuracy and other performance metrics being measured, which is good enough.  

**[filled by agent — statement of fact, not a justification]** n is inherited from the pinned
subsets already on disk; it was not chosen against a target effect size, and this spec gives
no power argument. Alex's note under Assumptions proposes bounding baseline performance with
confidence intervals to judge whether a steered result is distinguishable from chance.

**Alex's decision, 2026-07-30:** he declines to state a detectable effect size in this spec.
The prediction table is intended to carry direction and relative magnitude only, not deltas
sized against n. Recorded here so the file shows a decision rather than an omission. The
confidence-interval bound on the baseline named above is the check that remains, and it is
performed at analysis time, not stated here.

## What I predict

Write this before the run. It is not scored and no one else reads it; its only job is to exist
before the numbers do, so that your reading afterwards cannot quietly become what the data
said.

I believe I may have given my hypothesis before, but formally put, based on inclinations from previous results I've had until this point in this repository, I think that additive steering might actually be inducing truthfulness rather than enhanced visual reasoning capabilities in the model when applied additively as per the paper. The data that both the original authors use and I use for steering vector calculation are truthful captions subtracted from hallucinated captions. And it's been shown in previous papers and literature that Much more attention computation is devoted to attention between textual tokens than visual tokens in VLM. To me, this makes me think that a steering vector on the LLM backbone is operating less on the interaction between the textual and visual tokens, tokens, but rather the reasoning of how it should answer the question regardless of the image provided. And hence, if a potential effect of a steering vector meant to represent truthfulness is for the model to be more truthful, which is the standard kind of effect that one would expect from a steering vector, Then it may just be the case that for answers that the model would normally get wrong, where it actually knows the correct answer, the model suddenly becomes inclined to give the correct answer when the steering vector is applied, which ends up bearing itself as an increase in accuracy in evaluation, when in reality, that increase in accuracy doesn't have anything to do with improving the model's visual reasoning capabilities. 



## What would make me abandon the hypothesis

If baseline performance of the model showed no difference on these benchmarks than when steering is applied, then I would abandon the hypothesis that truthfulness is actually the behavior that's induced by the steering vector. If the model showed improvements on Both the discriminative benchmark and the generative benchmarks being tested, then I would expect that the visual reasoning capabilities of the model may have actually improved, And that further downstream experimentation would have to be done to assess whether the effect of the steering vector is visual reasoning improvement or something else that could have caused improvement on a generative benchmark like Chair. . 

**Clarification recorded 2026-07-30, in Alex's words.** "These benchmarks" in the first
sentence means the **discriminative** benchmarks only — AMBER and POPE. POPE and AMBER
discriminative test the same ability, so a statement about AMBER covers both. The criterion
therefore reads: if steering moves neither AMBER nor POPE off their baselines, the truthfulness
hypothesis is abandoned. CHAIR being unchanged is a prediction of Explanation A (rows 3 and 4),
not a trigger for abandonment.

## Assumptions that need checking first

Anything this design rests on regarding the data, the tokenizer, the extraction, the benchmark
subset, the parser, or model behaviour on the intended items. For each, what result would
invalidate the design.

- I can't think of any assumptions that would need to be checked. It can be reasonably assumed without testing that the model has some ability to perform visual reasoning and hence can have a baseline accuracy above zero, well above zero, to test against, and that the steering vector will have some meaningful effect on the performance of the model outside of what one would expect from random chance. I guess creating confidence intervals to bound the performance of the model at baseline would be a good way of determining whether the downstream results when the steering vector is applied are within random chance or not. 

**[filled by agent]** — written from Alex's instruction of 2026-07-30 on generation length,
and grounded against the repo.

- **Generation length must be long enough for a complete answer on the benchmark being run.**
  On a generative benchmark like CHAIR the cap is 256 new tokens. This is not a new choice:
  `CHAIR_CAP = 256` is already frozen in
  `evaluation/chair_amber_diagnostics/make_diagnostic_summary.py:37`, set by the Step 0 sweep
  whose output is `evaluation/results/2026-06-22/_diagnostics/step0_chair_token_cap.json`.
  Note that several existing scripts still default to `--max_new_tokens 64`
  (`rotation_strength.py:69`, `rotation_strength_chair_amber.py:81`), so the cap must be
  passed explicitly rather than left at the default. **What invalidates:** captions truncated
  before the model finishes suppress the object count per caption, which moves `chair_i` and
  `chair_s` through the denominator rather than through hallucination behaviour; a steered and
  a baseline arm generating at different effective lengths are not comparable. If the sweep
  shows CHAIR score still moving with the cap at 256, the cap is not safe and the design needs
  a length-matched comparison instead.

**[added 2026-07-30 by Alex]**

- **AMBER 450 is not balanced across gold labels, so the answer rate has to be reported
  alongside accuracy.** `data/amber/pinned_amber_disc_450.json` `_meta.gold_counts` is
  `{"no": 276, "yes": 174}` — 61.3% gold-no. The design assumes the model performs comparably
  on the gold-no and the gold-yes subset; that assumption is not established and is why the
  yes rate and the two per-gold-label accuracies are now required outputs (see Primary
  measurement). **What invalidates:** if baseline accuracy differs substantially between the
  gold-no and gold-yes subsets, or if the steered yes rate moves toward the majority label,
  then the aggregate accuracy row of the prediction table is not reporting discrimination and
  the row cannot be read as the table describes it.
