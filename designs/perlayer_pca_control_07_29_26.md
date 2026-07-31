# Design spec — perlayer_pca_control

## The question
We are trying to estimate the direction that represents the concept of 'truthfulness' or 'accuracy' in describing images by creating steering vectors. Until now, we've been use a global PCA approach to try to extract this direction, then slicing it across each layer to use the steering vector during intervention. This approach causes some confounds. For example, the global fit PC1 is likely dominated by whichever layers have the largest scatter, which is likely the layers that have the highest activation norms (ex. in Qwen, the later layers). However, these layers, simply because they have higher norms, may not contribute more to the model's overall representation of 'truthfulness' or 'accuracy'. Recent literature from VLM jailbreak detection and hallucination mitigation even suggest that middle layers, as well as late layers, have the greatest causal effect on the model's ability to perform these tasks. Because these tasks require the model to understand such concepts of truthfulness, it is a fair assumption that these layers would also be the ones in which the concept of truthfulness is best represented. 

All this to say, my question is: do steering vectors calculated using a per layer PCA approach display a lower cosine similarity to control vectors (calculated in the same way) at mid to late layers for both models, particularly llava, than a global PCA estimation approach, given that the norm of these vectors is constant across layers in llava? Note that while this question is really specific to Lava, I want to perform the same test on Qwen for comparison purposes, Even though qwen did not display a similar norm progression across layers between the control and the prediction vectors and the cosine similarities of these vectors at later layers in qwen were decreasing even with the global PCA approach.


## Competing explanations
1. The contribution of the representation quality of truthfulness at different layers across both models is not as strong as the effect of the increased norm progression across layers, suggesting that the high cosine similarity across all layers of lava using the global PCA approach may not be due to the estimation technique at all. 
2. The contribution of the representation quality of truthfulness at different layers varies significantly and hence the cosine similarity between a global PCA estimated steering vector versus a per layer estimated steering vector. should decrease as I expect that the representations of truthfulness will improve in later layers and hence the control vector should bear less and less similarity to the vector that tries to capture this concept across layers. 

## Prediction table
| Condition | A predicts | B predicts |
|---|---|---|
| Qwen, layers 0-10 | cos = 0.99 | cos = 0.95 |
| Qwen, layers 18-27 | cos = 0.6 | cos = 0.2 |
| LLaVA, layers 0-10 | cos = 0.99 | cos = 0.95-0.99 |
| LLaVA, layers 18-31 | cos = 0.96 | cos from [0.3, 0.5] ish |

Which single cell does the most work in separating A from B, and why: Cell 2 should do the most work in separating A from B, Since lava is the model that actually shows little to no difference between the control and prediction vectors across layers, even at later layers.

## Cells
| Cell | Model | Item set | Intervention | Layers | Beta |
|---|---|---|---|---|---|
| 1 | qwen2.5-vl-7b-instruct | demos850 all, nd 50/100/200/500 | none (geometric only) | all decoder | n/a |
| 2 | llava-1.5-7b-hf | demos850 all, nd 50/100/200/500 | none (geometric only) | all decoder | n/a |

## Item sets
- deployed — experiment_artifacts/vti/{model}/textual_v2/demos850_ba05bd96_all_nd{50,100,200,500}_s42_r2_partition — disjoint blocks of 850 — the directions under test
- shuffled control — experiment_artifacts/vti/{model}/shuffled_control_demos850/all_nd{50,100,200,500} — same demos, images deranged within block — image-correspondence control

## Primitives this design requires

The comparison needs per-layer-PCA directions for both arms, at all four sample sizes and both
models. These do not exist; only the global-fit versions are on disk. Everything they are built
from does exist, so this costs zero forward passes.

- Per-layer-PCA deployed directions — fit one centered PCA per decoder layer on that layer's
  `value − h_value` diffs, reconstruct as PC1 + mean exactly as the global scheme does, from the
  cached stacks in `experiment_artifacts/vti/{model}/textual_v2/_act_cache` (read-only; 5100
  files present per model). Write to
  `experiment_artifacts/vti/{model}/textual_v2_perlayer/demos850_ba05bd96_all_nd{N}_s42_r2_partition_perlayer/`.
- Per-layer-PCA control directions — same fit, from
  `experiment_artifacts/vti/{model}/shuffled_control_demos850/_act_cache` (read-only; 1700 files
  present per model, 850 pairs x 2 variants). Write to
  `experiment_artifacts/vti/{model}/shuffled_control_demos850_perlayer/all_nd{N}_perlayer/`.

Both use the same demo id blocks and the same derangement files as the existing global-fit
directions, so the two schemes are compared on identical data.

**What these must not shadow.** `textual_v2_slug()` in `evaluation/interventions/vti/directions_v2.py`
encodes demo file, content hash, dimension, num_demos, seed, rank, and selection policy — and
nothing that distinguishes a global fit from a per-layer fit. Two extractions differing only in
fit locus therefore resolve to the same directory, and `save_textual_v2_directions` writes
`directions.npz` and `components.npz` inside it, so the second run silently replaces the first.
The existing global-fit directions are the baseline this design reads against; overwriting them
destroys the comparison and violates the standing additive rule.

Two independent guards, both required rather than either alone:

1. A fit-locus field in the slug, defaulting to the current global behaviour so every existing
   slug string still resolves to its existing directory unchanged.
2. A separate parent directory per scheme, as in the paths above, so a code path that reuses the
   old slug builder still cannot land on an existing artifact.

Verification before any cosine is computed: the global-fit `directions.npz` files listed under
Item sets are byte-identical to their pre-run state, and every per-layer output is a new path.

## Primary measurement
Per-layer cosine between the deployed and control direction slices, one value per decoder layer.
Secondary: per-layer L2 norm of each direction, both arms.

Stated explicitly for the planner, because the question sentence admits more than one reading.
The cosine is always **deployed against control**, both fit under the same scheme, one value per
decoder layer per sample size per model. It is never per-layer-fit against global-fit. The run
produces one such curve per cell using the new per-layer directions, and one from the existing
global-fit directions for the same cell.

The global-fit *directions* for demos850 are on disk for both arms, but no cosine curve has been
computed from them — the only such curve in the repo,
`diagnostic_experiments/perception_diag/control/geometric_comparison/geometric_comparison_summary.md`,
is demos_v2 at nd200 and is a different item set. So the baseline curve is computed here too,
from the existing global-fit directions, read-only. That is numpy over cached arrays, not a
re-extraction: nothing under the global-fit paths is written.

So each cell yields two curves over layers — new per-layer scheme, existing global scheme — at
each of the four sample sizes, and the object of interest is how they differ across depth.

Rows in the prediction table are model-by-layer-band groupings, not rows of the Cells table.

## Sample size
n per cell: 50 / 100 / 200 / 500 demo pairs (four points, disjoint blocks)
Why that n: These are simply the demo pair sample sizes we have currently. Future experiments may test the noise floor of vectors of the same sample sizes estimated using different data samples. 

## What I predict
I think the prediction table largely shows my belief as to what I predict will happen. The cosine similarity at later layers of lava using per layer PCA calculated steering vector directions should decrease, though not as much as qwen because I expect qwen to have a better representation of the direction or concept of truthfulness as it is a newer model is a very well trained with generally better pre-training techniques and is of higher quality. 

## What would make me abandon the hypothesis
I would say that if lava showed basically no decrease in cosine similarity as layers progressed using a per layer PCA approach, showed a similar decline to that of the previous experiment using the global PCA-based steering directions at each layer, I would abandon this hypothesis or I would at least perform some follow-up tests to test it more reversely. for example calculating the noise floor to see whether the cosine similarities that I've generated are really meaningful compared to error in any way. 

## Assumptions that need checking first
- Per-layer PCA reconstruction keeps PC1 + mean, matching the global scheme's form — otherwise the arms differ in more than the fit locus
- LLaVA's PC1 share rises from ~9% to 20-72% under the refit; Qwen's from ~0.8% to 1.5-11% — What would invalidate this is if PC1 shares decreased over time. specifically if the norms of the steering vectors did not increase from layer to layer.
