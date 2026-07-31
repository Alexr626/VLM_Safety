# Discussion record — per-layer PCA, the image-shuffled control, and what the existing geometric comparison measures

Date: 2026-07-29
Participants: Alex (all interpretation, hypothesis, design decisions), Claude (facts, geometry, questions)
Subject of discussion: the already-existing artifact
`diagnostic_experiments/perception_diag/control/geometric_comparison/geometric_comparison_summary.md`
(dated 2026-07-22, demos_v2 `all` @ nd200), plus the new demos850 partition and shuffled-control
direction sets.

This file records a conversation. It is not a reading of a run, and it is not a design spec.
Interpretive statements below are attributed to Alex and are recorded as his, not endorsed.

---

## 1. What the extraction code actually does

Verified by reading the code, not inferred.

- `evaluation/interventions/vti/pca.py:33-34` — the PCA is **centered**. `mean_` is registered and
  subtracted before SVD. The deployed direction is the reconstruction `PC1 + mean`
  (`directions_v2.py:253`), which is not the same object as an uncentered PCA's PC1.
- `directions_v2.py:282-290` — the fit is a **single global flatten**. Each demo's `(L+1, d)` diff
  is flattened to one vector and PCA is fit on `(N, (L+1)·d)`. One PC1 and one mean span all
  layers; the result is reshaped back to per-layer slices.
- `steer.py:43` — each layer slice is **unit-normalized at application time**. Only the direction
  of each layer slice reaches the model; extracted magnitude does not.
- `pca.py:25-39` — the PCA class already performs **batched per-layer fits** if handed a 3-D
  `(L+1, N, d)` tensor. Per-layer PCA needs no new math, only a different input shape.
- Diff polarity is `value − h_value`; both members of a pair use the **same image** and differ only
  in caption text (`directions_v2.py:406-416`).

## 2. Norms, and why they govern everything downstream

Qwen2.5-VL-7B, demos850 `all` @ nd200, from `metadata.json`:

- `‖mean_l‖` spans **0.167 (layer 0) to 65.5 (layer 26)** — a ~400× range.
- Global-slice `‖PC1_l‖` spans 0.0004 to ~0.52. PC1/direction ratio is 0.002–0.008.

Consequence for a per-layer refit: per-layer PCA pins `‖PC1_l‖ = 1` at every layer while `mean_l`
is unchanged. The PC1:mean mixing ratio therefore becomes depth-dependent in the opposite
direction — PC1-dominated at early layers, mean-dominated at deep layers. Per-layer PCA does not
remove the norm from the problem; it relocates where the norm enters.

Approximate PC1:mean ratio under a per-layer refit (`1/‖mean_l‖`), from the 2026-07-22 table:

| | Qwen deployed | Qwen control | LLaVA deployed | LLaVA control |
|---|---:|---:|---:|---:|
| layer 0 | 6.0 | 5.9 | 47 | 47 |
| layer 15 | 0.41 | 0.52 | 1.10 | 1.24 |
| layer 18 | 0.106 | 0.336 | 0.72 | 0.77 |
| layer 26 | 0.015 | 0.085 | 0.31 | 0.32 |

Two asymmetries fall out of this table and both are design-relevant:

1. **Depth.** At Qwen layers 22–27 the reconstruction is the mean plus a 1.5–2.9% perturbation
   under either scheme, so the deployed-vs-control cosine there is `cos(mean_l^D, mean_l^C)` to
   about three decimals and PCA is not participating. The perturbation is larger at layers 18–19
   (7–11%), so "identical to three decimals" holds at the deepest layers only.
2. **Model.** LLaVA's `mean_l` norms are small at every depth, so a unit PC1 is 31–47% of the
   vector throughout. A per-layer refit changes LLaVA's directions substantially and Qwen's
   deep-layer directions barely at all.
3. **Arm.** For Qwen the control's mean norms are ~5.5× smaller than the deployed's at depth, so a
   per-layer refit perturbs the control arm more than the deployed arm. The two things being
   compared are not perturbed equally.

## 3. The finite-sample floor (concept, taught; not applied to project results)

For diffs `d_i = μ + ε_i` with per-coordinate noise variance `σ²` in dimension `D`:

```
cos(μ̂₁, μ̂₂) ≈ λ/(1+λ)      and      cos(μ̂, μ) ≈ √(λ/(1+λ)),      λ = n‖μ‖²/(Dσ²)
⟹  cos(μ̂, μ) ≈ √( cos(μ̂₁, μ̂₂) )         [split-half reliability]
```

Requires equal `n`, **disjoint** samples, identical method. Nested samples inflate the cosine
because the cross-term no longer vanishes. `1/cos − 1` is linear in `1/n` through the origin, so
several sample sizes test the functional form as well as estimating the slope.

Two limits Alex stated correctly and that should stay stated:

- This measures **reliability, not validity**. `μ` is the population mean difference under the demo
  distribution, not "the truthfulness direction." Split-half says whether an estimate has
  converged, not whether it converged to the right thing.
- The floor is **per-layer**. Layer 0 and layer 27 have different scatter and different effective
  dimension; there is no single number.

`data/vti/demos_850_partition_s42.json` is `disjoint_partition` with blocks 50+100+200+500 = 850 —
one block per size, so no same-`n` replicate pair currently exists. Sub-splitting a larger block
would produce replicates at smaller `n`.

## 4. Facts read from the 2026-07-22 geometric comparison

demos_v2 `all` @ nd200. **No equivalent comparison exists yet for the demos850 sets.**

- Qwen per-layer cosine: min 0.329, median 0.922, max 0.998. ~0.99 through layer 4, 0.87 at 15,
  0.59 at 18, 0.33–0.42 across 22–27.
- LLaVA per-layer cosine: min 0.821 (layer 15), median 0.948, max 0.999; 0.88–0.96 across 18–31.
- Magnitude: Qwen layer 26 deployed 65.5 vs control 11.8 (5.5×); LLaVA layer 26 deployed 3.24 vs
  control 3.11 (1.04×). Per-layer normalization means this never reaches the model.
- The deployed Qwen intervention is mlp layers 18–27 (CLAUDE.md), i.e. exactly the layers where the
  Qwen cosine is lowest. A median over all 28 layers includes 18 layers that are never steered.
- PC1 EVR, Qwen `all`, demos850, n = 50/100/200/500: deployed 0.68 / 0.71 / 0.65 / 0.66; control
  0.36 / 0.46 / 0.38 / 0.38. LLaVA figures not read (tool denial). EVR is computed on **centered**
  data, so the mean difference contributes zero to it — it describes the part of the data the
  steering direction almost entirely discards.

## 5. Positions Alex stated (recorded as his)

- The image-shuffled control exists to test whether the captured concept is image-conditioned at
  all. Scope note added in discussion: both arms hold the image fixed *within* each pair, so
  derangement varies caption-image **correspondence**, not image presence.
- Testing visual-reasoning behaviour under steering does not require the vector to be
  image-conditional; it requires the evaluation to demand visual reasoning.
- A high deployed-vs-control cosine does not support a truthfulness reading, because the same
  estimator on differently-constructed data cannot establish closeness to an unknown target.
  Open question left unresolved: if that objection blocks every reading, what work is the control
  doing?
- Reading generalized from LLaVA: because the cosines are high, the vectors may not capture an
  image-conditioned concept. Discrepancies raised against this: LLaVA's cosine is not flat
  (0.821–0.999); Qwen's low band coincides with its entire steering window; and LLaVA is the model
  CLAUDE.md records as showing effects at or inside noise.
- 0.329 cannot be called low without a noise floor.

## 6. Proposed next run (Alex's)

Per-layer PCA refit of both the deployed and shuffled-control directions at all four sample sizes
and both models, then the same comparison already run: per-layer cosine plus per-layer norm
progression, deployed vs control.

Stated hypothesis: cosine stays high at early layers and falls at middle-to-late layers in Qwen as
before, and the same pattern now appears in LLaVA. Stated rationale: mid-to-late layers carry the
largest causal effect for VQA and jailbreak-refusal decisions in the literature Alex has read, so
an image-conditioned concept should separate from its image-swapped control more at depth.

Prerequisites confirmed on disk: `_act_cache` coverage is complete — 1700 files (850 × 2 variants)
in each `shuffled_control_demos850/_act_cache`, 5100 in each `textual_v2/_act_cache`. **Zero
forward passes required.** Both models, both arms.

Open design questions carried into the spec, unanswered here:

- Under a per-layer refit, both "the concept separates with depth" and "the refit raised PC1's
  share of LLaVA's vector from ~9% to ~31%" predict a LLaVA cosine drop. What separates them?
- Qwen's deep-layer result is nearly fixed by the mean under either scheme. What can the Qwen arm
  of this run falsify that the 2026-07-22 run did not already settle?
- Against what floor is any resulting cosine read as high or low?
- Direction production is arguably an extraction and the cosine comparison is arguably an
  experiment. One spec or two?

## 7. Harness note

`Write(designs/**)` is denied at `.claude/settings.json:57`, so the design spec for this run is
Alex's to write. `templates/design_template.md` is the form. The `designs/` directory does not yet
exist.
