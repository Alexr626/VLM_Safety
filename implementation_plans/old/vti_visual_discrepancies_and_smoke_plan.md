# VTI visual arm — manuscript-vs-code discrepancy ledger + smoke-test plan

**Companion to:** `plan_visual_vti_reimplementation.md` (the build spec). This file is
(1) the authoritative list of discrepancies between the VTI manuscript and
`shengliu66/VTI` code for the **visual** steering arm, with the default we chose and the
inference behind it, and (2) the lambdab2 smoke-test plan that exercises the non-default
configurations and the grounding-vs-bias gate.

---

## Part 1 — Discrepancy ledger

Defaults in the re-implementation are **paper-faithful**; the code-faithful configuration
is reproducible by flags. "Inference" marks values the manuscript leaves unspecified,
where we inferred a default from its wording.

| # | Variable | Manuscript | Code | Our default | Status |
|---|---|---|---|---|---|
| 1 | Steering geometry | Additive: add direction to hidden state | Norm-preserving rotation (nlerp, `eps=0.1`), in `VTILayer` — identical math to the textual arm | `additive` is the paper cell; both geometries built (4-variant grid) | **Smoke-tested** (core question) |
| 2 | Hook site | Adds to hidden state `h_l,t` → layer (post-residual) output implied | Wraps `layer.mlp` → MLP sub-block output, pre-residual | `layer` is the paper cell; both sites built | **Smoke-tested** (core question) |
| 3 | `mask_ratio` | Main text: "randomly mask **a few** patches"; **appendix: "set to 0.99"** (explicit, all setups) | **0.99** | **0.99** — appendix authoritative; an *internal manuscript inconsistency* (main text vs appendix), NOT a paper-vs-code discrepancy; see N1 | Optional curiosity probe only |
| 4 | Mask fill | Unspecified ("mask") | **Mean-fill** (per-image spatial mean pixel) | **zero** — *inference*: plain masking = blanking | **Smoke-tested** |
| 5 | Perturbation type | Random patch masking; gaussian noise appears in the paper's *diagnostic* experiments only | Patch masking only | `patch_mask`; `gaussian_noise` exposed as alternate | Smoke-optional (stage 2) |
| 6 | Direction reconstruction | "first princip[al] direction" of stacked Δ's | `(pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1)` — top PC **plus the PCA mean added back**; mean-inclusive, not a pure PC | **`top_pc`** (sign-aligned to mean diff); `legacy_pc_plus_mean` exposed | **Smoke-tested**; deserves its own dedicated discussion (see N2) |
| 7 | `num_trials` | "average across 50 random masks" (appendix) | 50 | 50 | No discrepancy |
| 8 | `num_demos` | "a few examples N" (unspecified) | 70 (README run cmd) | 70 — *inference* from run command | Noted only |
| 9 | Alphas | Appendix: CHAIR α=0.4 (visual-only); combined VTI α=0.2; other experiments α=0.9 | `--alpha_image 0.9` in README cmd | Smoke grid `{0.2, 0.4, 0.9}` covers all three regimes | Grid |
| 10 | CLS token | Not mentioned | Included in extraction AND steered (despite the code's own `# no CLS token` comment, which is wrong about its code) | Include (`include_cls=True`); toggle exposed, revisit after smoke | Noted; revisit post-smoke |
| 11 | Embedding layer | `l` over transformer layers | `hidden_states` incl. embedding row; caller slices `vti_vision[1:]` | Drop embedding from extraction; never steer it | No real discrepancy; bookkeeping |
| 12 | Cosine gate (`lambda_sim`) | Not described | Present but commented out / hardcoded 1.0 | **Excluded entirely** (treated as an external design element, cf. Spherical Steering; not a variant here) | Out of scope |
| 13 | Decode branch | — | `x.size(1) < 2` branch in `VTILayer` (single-token decode path; shared class with the text decoder) | Not implemented — dead code for the ViT (always 577 tokens) | Out of scope |
| 14 | Sign conventions | Visual: `Δ = h̄(masked) − h(clean)` (Eq. 1). Textual: clean − hallucinated | Same in code for both arms | Preserve per-arm signs exactly; **not a discrepancy** — the two arms simply differ from each other, consistently in paper and code | Correctness flag only |
| 15 | Extraction site | Hidden states (layer outputs) | Same (`output_hidden_states=True` block outputs) | Same — one extraction at post-residual layer outputs, reused by all four variants | No discrepancy |
| 16 | PCA batch layout (visual) | "first princip[al] direction" per (l,t) implied | `reshape(n_tokens, -1)` on a **layer-major** tensor **scrambles** the PCA batch axis — each row is a 25-token strip of one layer (some straddling layer boundaries), not a token. The **mean term survives** the reshape (correct per-(layer,token) mean diff); the **PC term does not** (fit on garbled rows) | `top_pc` = clean per-(layer,token) PCA (default); `legacy_repo_reshape` ports the scramble verbatim for parity | **Smoke-tested** via the direction_recon toggle; strengthens N2 |

### N1 — mask_ratio: internal manuscript inconsistency, and why 0.99 is load-bearing
The main text says "randomly mask a few patches," but the appendix states explicitly:
"In all experimental setups, the mask ratio to compute visual direction is set to 0.99,
and we average across 50 random masks." The appendix (specific, quantitative) is
authoritative; the main-text phrase is loose language, and the code agrees with the
appendix. **Default = 0.99.** This row is an internal manuscript inconsistency, not a
paper-vs-code discrepancy.

Why 0.99 is load-bearing rather than an implementation excess: at r=0.99 each masked
copy is ~99% flat fill with ~6 surviving patches that differ per trial, so the 50-trial
average h̄ converges toward the encoding of a near-blank image — a content-free anchor
(fill color + positional embeddings). The diff Δ = h̄ − h then points from "this image's
encoding" toward "no content" — a large, clean, content-erasure direction. At low r
(e.g. 0.1) each copy is ~90% intact, h̄ sits a short step from the clean h, and Δ is a
small, image-specific perturbation-response vector with poor PCA SNR across demos —
nearly nothing to steer toward. The paper's "robust embedding" intuition only yields a
direction meaningfully different from clean under heavy masking. So low-r probes, if
ever run, are expected to look weak for mechanistic reasons, not because the method
"fails" there — the direction is a different object at low r, not a worse estimate of
the same one.

### N2 — the `legacy_pc_plus_mean` reconstruction
The code's direction is not the first principal component; it is (approximately) the top
PC **with the PCA mean added back**, i.e. it carries the *average* masked−clean shift.
Whether the steering effect is driven by the variance direction (PC) or by the mean shift
is a substantive question — mean-dominated would suggest a content-agnostic bias
component, PC-dominated a structured one. Parked for a dedicated discussion; the toggle
exists so the eventual ablation costs nothing to run. Same reconstruction quirk exists in
`obtain_textual_vti` — any conclusion here likely transfers to the textual arm.

At r=0.99 the decomposition is sharpest: h̄ᵢ ≈ the same content-free anchor for every
demo image, so Δᵢ ≈ anchor − hᵢ. The mean of the Δ's is then ≈ (anchor − mean clean
embedding) — a pull-toward-prior / content-erasure term — while the demo-to-demo
variance of the Δ's is just the (negated) variance of the clean embeddings themselves.
So `legacy_pc_plus_mean` (the code) is plausibly dominated by "erase content toward the
prior," while `top_pc` (the paper) captures the dominant axis along which clean images
differ from one another. Different mechanisms, same word "direction" — the dedicated
direction_recon discussion should start from this framing, and it is why that toggle is
now the top-priority Stage-2 test.

Cursor's feasibility review adds a decisive fact (ledger #16): the reference's PCA
batch axis is scrambled by a layer-major `reshape(n_tokens, -1)`, so the legacy
direction's PC component is fit on garbled rows while its mean component survives the
reshape intact. The code's direction is therefore even more mean/prior-dominated than
argued above — its "PC" term is partially structured noise. Expectation on record:
differences between the legacy and `top_pc` cells in the smoke should be attributable
mostly to the mean term, and a "mean-only" direction (mean diff, no PCA at all) is the
natural third point for the eventual dedicated ablation.

### Future work (recorded so it isn't lost)
- **Extraction-site = application-site variants:** currently one extraction (layer
  outputs) feeds all four variants, matching paper and code. A future configuration
  extracts at the MLP output for the MLP-site variants (measure-where-you-apply). Exposed
  later as an extraction param; not built now.
- **mask_ratio characterization sweep** (a study, not a fidelity test): how the direction
  morphs from content-erasure (high r) to perturbation-response (low r); `noise_sigma`
  sweep for gaussian; `include_cls=False` re-run if the smoke's CLS behavior looks odd.
- **Gating** as a fifth geometry, if/when Spherical-Steering-style gates re-enter scope.

---

## Part 2 — Smoke-test plan (lambdab2)

**Purpose:** qualitative first light for the visual arm across the configuration space
above, with the grounding-vs-bias gate attached — NOT statistics. Read responses; the
aggregates at this N are directional only.

**Compute:** lambdab2, one A6000, single GPU, no offload/sharding. `target=lambdab2`.
No RunAI. Scaling to AMBER-450/CHAIR-500 on RunAI is gated on this smoke.

**Samples (pinned, byte-identical to prior qualitative review):**
- **AMBER-25:** the stratified 5-per-(dimension × gold) ids from the last diagnostic's
  sampled set (`llava-1_5-7b-hf_amber_response_samples.md`) — existence×{no},
  attribute×{no,yes}, relation×{no,yes}.
- **CHAIR-5:** the 5 pinned CHAIR smoke samples already known in the repo (confirm the
  id file; flag if none is pinned and pin one). CHAIR prompt verbatim
  `"Please Describe this image in detail."`, `max_new_tokens=64` (frozen, consistent
  with prior runs). n=5 captions are for *reading*, not for CHAIR statistics — report
  chair_s/chair_i but label them qualitative.

**Model:** `llava-hf/llava-1.5-7b-hf`, greedy, Policy-A resolution, single GPU.

**Smoke driver:** a dedicated script under `diagnostic_experiments/vti_visual_smoke/`
(run_scripts driver + config), NOT an extension of `run_eval.py` — many configs, pinned
subset ids, direction-cache management, and gate instrumentation belong in the
diagnostic-experiments pattern. The only `run_eval.py` change is the `--alpha` threading
(build plan). The driver MUST store per-item `p_yes` (via `compute_yes_prob` under the
active hook ctx) for every AMBER item in EVERY cell, **baseline included** — one extra
forward per item, negligible at n=25 — so the gate's c-AUC and c-curve are computed
**offline** from stored JSONs by a small analysis script, not at run time. That analysis
script must be intervention-agnostic: it operates on any cell's `per_item_p_yes.json`,
textual or visual, which is what makes the gate reusable for future geometries on the
LLM backbone.

### Stage 0 — gate validation on the TEXTUAL arm (positive control)

Before trusting the gate on the visual arm, validate it against a known outcome: textual
`uniform_rotation` is an established bias knob (h−=0, threshold slide). On the AMBER-25:
- `no_intervention`, `vti_textual_uniform_rotation_layer` at β∈{0.2, 0.4}, and
  `vti_textual_additive_layer` at β=0.9 (the known mild-calibration case).
- Compute the full gate readout (below) for each.
- **Expected:** textual rotation reproduces the bias signature (yes_ratio shifts, h−≈0,
  AUC ~flat, points on/below the baseline threshold curve); additive shows its milder
  profile. If the gate machinery shows this, it is validated; if not, fix the gate before
  reading any visual result. This also delivers the standing capability you asked for:
  the gate utility must accept **any registered intervention** (textual or visual), so
  future geometric variants on the LLM backbone can be gate-checked the same way.

### Stage 1 — default (paper-faithful) visual configuration, full variant grid

Direction: `perturb_type=patch_mask, mask_ratio=0.99, mask_fill=zero, num_trials=50,
num_demos=70, direction_recon=top_pc, include_cls=True` (one extraction).

Conditions on AMBER-25 and CHAIR-5:
- `no_intervention` baseline (once).
- **Shakedown first:** paper cell `vti_visual_additive_layer` + code cell
  `vti_visual_uniform_rotation_mlp`, each × α∈{0.2, 0.4, 0.9} (6 steered cells per
  benchmark). Read the raw outputs before proceeding — a systemic bug should be caught
  here, not after the full grid.
- Then the remaining two variants (`additive_mlp`, `uniform_rotation_layer`) × the same
  α grid → 12 steered cells total per benchmark. No extra direction-extraction cost:
  all four variants share the single Stage-1 direction artifact (the expensive part —
  ~70×50 ViT forwards — happens once per config).

### Stage 2 — discrepancy toggles, one at a time

Anchor cells: **paper cell** = `additive_layer` (manuscript geometry+site) and
**code cell** = `uniform_rotation_mlp` (reference geometry+site), each at the α that
looked most alive in Stage 1 (default α=0.9 if nothing distinguishes). From each anchor,
toggle ONE variable per run, holding all else at Stage-1 defaults:

1. `direction_recon: top_pc → legacy_pc_plus_mean` (code value) — **top priority** now
   that mask_ratio is settled; this is the N2 question (content-erasure mean term vs
   clean-image variation axis).
2. `mask_fill: zero → mean` (code value).
3. (optional, time permitting) `perturb_type: patch_mask → gaussian_noise` (σ=0.1).
4. (optional curiosity, time permitting) `mask_ratio: 0.99 → {0.5, 0.1}` — no longer a
   fidelity question (N1); a characterization probe of how the direction morphs from
   content-erasure toward perturbation-response as r drops. Expect weak/noisy behavior
   at 0.1 per N1 — that outcome would be consistent with the mechanism, not evidence
   against the method.

Each toggle requires re-extracting the direction (new artifact + metadata); the four
variants reuse the anchor's variant setting. AMBER-25 + CHAIR-5 each.

**Full code-faithful reproduction cell** (all code values at once:
`uniform_rotation_mlp, 0.99, mean, legacy_pc_plus_mean, α=0.9`) — run once as the
"reference behavior" anchor; this is the configuration whose extraction already passed
numerical parity in the build's acceptance check.

### Gate readout (attached to every AMBER cell, stages 0–2)

1. **h− / h+** (from `score_amber_discriminative_records`): h− > 0 = the intervention
   removes hallucinations = necessary grounding signature. h−=0 with h+ climbing = bias
   knob. Headline question: does any visual cell produce h− > 0, which no textual
   rotation cell ever did.
2. **c-AUC:** AUC of per-item `p_yes` (via `compute_yes_prob` under the active hook ctx)
   vs gold, baseline vs steered. AUC up = ranking/separability improved = grounding.
   AUC flat while accuracy moves = threshold slide. Threshold-free; no assumption that
   α/β cleanly instruments yes_ratio.
3. **c-curve:** accuracy-vs-yes_ratio envelope from sweeping a decision threshold τ over
   the frozen **baseline** `p_yes`; overlay every steered cell's operating point. On/below
   the curve = achievable by pure thresholding = bias; above = grounding. At N=25 this is
   directional; the machinery becomes the money plot at RunAI scale.

Prediction on record (Romanus): the yes-bias phenomenon seen with textual rotation is
NOT expected to appear under visual steering. The gate tests this rather than assumes it.

### CHAIR readout (per cell)
chair_s, chair_i, avg_objects_mentioned, avg_caption_len_chars, and the raw 5 captions
side-by-side with baseline (the qualitative point: does visual steering change *what the
model describes* — objects, attributes — vs its length/style).

### Outputs
- Standard dated tree per cell (`evaluation/results/{run_date}/llava-1.5-7b-hf/...`),
  visual cells suffixed `__a{alpha}` (textual Stage-0 cells keep `__b{beta}`);
  responses.json + metric_summary.json + **`per_item_p_yes.json`** (item id, gold,
  p_yes) in every AMBER cell **including baseline** — this is what makes c-AUC/c-curve
  computable offline; direction artifacts with metadata per config, each including its
  `perturbed_examples/` PNG dump (per the build plan) — link these in the consolidated
  summary so the perturbation at each config can be eyeballed alongside its results.
- One consolidated smoke summary markdown: Stage-0 gate-validation table, Stage-1
  variant×α grid (AMBER gate columns + CHAIR columns), Stage-2 toggle table (each toggle
  vs its anchor), raw-response dumps linked. **Facts only, no interpretation** —
  interpretation is the analyst's via Romanus. Append a pointer entry to RESEARCH_LOG.md
  recording configs, artifact paths, seeds.

### Order of operations
1. Build-plan acceptance checks pass (incl. the two-part reference parity test).
2. Stage 0 (gate positive control on textual arm) and Stage 1 generation may run in
   either order or in parallel — per-item `p_yes` is stored in every cell, so gate
   analysis is offline. **Hard rule: Stage 0 must be complete and the gate validated
   before any visual gate readout is interpreted.**
3. Stage 1 shakedown cells → read responses → remaining variants.
4. Stage 2 toggles (skip or reorder based on Stage-1 reading; Romanus decides which
   toggles matter after seeing Stage 1).
5. Consolidated summary + RESEARCH_LOG. No RunAI until Romanus reviews.
