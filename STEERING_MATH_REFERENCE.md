# Steering geometry and PCA — reference, understanding status, and open questions

**Date:** 2026-07-15 (updated same day: added §6, diagnostic choice for the perception plots; handoff summary renumbered to §7)
**Origin:** Claude discussion chat (PCA / linear algebra / trig track), spun off from "Spherical steering for VLM object recognition tasks."
**Purpose:** (1) Record of the full math walkthrough of VTI-style direction extraction and steering geometry, grounded in the repo code (`directions.py`, `directions_v2.py`, `pca.py`, `steer.py`). (2) Explicit map of what Alex has verified he understands vs. what is pending. (3) Handoff reference for the other Claude instance to plan which experiments can run *now* without depending on the pending math.

**How to read the status flags:**
- ✅ VERIFIED — Alex worked through this and confirmed understanding.
- ⚠ PENDING — derivation recorded here but not yet understood; do not build experiment *interpretation* on it until cleared.
- ⚠⚠ EXPLICIT CONFUSION — Alex named this specific step as the point where understanding breaks down; first target when the math discussion resumes.

---

## 1. Code facts (grounded in the uploaded files; no math prerequisite) — ✅ VERIFIED

These are readings of the actual repo code, not derivations. Safe to plan against.

**`steer.py` (applied per layer, per forward, at the hook site):**
- `additive`: `x' = x + α·d̂`, where `d̂` is the per-layer direction unit-normalized at application time. No norm preservation.
- `uniform_rotation`: `x' = normalize(normalize(x) + eps_coeff·α·d̂) · ‖x‖`. Effective chord step is **ε = eps_coeff · α**, with `eps_coeff` defaulting to the reference code's hardcoded **0.1**. So the grid's β ∈ {0.2, 0.5, 0.9} corresponds to ε ∈ {0.02, 0.05, 0.09} — small rotations (≤ ~5°; see §4). **β multiplies 0.1; it does not replace it.**
- `gated_rotation`: same, with λ = 1 + max(0, cos(x, −d̂)) ∈ [1, 2] multiplying the step. The gate boosts steering for activations leaning toward the *hallucinated* side. **Implementation quirk:** the gate only computes when `x.size(1) < 2`, i.e. during decode steps; during prefill, `gated_rotation` silently behaves as `uniform_rotation`.
- Additive α is in **absolute activation units**; rotation ε is **dimensionless (≈ radians of arc)**. Cells "at the same β" across methods are not matched intervention strengths.

**`directions_v2.py` + `pca.py` (extraction):**
- **Pairwise:** one diff per demo pair, `value − h_value` (clean − hallucinated), from last-token hidden states of the full (image + question + caption) forward.
- **Joint flattened PCA:** each pair's `(num_layers+1, hidden_dim)` stack is flattened to one ~135k-dim vector (LLaVA: 33 × 4096); a single PCA is fit on the (N, flat) matrix. **Not per-layer PCA** — per-layer directions are slices of one global component (their squared norms sum to 1; logged as `pc1_layer_norms`). The overnight plan text said "per layer"; the docstring says the flatten is deliberate (author-code geometry). → Confirm intent with Cursor.
- **Centered:** `pca.py` subtracts the row mean before SVD. Components are principal directions of the diffs *around their mean*.
- **The steering object is `direction = PC1 + mean`** — the raw, full-magnitude mean diff δ̄ plus one unit of centered PC1. Relative weighting is set by units, not principle: ‖PC1‖ = 1 in flat space, ‖δ̄‖ is whatever raw activation units give. If ‖δ̄‖ ≫ 1 the composite is essentially CAA (mean-diff) steering; if ‖δ̄‖ ≲ 1, PC1 materially bends it.
- **Sign convention:** `svd_flip` (largest-|projection| positive), **not aligned to the mean diff** (`SIGN_CONVENTION` in the file says so explicitly). PC1's sign relative to δ̄ is incidental and can flip between (dimension, N) configs in the nested sweep — a comparability hazard if the mean does not dominate.
- `steer.py` re-normalizes each layer's slice of `PC1 + mean` at application time.
- Rank-2 fit; PC2 cached in `components.npz` (with `pca_mean_flat`) for the projection diagnostic; steering reconstruction uses component 0 + mean only.

---

## 2. PCA on the demos — ✅ VERIFIED

**Setup.** For dimension d and sample size N: pairs of last-token, per-layer hidden states for truthful vs. hallucinated captions; per-pair diff δᵢ = aᵢ⁺ − aᵢ⁻ (clean minus hallucinated, so +β points toward truthful); stack into Δ.

**Where pairing matters.** Three recipes:
1. Difference of means: mean(clean) − mean(hallucinated).
2. Mean of paired diffs: (1/N) Σ (aᵢ⁺ − aᵢ⁻).
3. PCA of paired diffs.

(1) and (2) are algebraically identical when pairs are complete — for the *mean* component, pairing buys nothing. Pairing is load-bearing for (3) and any variance-based statistic: each per-pair diff cancels everything shared within the pair (image, question, unchanged caption content) *before* second-order statistics are computed. PCA on pooled unpaired states would be dominated by between-image variance. Pairing = within-subject design; differencing removes the per-image random effect.

**Centered vs. uncentered (2D toy that resolved it).** 100 diffs ≈ (10, 0) + noise, noise mostly vertical.
- *Uncentered* top direction (best line through the origin, maximizing Σ⟨δᵢ, u⟩²): ≈ (1, 0) ≈ normalized mean — recovers the consistent shift.
- *Centered* PC1: subtract δ̄ = (10,0) first; the shift is gone by construction (moved into `mean_`); PC1 ≈ (0, 1) — the axis along which pairs disagree *with each other*, possibly orthogonal to the mean.
- Centered PC1 answers "how do pairs differ from each other," not "how does clean differ from hallucinated." The consistent contrast lives entirely in δ̄.
- This is why the ICV/VTI reconstruction adds the mean back: `PC1 + mean` restores the shift and rides one unit of the dominant disagreement axis on top. Whether the PC1 unit helps (real substructure, e.g. hallucination subtype) or hurts (nuisance: caption length, edit style, COCO category) is empirical, not settled by the method.
- StatQuest mapping: the video's "shift the data so the average is at the origin" step *is* the centering; its PC1 is always the centered object.

**Small-N caveat.** N ≤ 500 samples in ~135k dims: covariance rank ≤ N−1; explained-variance ratios at small N are mechanically inflated. PC1 *stability across the nested-N sweep* (cosine of the N=50 vs. N=500 directions) is the honest signal check, not any single EVR number.

---

## 3. Framing corrections carried over from earlier in the discussion — ✅ VERIFIED (accepted)

Recorded because they constrain how results get written up:
- The hallucinated captions are **synthetic minimal edits**, not dishonesty. The guaranteed contrast in the pairs is "text consistent vs. inconsistent with the image," in caption context, plus confounds (edit style, counting hedge-nouns, category distribution). Whether the direction aligns with an internal honesty/agreeableness axis used at decision time is a hypothesis the diagnostics test, not a premise.
- The claim "additive steering does not move representations toward leading-sentiment" is **not established**. What the data supports is an asymmetry: additive on LLaVA/POPE produced some genuine flips (h⁻ > 0); rotation produced h⁻ = 0 everywhere and slid yes_ratio, with **model-specific sign** (Qwen2.5 more agreeable, LLaVA less, under the same direction sign).
- Defensible hypothesis form: *rotation modulates a response-prior axis (possibly the same axis separating neutral from leading prompts) without touching the evidence channel; additive at least partially reaches the evidence channel.* Decomposes into two testable sub-claims: (a) directional alignment between the steering displacement and the (leading − neutral) prompt displacement; (b) evidence attenuation specific to rotation. Sign of any sycophancy coupling is model-dependent.
- Extraction/application mismatch: directions are extracted at the last caption token but applied at **every** position of a structurally different input (including vision tokens during prefill).

---

## 4. Geometry of the two steering operations — ⚠ PENDING (derivations recorded; not yet understood)

Everything below is written step-by-step so it can be studied later. Alex confirmed §§1–3; understanding breaks down starting here.

### 4.0 Prerequisites to refresh before re-deriving (the actual study list)

1. **Dot product and angle** — why ⟨x̂, v⟩ = cos φ for unit vectors; dot product as projection length. ⚠⚠ EXPLICIT CONFUSION feeds from here.
2. **Orthogonal decomposition** — splitting a vector into components parallel and perpendicular to another vector; verifying orthogonality with a dot product. ⚠⚠ EXPLICIT CONFUSION: where ‖v⊥‖² = sin²φ comes from.
3. **Orthonormal bases and coordinates** — what it means to write v = cos φ·e₁ + sin φ·e₂; why choosing e₁ = x̂ and e₂ = v⊥/‖v⊥‖ is legitimate and what it buys. ⚠⚠ EXPLICIT CONFUSION: where e₁, e₂ "come from."
4. **Right-triangle trig** — tan θ = opposite/adjacent for a vector with coordinates (a, b).
5. **Pythagorean identity** sin²+cos²=1; **sine difference formula** sin(A−B) = sinA·cosB − cosA·sinB. (These two are the *only* identities the whole derivation uses.)
6. **Small-angle approximations** — tan θ ≈ θ for small θ.
7. **Unit circle picture of sin/cos** — trig functions as coordinates of circle points (this is what makes the slerp formula transparent).

Suggested refreshers (short): 3Blue1Brown *Essence of Linear Algebra* — dot products (ch. 9), basis/coordinates (ch. 2); any unit-circle trig review for (5)–(7).

### 4.1 Why 2D pictures are exact

Every operation in `steer.py` (add β·v; divide by a norm; multiply by a norm) outputs a vector in span{x, v} — the 2-plane containing the activation and the direction. Scalar multiples and linear combinations never leave the plane; the other ~4094 dimensions are untouched. A 2D figure of that plane is therefore a complete picture, not a projection. High-dim caveats: each token position has its own plane and its own φ; and by concentration of measure, unrelated directions in R⁴⁰⁹⁶ have cos φ ≈ 0 ± 1/√4096 ≈ ±0.016, so typical activations sit near φ = 90° (sin φ ≈ 1) — deviations of φ from 90° are themselves signal (what the gate reads).

### 4.2 Decomposition (the step containing both explicit confusion points)

Let x̂ = x/‖x‖, v unit. Define φ by cos φ = ⟨x̂, v⟩.
- v∥ = (cos φ)·x̂ (component of v along x̂).
- v⊥ = v − (cos φ)·x̂. Orthogonality check: ⟨v⊥, x̂⟩ = ⟨v, x̂⟩ − cos φ·⟨x̂, x̂⟩ = cos φ − cos φ = 0.
- ‖v⊥‖² = ⟨v − cosφ·x̂, v − cosφ·x̂⟩ = ‖v‖² − 2cosφ·⟨v, x̂⟩ + cos²φ·‖x̂‖² = 1 − 2cos²φ + cos²φ = 1 − cos²φ = **sin²φ**.  ⚠⚠
- Orthonormal basis of the plane: e₁ = x̂, e₂ = v⊥/sin φ. Then **v = cos φ·e₁ + sin φ·e₂** — ordinary coordinates of a unit vector at angle φ.  ⚠⚠

### 4.3 nlerp (uniform_rotation) — angle, norm, contraction

Chord point: w = x̂ + εv = (1 + ε cos φ)·e₁ + (ε sin φ)·e₂, with ε = eps_coeff·α·λ.
- **Angle** (right-triangle trig, no identity): tan θ = (ε sin φ)/(1 + ε cos φ). Small ε: θ ≈ ε sin φ. Grid values ε ≤ 0.09 → θ ≤ ~5°, achieved only at φ = 90°.
- **Norm of the chord:** ‖w‖² = (1+ε cosφ)² + (ε sinφ)² = 1 + 2ε cos φ + ε² (Pythagorean identity collapses the ε² terms).
- **Norm preservation is trig-free:** ‖x‖ is factored out as a scalar first, the direction is updated on the unit sphere, and ‖x‖ is multiplied back. The magnitude never participates in the update.
- **Dead zones:** φ = 0 (already aligned) and φ = 180° (anti-aligned) give θ = 0 — the chord is purely radial and renormalization erases it. Rotation is intrinsically gated by sin φ. `gated_rotation`'s λ = 1 + max(0, −cos φ) doubles the step exactly where sin φ is shrinking (hallucination-leaning side) — it compensates for the geometry's dead zone. (Resolved exercise from the prior turn.)
- **Contraction:** rewriting, x' = (x + ε‖x‖·v) / ‖x̂ + εv‖. Rotation with coefficient ε is additive with an *adaptive step ε·‖x‖*, followed by a global rescale by 1/√(1 + 2ε cos φ + ε²). The rescale multiplies **everything already in x** — when cos φ > 0, original content is uniformly shrunk to make room for v. Additive superposes without erasure; rotation superposes *and* attenuates.

### 4.4 Additive — dilution

x' = x + βv = (‖x‖ + β cos φ)·e₁ + (β sin φ)·e₂.
- tan θ_add = (β sin φ)/(‖x‖ + β cos φ) ≈ (β/‖x‖)·sin φ for β ≪ ‖x‖. The adjacent leg is ‖x‖ instead of 1 — adding a short vector to a long one barely turns it. Additive's angular effect dies as 1/‖x‖; rotation's does not.
- Norm changes: ‖x'‖² = ‖x‖² + 2β‖x‖cos φ + β².
- **Unit mismatch, concrete:** at ‖x‖ = 100, uniform_rotation at α = 0.5 injects an effective additive step of 0.1 × 0.5 × 100 = 5 — ~10× additive's literal 0.5 at the "same" coefficient — and delivers the same angular kick to every position regardless of norm, including high-norm sink positions additive cannot budge.

### 4.5 Exact slerp (Spherical Steering paper) — where the trig actually lives

Only two identities are used anywhere: sin²+cos²=1 and sin(A−B) = sinA·cosB − cosA·sinB.

slerp(x̂, v; t) = [sin((1−t)Ω)·x̂ + sin(tΩ)·v] / sin Ω, with cos Ω = ⟨x̂, v⟩. Substituting v = cos Ω·e₁ + sin Ω·e₂:
- e₁ coefficient: [sin((1−t)Ω) + sin(tΩ)cos Ω]/sin Ω. Expand sin((1−t)Ω) = sin(Ω − tΩ) = sinΩ·cos(tΩ) − cosΩ·sin(tΩ); the cross terms cancel → **cos(tΩ)**.
- e₂ coefficient: sin(tΩ)·sinΩ/sinΩ = **sin(tΩ)**.
- So u(t) = cos(tΩ)·e₁ + sin(tΩ)·e₂: literally *the point at angle tΩ on the unit circle* in the plane of x̂ and v, written in the non-orthogonal basis (x̂, v). Unit norm is immediate; angle grows linearly in t (constant angular velocity).
- nlerp walks the straight chord and projects radially back: same plane, circle, endpoints; non-uniform angular speed. At chord steps 0.02–0.09 the difference is third-order (cosmetic); it matters in the Spherical Steering paper because their calibrated vMF-gated steps are larger.

### 4.6 Mechanism-level interpretation — ⚠ PENDING and CONJECTURE (both flags)

Consistent with the data but not confirmed by it: rotation = prior injection + evidence attenuation ⇒ pure operating-point (threshold) movement ⇒ the observed h⁻ = 0 signature; additive = bias on top of unchanged evidence terms, with 1/‖x‖ dilution and no reach into high-norm carrier positions. The diagnostics exist to probe exactly this. Do not commit this story to writing before the diagnostics run and before §4 is understood.

---

## 5. Open items

### 5.1 Zero-cost artifact checks — NO math prerequisite; can run immediately (CPU, offline, from cached artifacts)
1. Per config (all 40): from `components.npz` (`pc0`, `pca_mean_flat`) and `directions.npz`, compute per-layer **cos(direction_ℓ, δ̄_ℓ)** and **‖δ̄_ℓ‖ / ‖PC1_ℓ‖**. Settles whether steering has effectively been CAA (mean-dominated) or PC1-shaped, per layer.
2. Check **PC1 sign stability across the nested-N sweep** per (model, dimension): cos between PC1 at N=50/100/200/500. `svd_flip` makes sign flips possible; flips matter only if the mean does not dominate (see check 1).
3. Confirm whether `run_eval.py` ever overrides `eps_coeff` from the 0.1 default (determines what the grid's β axis means).
4. Confirm with Cursor that joint-flattened PCA (vs. the per-layer PCA in the overnight plan text) is the intended v2 semantics; record in IMPLEMENTATION.md either way.
5. PC1 stability across N (cosine N=50 vs N=500) as the small-N signal check (§2).

### 5.2 Experiments discussed in the other chat that do NOT depend on the pending math
(Candidates for the interim plan; final scoping belongs to the other chat.)
- **Leading-clause augmentation** of the pinned POPE/AMBER subsets — CPU-side prompt construction, no dependency on any result.
- **Yes-logit threshold-equivalence test** — compares steering under leading prompts against thresholding P(yes); uses existing per-item yes-probability machinery (c-AUC infrastructure). Interpreting it needs §3, which is cleared, not §4.
- **Variance/seed pass** on POPE + AMBER for the bias-knob claim (flagged urgent in the prior chat: the single-direction "agree/disagree knob" story is already benchmark- and model-conditional).
- **New `no_intervention` CHAIR baselines at max_new_tokens=512** (comparability with June 22 baselines is already broken by design).
- **RunAI pinned-subset reproduction** (CHAIR-500, AMBER-450) — already deferred there; mechanical.
- **MM-SY dataset inspection** on HuggingFace (confirmed available; verify usability/protocol fields).
- **Perception-plot infrastructure (§6)** — activation dumps for held-out pairs and leading-clause-augmented subsets, per-layer linear probes, fixed-frame projection code. Building and running is §5.2-class; only the rotation-vs-additive matched-strength *readout* is §5.3-class (see §6.3).

### 5.3 Blocked until §4 is understood (defer; return to the math chat first)
- **Representation-level cosine diagnostics**: cos(steered(x_neutral) − x_neutral, x_leading − x_neutral) per layer (sub-claim (a)); evidence-attenuation measurement (sub-claim (b)). Designing and reading these *is* the geometry.
- **Matched-angular-displacement comparisons** between additive and rotation (requires internalizing the unit mismatch, §4.3–4.4).
- Any manuscript text about the mechanism (§4.6).

### 5.4 Questions queued for this chat when the math discussion resumes
1. Dot product → cos φ; orthogonal decomposition; ‖v⊥‖² = sin²φ (⚠⚠).
2. Orthonormal basis construction e₁, e₂ and what "coordinates in the plane" means (⚠⚠).
3. Re-derive the nlerp triangle and contraction factor end-to-end without notes.
4. Slerp derivation and why trig functions are circle coordinates.
5. Small-N PCA behavior (why EVR inflates; Marchenko–Pastur intuition) — optional depth.

---

## 6. Diagnostic choice for measuring model "perception" of hallucination / leadedness — ✅ VERIFIED (discussed 2026-07-15)

**Context.** Modeled on ShiftDC ("Understanding and Rectifying Safety Perception Distortion in VLMs"). Their Figure 5 is a t-SNE of **layer-15 last-token activations** over four groups (safe/unsafe × text-only/vision-language). Two reading notes that shaped the decision here: the quantitative backbone of that paper is not the t-SNE but **per-layer linear probes** (their Figure 3, classification accuracy per layer, plus Figure 4 confusion matrices) — the t-SNE illustrates what the probes established; and the red "boundary" line in their Figure 5 is drawn *in the 2D embedding space*, corresponding to no hyperplane in activation space (decorative, not a claim).

### 6.1 t-SNE vs. linear (PCA-plane) projection — the decision and why

- **Linear projection:** deterministic; plotted distances and directions are real (projected) activation-space geometry; the axes are vectors in R^d that can be reused (cosines against the steering direction, out-of-sample transform of new points is a matrix multiply).
- **t-SNE:** nonlinear, stochastic (seed-dependent), perplexity-sensitive; preserves *local neighborhood* structure only. Cluster sizes and between-cluster distances in the output are **not meaningful**; it can manufacture apparent clusters from unclustered data; vanilla t-SNE has **no out-of-sample transform**.
- **Opposite failure modes.** A PC1/PC2 plot only shows separation living in the top-variance subspace — with image-content variance dominating, classes can look intermixed in the plane while a linear probe separates them at 95% along a lower-variance direction (absence of separation in the plane is weak evidence of absence). t-SNE errs the other way: crisp-looking clusters can reflect nonlinear or nuisance structure ("separable in t-SNE" ≠ linearly separable).
- **Precision on the takeaway** (avoid overcorrecting): t-SNE is not zero-information — it can indicate that *some* local cluster structure exists. It is rejected as the working diagnostic because it cannot (i) certify **linear** separability, the property relevant to linear steering interventions and linear direction extraction, or (ii) provide a **fixed frame** for cross-condition comparison. It remains acceptable as a one-off "does any structure exist" picture on baseline data, or as figure dressing.
- **The decisive requirement:** comparing conditions (no_intervention / additive / rotation) needs a frame **fit once on baseline activations** and applied unchanged to every condition, so that displacement of the same items is directly visible. Linear projections satisfy this trivially; t-SNE forces either per-condition re-embedding (mutually incomparable maps) or joint embedding (displacement geometry uninterpretable).
- **Claims rest on cross-validated linear probes** (per-layer accuracy/AUC sweeps, as in their Figure 3); 2D plots illustrate, never carry the claim. Report the layer sweep; don't cherry-pick the illustration layer.
- **Preferred frame** over raw PC1/PC2 (nuisance-dominated): a supervised frame — x-axis = class difference-in-means direction (or the probe weight vector), y-axis = top PC of the residual after projecting axis 1 out. Axis 1 is the hypothesis direction; axis 2 shows within-class spread.
- **Be explicit which PCA.** PCs of the *paired diffs* (the extraction space — this is the already-planned PC2 projection diagnostic; the axes are literally the steering components) vs. PCA of *pooled raw activations* (image-content dominated). Different planes, different questions.

### 6.2 The equivalent plots for the two research directions (recipe abstracted from ShiftDC)

General recipe: (1) pick the token position where the property should be *decided*; (2) pick groups whose separation operationalizes "the model perceives the property"; (3) establish separability with per-layer probes; (4) illustrate with a 2D projection; (5) overlay intervened activations **in the same baseline-fit frame** to show clusters moving as the method predicts (their Figure 8-right move).

**Leadedness / sycophancy — near-direct analogue (leadedness is an *input* property):**
- Site: last token of (image + question [+ leading clause]), pre-answer, per layer.
- Groups: neutral / leading-toward-yes / leading-toward-no. Items are **paired** (same image+question ± clause) → per-item displacement vectors (leading − neutral), the same within-pair cancellation as the demos.
- Hypothesis plot: frame fit on baseline; x = mean leading-displacement direction, y = steering direction (orthogonalized against x); overlay steered-neutral points; ask whether rotation moves the neutral cluster toward the leading cluster while additive does not.
- Quantitative core: per-layer cos(steered − unsteered, leading − neutral). Probe version: train leading-vs-neutral probes, then test whether they fire on steered-neutral inputs.

**Truthfulness of outputs — needs translation (the property lives in the *output*; there is no "hallucinated input"):**
- *Caption-reading version* (machinery exists): last-token activations of (image + question + caption) for **held-out** truthful vs. hallucinated pairs — held-out is non-negotiable (projecting the extraction demos onto their own PCs separates trivially). **Confound:** last-token states strongly encode current-token identity; if pair captions end in different words, separation can be trivially lexical. Mitigations: append a fixed terminator to both, mean-pool over caption tokens, or restrict to pairs with non-sentence-final edits.
- *Decision-state version* (matches the actual output goal): last-token activation of (image + question) at the pre-answer position on POPE/AMBER items, colored two ways — (i) by **gold label** (does the state encode object presence: the evidence channel) and (ii) by **model outcome** (will this item be answered wrongly). Overlay all conditions in the baseline frame. **Bias-knob prediction:** rotation translates the entire cloud along one axis *without* improving gold-label separability (probe AUC flat — the c-AUC result rendered geometrically); additive is the candidate for actually changing separability. If observed, this is the single figure that makes the operating-point argument legible.
- *Free-generation (CHAIR) equivalent* — activations at positions where the model commits to an object word — deferred (real lift: requires identifying commitment positions).

### 6.3 Dependency status

Infrastructure (leading-clause augmentation, activation dumps, per-layer probes, fixed-frame projection code) is **§5.2-class: runnable now** — probe training and matrix projections, nothing from §4. What stays **§5.3-class** is the *interpretation* of rotation-vs-additive comparisons at matched strength, which depends on the §4 unit-mismatch and contraction results.

---

## 7. One-paragraph handoff summary for the other Claude instance

Alex has verified understanding of: the code's actual extraction and steering operations (§1), pairwise-diff PCA including centered vs. uncentered and the `PC1 + mean` composite (§2), and the framing corrections (§3). He has *not* yet worked through the vector-geometry derivations (§4) — dot-product/decomposition/orthonormal-basis steps are the explicit sticking points — so any experiment whose *design or interpretation* leans on §4 (the representation-level cosine diagnostics, matched-displacement comparisons, mechanism write-ups) should be deferred. Everything in §5.1 and §5.2 is runnable now: the artifact checks are offline CPU one-liners against cached `.npz` files, and the behavioral experiments (leading-clause augmentation, threshold-equivalence, seed/variance pass, CHAIR-512 baselines, RunAI reproduction) require only §§1–3. Compute routing discipline unchanged: smoke on lambdab2, scale on RunAI, unsharded single-GPU for anything with hooks. §6 records the agreed diagnostic design for the perception plots — fixed-frame linear projections plus per-layer probes, not t-SNE — with the concrete plot recipes for both the sycophancy and hallucination directions; its infrastructure is runnable now, and only the rotation-vs-additive matched-strength readout waits on §4.
