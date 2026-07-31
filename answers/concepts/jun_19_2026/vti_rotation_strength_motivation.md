# Why we ran the VTI rotation-strength experiment (recap)

Date: 2026-06-19
Source of truth: VTI reference `VTI/vti_utils/llm_layers.py`; our
`evaluation/interventions/vti/steer.py`, `hooks.py`, `intervention.py`;
`RESEARCH_LOG.md` 2026-06-18.

## 1. Paper vs released code: a method mismatch
- **The VTI paper describes** additive steering of the **residual stream**: add a
  small multiple of a truthful direction to the full decoder-layer output (post
  residual), `h <- h + alpha * d`.
- **The released code does something different.** `VTILayer.forward`
  (`VTI/vti_utils/llm_layers.py:17-28`) does a **norm-preserving
  renormalization**, not a plain addition:
  ```
  norm = ||x||
  y = mean_i( lam[i] * lambda_sim * normalize(d_i) )
  x = normalize( normalize(x) + 0.1 * y ) * norm     # 0.1 = eps_coeff
  ```
  This rescales the steered vector back to the original norm — geometrically a
  small **rotation on the sphere** (what we call `uniform_rotation`), with a fixed
  blend `eps_coeff = 0.1`.
- **And it applies at the MLP, not the residual.** `add_vti_layers`
  (`llm_layers.py:115-121`) wraps `layer.mlp = nn.Sequential(original_mlp,
  VTILayer(...))` — so the steering hooks the **MLP sub-block output** (before the
  residual add), not the full layer output.
- **The gate is commented out.** In `VTILayer` the `lambda_sim` term (a cosine
  gate) is hardcoded to `1.0` with the real expression commented (`llm_layers.py:22`).

So the method the codebase actually implements **and leaves as the default** is
**`uniform_rotation` at the MLP site** — i.e. our `vti_textual_uniform_rotation_mlp`.
(Our `intervention.py` mirrors this default: `variant="uniform_rotation",
hook_site="mlp"`.) The paper's *described* method (additive, residual stream) is a
different point in the design space.

## 2. The grid of variants we built to cover the mismatch
To compare "what the paper says" against "what the code does", we implemented the
cross product:
- geometries: `additive` (paper) / `uniform_rotation` (code) / `gated_rotation`
  (the code's disabled cosine gate, re-enabled);
- hook sites: `mlp` (code) / `layer` = full residual-stream output (paper).

## 3. Why the residual-site rotations returned EOS / empty outputs
When we ran the **rotation** geometries at the **layer (residual) site**
(`uniform_rotation_layer`, `gated_rotation_layer`) on LLaVA-1.5 at the default
strength (then `alpha_text=0.9`, now `beta=0.9`), the model emitted **EOS as the
first generated token** → empty responses (POPE acc/F1 = 0).

Diagnosis (RESEARCH_LOG 2026-06-18):
- The steering was applied correctly (`cos(x, steered) ≈ 0.996`, the designed ~5°
  rotation), no NaNs — not a code bug.
- The **same rotation is far stronger at the layer site than at the MLP site.** At
  the MLP, the steered tensor is the MLP sub-block output, which is then **diluted
  by the residual add**. At the layer site it **is** the full residual stream, and
  the rotation **compounds across all ~32 layers**.
- Probes localized the collapse to **prefill steering of sequence position 0** (the
  BOS / attention-sink token): leaving position 0 unsteered, or steering only
  during decode, both prevented the collapse. Because every later token attends to
  the sink, rotating it at every layer destabilizes the whole sequence → immediate
  EOS.

So `*_rotation_layer` is not "broken" — it is the paper's residual site driven at
a strength calibrated for the (weaker) MLP site, i.e. **mis-calibrated**.

## 4. Why we then swept beta on residual-site rotation
That made the open question a **calibration** question, not a bug hunt: *does the
residual-site rotation have a usable operating band below the collapse, and what
does the strength knob actually do there?* Hence the **VTI rotation-strength
experiment** — sweep `beta` for layer-site rotation over N POPE samples and, per
beta, measure POPE accuracy/precision/recall/F1, yes_ratio, length, the
empty/identical/changed split, and the decision-flip decomposition (suppressions
vs induced/removed hallucinations). The goal is to learn whether strength is just
a suppression knob (shorter, more "no" answers) or whether there is a regime where
rotation genuinely corrects content — and whether that differs by model backbone.

## TL;DR
- Code ≠ paper: the released VTI code does **norm-preserving rotation at the MLP**
  (gate disabled), and that is the default; the paper *describes* **additive
  steering on the residual stream**.
- Rotating the **residual stream** at the MLP-calibrated strength over-steers
  (compounds across layers; corrupts the position-0 attention sink during prefill)
  → immediate EOS / empty output.
- So we swept **beta** on residual-site rotation to find its real operating band
  and characterize what the strength knob does (suppression vs genuine correction).
