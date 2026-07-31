# The VTI `eps_coeff` (the hidden second strength coefficient)

Date: 2026-06-22
Question: "Is there any documentation in the original VTI code that explains the
use of `eps_coeff`? My manager pointed out it conflates additive vs rotation."

## Short answer
**No.** There is no comment, README note, or paper description of `eps_coeff`
anywhere in the original VTI repo. It exists only as a hardcoded literal `0.1`
inside `VTILayer.forward` (`VTI/vti_utils/llm_layers.py:28`):

```python
x = F.normalize(F.normalize(x.float(), dim=-1) + 0.1 * y, dim=-1) * norm
```

The VTI `README.md` documents only `--alpha_image` (α) and `--alpha_text` (β) as
the strength knobs; the `0.1` blend factor is never mentioned. We inherited it
verbatim and surfaced it as the named parameter `eps_coeff` (default 0.1) in
`evaluation/interventions/vti/steer.py`.

## What it does (mechanics)
In our `steer.py` the coefficient enters the two geometries differently:

- **additive** (`steer.py:45`): `x_out = x + alpha · d`  (d = unit direction).
  The full `alpha` (= our `beta`) multiplies the unit direction. `eps_coeff` is
  never used.
- **uniform/gated rotation** (`steer.py:72`):
  `x_out = normalize( normalize(x) + eps_coeff · y ) · ‖x‖`, with
  `y = alpha · lam · d` and `lam = 1`. So the perturbation added to the
  unit-normalized activation has magnitude `eps_coeff · alpha = 0.1·beta`, then
  it is renormalized back to the original norm `‖x‖`.

Perturbing a unit vector by `0.1·beta·d` and renormalizing is a pure **rotation**
by angle θ with `tan θ ≈ eps_coeff·beta`. At β=0.9 → θ ≈ 0.09 rad ≈ 5.1° →
`cos θ ≈ 0.996`, which is exactly the `cos(t, steered) ≈ 0.996` measured during
the layer-site collapse debugging. **The effective rotation coefficient is
`0.1 · beta`** — the nominal β is silently divided by ten in the rotation path.

## Why it conflates additive vs rotation
Comparing "additive @ β" vs "uniform_rotation @ β" at the same β is not a
strength-matched comparison, for two independent reasons:

1. **Hidden 10× factor.** Additive scales the unit direction by `beta`; rotation
   scales it by `eps_coeff·beta = 0.1·beta` before renormalizing. Same nominal β,
   ~10× different nominal magnitude.
2. **Different dependence on activation norm.** Additive adds an *absolute* vector
   of length `beta` to `x` (whose norm at the layer site can be ~18–78), so its
   *relative* effect is `beta/‖x‖` — norm-dependent. Rotation yields a *fixed
   angle* `≈0.1·beta` independent of `‖x‖`. Even after correcting the 10×, the two
   are different kinds of perturbation (norm-relative translation vs
   norm-invariant rotation).

This likely explains why the rotation-strength sweep needed β pushed toward
0.5–0.6 before LLaVA collapsed (a ~5° per-layer rotation only destabilizes once
it compounds across 32 layers), whereas additive at β=0.9 is already a large
nudge. The two knobs live on different scales.

## Options to de-conflate
- Set `eps_coeff = 1.0` and let `beta` be the only knob; or always report the
  **effective** coefficient (`eps_coeff·beta` for rotation, `beta` for additive)
  so tables compare like-for-like.
- Or match the *outcome* (induced Δ-angle or Δ-norm) instead of the raw
  coefficient, since additive (a translation) and rotation (an angle) are not
  dimensionally identical regardless of the constant.

## Pointers
- Original constant: `VTI/vti_utils/llm_layers.py:28` (`0.1 * y`), undocumented.
- Original README knobs: `VTI/README.md` (only `--alpha_image`, `--alpha_text`).
- Our implementation: `evaluation/interventions/vti/steer.py:19` (`eps_coeff=0.1`
  default), `:45` (additive, no eps), `:72` (rotation, uses eps).
- Not verified: whether the paper's appendix (arXiv:2410.15778) mentions a 0.1
  blend; the public README/code do not. Worth a methods/appendix check if needed.
