# SVD sign-flip / sign-alignment in the truthful-vs-hallucinated PCA

Date: 2026-07-29
Question: "What is the SVD sign-flip alignment phenomenon when fitting PCA to
truthful-vs-hallucinated caption differences, how does it affect the global (VTI-paper) versus
the per-layer PCA steering-vector estimator, and why does it occur in the first place?"

Scope: mechanism + method note, written against the four scripts that implement PCA in this
repo. General mathematics is stated as such; every code claim carries a `file:line` citation.
One claim about `svd_flip` indexing was verified by execution, not by reading alone — §3.3.
No project results are used or interpreted here.

---

## 0. The four scripts, and which is which

| Script | Role | Sign handling |
|---|---|---|
| `evaluation/interventions/vti/pca.py` | The shared `PCA` class and `svd_flip`. Every path below calls this. | Defines the convention |
| `evaluation/interventions/vti/directions.py:90-114` | `obtain_textual_vti` — the VTI-paper global flatten PCA | inherits `svd_flip`, no anchoring |
| `evaluation/interventions/vti/directions_v2.py:252-274` | `_live_pca_fit` — the demos_v2 / demos_850 global path actually deployed | inherits `svd_flip`, no anchoring (documented at `:42-45`) |
| `evaluation/interventions/vti/perlayer_pca.py:124-154` | `perlayer_pca_fit` — the per-layer estimator from the last experiment | inherits `svd_flip`, no anchoring, **one decision per row** |

A fifth path, `visual_directions.py:51-54`, is the only one in the repo that *does* anchor the
sign. It is discussed in §3.4 as the contrast case.

---

## 1. The ambiguity itself

For a centered data matrix $Z$, the SVD $Z = U S V^\top$ is not unique. For any diagonal
matrix $D$ with entries $\pm 1$:

$$
Z = (UD)\,S\,(VD)^\top
$$

Flipping the $i$-th left singular vector and the $i$-th right singular vector *together*
leaves the factorization exactly intact. Equivalently: PCA components are unit eigenvectors
of the covariance, and an eigenvector is defined only up to scale — normalization pins
$\lVert v \rVert = 1$ but leaves the $\pm$ free.

Everything PCA is *supposed* to give you is invariant to this: the spanned subspace, the
singular values, the explained-variance ratios, the reconstruction error, and
$\lvert \cos \rvert$ between the components of two fits. Only the sign is arbitrary. LAPACK
returns whatever the numerical routine happens to produce — deterministic for a given input,
but an arbitrary and discontinuous function *of the data*.

---

## 2. `pca.py` — where the convention is imposed

`PCA.fit` (`pca.py:24-39`) does three things that matter here.

**2.1 It promotes 2-D input to a batch of one.** `pca.py:26-30`:

```python
if X.ndim == 2:
    n, d = X.size()
    X = X.unsqueeze(0)          # (1, n, d)
elif X.ndim == 3:
    _, n, d = X.size()          # (B, n, d) — B independent fits
```

This single branch is the entire structural difference between the global and per-layer
estimators. The global paths hand in a 2-D matrix and get $B = 1$. `perlayer_pca_fit` hands
in a 3-D tensor and gets $B = L{+}1$ independent fits. Everything downstream — including the
number of independent sign decisions — follows from which branch was taken.

**2.2 It centers, decomposes, and flips.** `pca.py:33-38`:

```python
self.register_buffer("mean_", X.mean(1, keepdim=True))   # (B, 1, d)
Z = X - self.mean_
U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
Vt = Vh
U, Vt = svd_flip(U, Vt)
self.register_buffer("components_", Vt[:, :d])           # (B, rank, d)
```

Note that `U` is returned by `svd_flip` and then never used again — only `components_` (from
`Vt`) and `mean_` are registered. Any effect the flip has on `U` is discarded.

**2.3 `svd_flip` itself.** `pca.py:6-16`:

```python
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 1)     # dim 1 = the sample axis
    i = torch.arange(u.shape[2]).to(u.device)
    max_abs_cols = max_abs_cols.unsqueeze(-1)
    signs = torch.sign(torch.gather(u, 1, max_abs_cols))
    u *= signs
    v *= signs.view(v.shape[0], -1, 1)
    return u, v
```

The intent is sklearn's: **the sign of each component is chosen so that the single sample with
the largest-magnitude loading on that component has a positive loading.** Here a "sample" is
one demo pair. `v *= signs.view(v.shape[0], -1, 1)` applies `signs[b, j]` to row $j$ of $V^\top$
in batch $b$, which is the correct axis.

This is a *convention*, not an anchoring to anything semantic. Nothing in it refers to the
truthful → hallucinated polarity of the contrast. `directions_v2.py:42-45` states this
outright:

```
SIGN_CONVENTION = (
    "live_textual: PCA.svd_flip on U/V (max-abs column of U); "
    "steering direction = (PC1 + mean) reshaped — no mean-diff sign-align"
)
```

---

## 3. Why the sign is a real perturbation here, not a cosmetic one

### 3.1 The reconstruction adds a sign-arbitrary vector to a sign-definite one

If the steering vector were PC1 alone, the sign would be a nuisance you handle with
$\lvert \cos \rvert$. It is not. All three textual paths reconstruct as $\mathrm{PC1} + \text{mean}$.

**VTI-paper global** — `directions.py:101-114`:

```python
hidden_states_all.append((pos.view(-1) - neg.view(-1)).float())   # flatten (L+1)*h
fit_data = torch.stack(hidden_states_all)                          # (N, flat) → 2-D
pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
direction = (
    pca.components_.sum(dim=1, keepdim=True) + pca.mean_
).mean(0).view(...)
```

with `rank: int = 1` (`directions.py:93`), so `.sum(dim=1)` over the rank axis is just PC1.

**Deployed global** — `directions_v2.py:293-303` and `:261-266`:

```python
d = (v_act.reshape(-1) - h_act.reshape(-1)).float()   # value − h_value, flattened
fit_data = torch.stack(diffs, dim=0)                  # (N, flat) → 2-D
...
pca = PCA(n_components=rank).to(diffs_flat.device).fit(diffs_flat.float())
components = pca.components_.squeeze(0)               # (rank, flat)
mean = pca.mean_.reshape(-1)                          # (flat,)
direction_flat = (components[STEER_COMPONENT] + mean).float()
```

`STEER_COMPONENT = 0` (`directions_v2.py:46`). This path fits `rank=2` so PC2 is available for
diagnostics but indexes component 0 explicitly for steering, rather than summing — at rank 1
the two are identical.

**Per-layer** — `perlayer_pca.py:139-143`:

```python
x = diffs_3d.float()                       # (L+1, n_pairs, hidden) → 3-D
pca = PCA(n_components=rank).fit(x)
components = pca.components_               # (L+1, rank, hidden)
mean = pca.mean_[:, 0, :]                  # (L+1, hidden)
direction = components[:, STEER_COMPONENT, :] + mean
```

In all three, the mean term is the CAA mean-difference, whose sign is **definite** — fixed by
the `value − h_value` polarity (`directions_v2.py:296-297`, `perlayer_pca.py:176`,
`DIFF_POLARITY = "value_minus_h_value"` at `directions_v2.py:41`). So you are summing a
sign-definite vector and a sign-arbitrary one. The convention decides whether PC1 adds to or
subtracts from the mean, and that is not a symmetry of the output.

### 3.2 How large the perturbation is

Write $m$ for the mean term with $M = \lVert m \rVert$, and $p$ for PC1 with
$\lVert p \rVert = 1$ (rows of $V$ are orthonormal). The two admissible answers are
$d_\pm = m \pm p$, differing by $2p$:

$$
\langle d_+,\, d_- \rangle \;=\; \lVert m \rVert^2 - \lVert p \rVert^2 \;=\; M^2 - 1,
\qquad
\lVert d_\pm \rVert^2 \;=\; M^2 \pm 2\langle m, p\rangle + 1
$$

$$
\cos(d_+, d_-) \;=\; \frac{M^2 - 1}{\sqrt{\left(M^2 + 2\langle m,p\rangle + 1\right)\left(M^2 - 2\langle m,p\rangle + 1\right)}}
\;\;\approx\;\; \frac{M^2 - 1}{M^2 + 1}
$$

exact when $\langle m, p \rangle = 0$. **The entire effect of the flip is governed by $M$ — the
norm of the mean the unit PC is added to.**

| $M$ | $\cos(d_+, d_-)$ |
|-----|------------------|
| 30  | 0.998 |
| 10  | 0.980 |
| 5   | 0.923 |
| 2   | 0.600 |
| 1   | 0.000 |
| 0.5 | $-0.600$ |

Below $M = 1$ the two conventions give directions pointing to opposite sides.

### 3.3 A verified quirk in `svd_flip`, and why it does not touch the deployed direction

sklearn's `svd_flip` picks `sign(U[argmax_n |U[:, j]|, j])` — a diagonal pick, one column per
component. The port at `pca.py:13` builds a gather index of shape $(B, k, 1)$ and gathers
along dim 1, which evaluates to `U[b, argmax_n |U[b, :, j]|, 0]` — it always reads **column 0**
of `U`. The unused `i = torch.arange(u.shape[2])` at `pca.py:10` is the leftover of sklearn's
`range(u.shape[1])` diagonal index.

I verified this by execution rather than inference, comparing the repo's `svd_flip` against the
sklearn semantics on random batched inputs:

```
b=0
  sklearn : [-1.0, -1.0,  1.0, -1.0,  1.0,  1.0]
  repo    : [-1.0,  1.0,  1.0, -1.0,  1.0,  1.0]
  col0hyp : [-1.0,  1.0,  1.0, -1.0,  1.0,  1.0]
  repo == col0hyp: True   repo == sklearn: False   component 0 agrees: True
```

Consequences, stated narrowly:

- **Component 0 is correct.** For $j = 0$ the column-0 read *is* the diagonal read, so PC1's
  sign follows the intended convention exactly. Since `STEER_COMPONENT = 0` and all three
  textual paths steer on PC1 only, **every deployed direction is unaffected by this quirk.**
- **Components $j \ge 1$ are not.** PC2's sign is keyed to the *PC1* loading of whichever demo
  maximises $\lvert \mathrm{PC2}\rvert$. `rank = 2` is fit in the deployed and per-layer paths
  (`shuffled_control_partition.py:57`), and PC2 is saved — `components` /`components_3d` and
  `pc2_layer_norms` (`directions_v2.py:307`, `perlayer_pca.py:188,196,203,205`). Any PC2
  diagnostic that is sign-sensitive inherits a mis-keyed sign; sign-invariant PC2 quantities
  (norms, $\lvert\cos\rvert$, explained variance) are fine.
- **The `u *= signs` line is harmless.** The broadcast applies component signs along `u`'s
  *sample* axis, corrupting `U` — but `PCA.fit` discards `U` (`pca.py:37-38`), so nothing
  downstream reads it.

### 3.4 The one path that does anchor

`visual_directions.py:51-54`, applied at `:73`:

```python
def _sign_align(pc, mean_diff):
    if torch.dot(pc.flatten(), mean_diff.flatten()) < 0:
        pc = -pc
    return pc
```

The visual per-(layer, token) path forces $\langle \mathrm{PC}, \bar{d}\rangle > 0$ before
reconstruction. The textual global and per-layer paths do not. Two conventions coexist in the
repo, and only one of them is stable under the causes in §5.

---

## 4. Global vs per-layer

### 4.1 Magnitude — the $\sqrt{L{+}1}$ factor

The global paths (`directions.py:104-107`, `directions_v2.py:297-302`) flatten to one
$(L{+}1)\cdot h$ vector and add *one* unit PC1 to the whole thing. From
`perlayer_pca.py:67-70`:

```python
EXPECTED_DECODER_SHAPES = {
    "llava-1.5-7b-hf":        (32, 4096),
    "qwen2.5-vl-7b-instruct": (28, 3584),
}
```

so with the embedding row included, $L{+}1 = 33$ and flat dim $135{,}168$ for LLaVA;
$L{+}1 = 29$ and flat dim $103{,}936$ for Qwen. The relevant norm is

$$
M_{\text{flat}} = \Big(\textstyle\sum_\ell \lVert m_\ell \rVert^2\Big)^{1/2} \approx \sqrt{L{+}1}\;\bar{M}_\ell
$$

and the single unit PC1 is spread across all rows, so each layer's slice of it has norm
$\approx 1/\sqrt{L{+}1}$, competing against a mean slice of norm $M_\ell$.

`perlayer_pca_fit` (`perlayer_pca.py:139-143`) instead gives *every* row its own full unit PC1,
competing against that row's own $M_\ell$. The relative size of the sign perturbation is

$$
\underbrace{\frac{1}{M_\ell}}_{\text{per-layer}}
\qquad\text{vs.}\qquad
\underbrace{\frac{1}{\sqrt{L{+}1}\,M_\ell}}_{\text{global, same row}}
$$

**larger by exactly $\sqrt{L{+}1}$** — $\sqrt{33} \approx 5.74$ on LLaVA, $\sqrt{29} \approx 5.39$
on Qwen.

It is also not uniform across rows. Where $M_\ell$ is small, the unit PC1 is comparable to or
larger than the mean, and by the table in §3.2 the sign choice stops being a small rotation:
$M_\ell < 1 \Rightarrow \cos(d_+, d_-) < 0$ for that row.

Two things compound this:

1. `steer.py` unit-normalizes each layer's direction slice at application time, so magnitude is
   discarded. A row whose direction is dominated by a sign-arbitrary unit PC1 is applied at
   exactly the same strength as a row where the mean dominates. Normalization does not damp the
   flip; it removes the only thing that was making it look small.
2. The mean-dominance figure recorded in `CLAUDE.md` (cosine $\ge 0.999$ with the raw CAA
   mean-difference) was measured on the **global** path. That is a statement about
   $M_{\text{flat}}$ and it does not transfer to the per-layer estimator.

### 4.2 Coherence — one sign decision versus $L{+}1$

This is the structural difference, and it traces back to the `X.ndim` branch at `pca.py:26-30`.

**Global.** `directions.py:106` and `directions_v2.py:301` both stack to a 2-D `(N, flat)`
tensor, so `PCA.fit` unsqueezes to $(1, N, \text{flat})$ and `svd_flip` produces `signs` of
shape $(1, k, 1)$ — **one** sign per component for the entire concatenated vector. The two
admissible outputs are $m + p$ and $m - p$: a single bounded rotation of the whole object,
repairable after the fact by checking $\langle d, m\rangle$ and negating.

**Per-layer.** `perlayer_pca.py:139-141` passes a 3-D `(L+1, n_pairs, hidden)` tensor, so
`svd_flip`'s `argmax` over dim 1 runs *within each row independently* and `signs` has shape
$(L{+}1, k, 1)$ — **$L{+}1$ independent decisions**. The demo pair with the largest loading is
generally a different demo in different layers. The sign configuration lives in

$$
\{-1, +1\}^{\,L+1}
$$

i.e. $2^{33}$ admissible outputs on LLaVA rather than $2$. Two extractions can agree on some
rows and disagree on others, and **no single global negation repairs it.** A ragged per-layer
cosine profile between two otherwise identical extractions is the signature.

Note that the integrity checks in `run_perlayer_cell` do not catch this. `perlayer_pca.py:386-399`
verifies the per-layer `pca_mean_3d` equals the reshaped global `pca_mean_flat` — true, since
the mean is sign-definite and the two fits center identically — and `:400-407` verifies the
reconstruction identity `components[l,0] + mean[l] == direction[l]`. Both hold under any sign
configuration. They check that the per-layer fit is internally consistent, not that its signs
are stable.

---

## 5. Why flips actually occur between two runs

Three distinct causes, worth keeping separate.

**(1) Mathematical.** The sign is not determined by the eigenproblem (§1). Any convention is a
choice, not a recovery of truth.

**(2) The convention is itself statistically unstable.** `pca.py:8` keys the sign on
$\arg\max_n \lvert u_n \rvert$ — the identity of one demo pair. That is a discrete function of
the data. `PARTITION_SIZES = (50, 100, 200, 500)` (`directions_partition.py:44`) with
`PARTITION_SEED = 42` (`:47`) means the sweep re-fits on nested and disjoint demo sets; the
shuffled control re-fits under a derangement with `DERANGEMENT_SEED = 1234`
(`shuffled_control_partition.py:55`). Any of these can move the argmax to a different demo whose
loading has the opposite sign. The underlying quantity did not move; the tiebreak did. This is
why a sign flip is not a rare numerical accident but something that appears systematically in
sample-size sweeps, partition comparisons, and control-vs-deployed comparisons.

**(3) Genuine near-degeneracy.** When $\sigma_1 \approx \sigma_2$, perturbation theory gives the
top eigenvector an error scaling as

$$
\sim \frac{1}{\sigma_1^2 - \sigma_2^2}
$$

so PC1 is not merely sign-unstable, it is *direction*-unstable, rotating freely within a
near-degenerate subspace. At $n \ll h$ — and the largest partition here is $n = 500$ against
$h = 3584$ or $4096$ — with PC1 explaining a small fraction of variance, the spectrum is close
to flat and the identity of "PC1" is barely defined.

Cause (3) makes cause (2) worse: when the spectrum is flat the loadings are near-uniform, and
which demo is the argmax approaches a coin flip.

---

## 6. The sign-invariant reads, and what is already on disk

Comparisons that are immune, and therefore the ones to compute when the question is whether two
extractions found the same thing:

- $\lvert \cos \rvert$ between the **PC components** (not the reconstructed directions);
- the principal angle between spanned subspaces;
- explained-variance ratios;
- $\cos$ between the **mean-difference parts alone**, which carry no ambiguity at all.

Both estimators already write the quantities needed to check, row by row, how much of a layer's
direction is sign-arbitrary:

| Quantity | Global | Per-layer | What it tells you |
|---|---|---|---|
| PC1 explained variance | `directions_v2.py:310` (`pc1_explained_variance`, scalar) | `perlayer_pca.py:194` (`pc1_explained_variance_per_layer`, length $L{+}1$) | how flat the spectrum is → how much cause (3) applies |
| mean-diff norms | `directions_v2.py:315-317` | `perlayer_pca.py:198-200` | $M_\ell$, hence $1/M_\ell$ = size of the sign perturbation against a unit PC1 |
| PC1 layer norms | `directions_v2.py:312` | `perlayer_pca.py:195` | how the unit PC1 distributes across rows (the $1/\sqrt{L{+}1}$ claim in §4.1) |
| raw components | `directions_v2.py:320`–`components` | `perlayer_pca.py:203,205` (`components`, `components_3d`) | permits recomputing $\lvert\cos\rvert$ and $\langle \mathrm{PC1}_\ell, m_\ell\rangle$ post hoc |

The last row matters: because `components_3d` and `pca_mean_3d` are both persisted
(`perlayer_pca.py:205-206`), the sign of every row can be re-derived and re-anchored after the
fact without re-running any forward passes.

---

## Scope note

This note covers the mathematics and what the code in this repo does. It does not say what any
of this implies about the existing per-layer or shuffled-control runs, whether the
global/per-layer contrast on disk is affected, or what should be re-run. Those readings are
Alex's.
