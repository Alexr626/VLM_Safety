# Comparing cosine similarities across estimators with different sampling variance

Date: 2026-07-28
Question: "How do I compare a similarity measurement between two estimators that have
different sampling variance, and what does a noise floor have to be matched on for the
comparison to mean anything?"

Scope: general method note. No project results are used or referenced here; the worked example
is synthetic. Nothing in this note depends on a fact read from the codebase, so there are no
`file:line` citations.

---

## 1. The core problem

A cosine between two *estimated* directions is a reliability-attenuated version of the cosine
between the two *target* directions. A noisier estimator is pulled harder toward the null. So a
raw cosine comparison across estimators with different sampling variance measures the variance
difference, not the target difference.

Every fix below follows from that one sentence.

## 2. Setup and derivation of the attenuation

Let the estimator be

$$\hat\mu = \mu + \varepsilon, \qquad \mathbb{E}[\varepsilon] = 0, \qquad \operatorname{Cov}(\varepsilon) = \frac{\Sigma}{n}$$

where $\mu \in \mathbb{R}^d$ is the target direction, $n$ is the number of contrast pairs, and
$\Sigma$ is the per-pair covariance of whatever quantity is being averaged. This is exact for a
difference of sample means and holds to leading order for any $\sqrt n$-consistent estimator,
with a different $\Sigma$ per estimator — that difference in $\Sigma$ is precisely what makes
the naive comparison invalid.

Take two estimates $\hat\mu_1, \hat\mu_2$ built from **disjoint** halves, so
$\varepsilon_1 \perp \varepsilon_2$. Then

$$\mathbb{E}\langle \hat\mu_1, \hat\mu_2\rangle
= \|\mu\|^2 + \underbrace{\mu^\top(\mathbb{E}\varepsilon_1 + \mathbb{E}\varepsilon_2)}_{=0}
+ \underbrace{\mathbb{E}[\varepsilon_1^\top \varepsilon_2]}_{=0}
= \|\mu\|^2$$

$$\mathbb{E}\|\hat\mu\|^2 = \|\mu\|^2 + \mathbb{E}\|\varepsilon\|^2 = \|\mu\|^2 + \frac{\operatorname{tr}\Sigma}{n}$$

Approximating the expectation of the ratio by the ratio of expectations (see §2.1 for when this
is safe):

$$\boxed{\;\mathbb{E}\big[\cos(\hat\mu_1,\hat\mu_2)\big] \;\approx\; \frac{\|\mu\|^2}{\|\mu\|^2 + \operatorname{tr}\Sigma / n} \;=\; \frac{S}{1+S}, \qquad S \;\equiv\; \frac{n\,\|\mu\|^2}{\operatorname{tr}\Sigma}\;}$$

Read what this says. The split-half cosine depends on the estimator **only through the scalar
$S$** — signal energy over total noise energy in the estimate. Two estimators that target the
identical direction return different split-half cosines at the same $n$ whenever their $\Sigma$
or their $\|\mu\|$ differ. The split-half cosine is not a property of the target; it is a
reliability coefficient of the estimator.

Two further readings worth keeping:

- $\operatorname{tr}\Sigma$, not $\Sigma$, sets the attenuation. Only the *total* noise energy
  matters, not how it is distributed across directions. That is why a nominally huge $d$ is not
  automatically fatal: if the noise is concentrated in a few eigendirections,
  $\operatorname{tr}\Sigma$ can be modest.
- $S$ scales linearly in $n$, so reliability improves with $n$ but saturates: going from
  $r = 0.9$ to $r = 0.95$ costs as much data as going from $0$ to $0.9$.

### 2.1 When the ratio-of-expectations step is safe

The step ignores fluctuations in $\|\hat\mu_1\|\,\|\hat\mu_2\|$. That is justified when the norm
concentrates, which happens when the noise energy is spread over many eigendirections. The
relevant quantity is the participation ratio

$$d_{\text{eff}} = \frac{(\operatorname{tr}\Sigma)^2}{\operatorname{tr}(\Sigma^2)}$$

with $\operatorname{Var}(\|\varepsilon\|^2) \asymp (\operatorname{tr}\Sigma)^2 / d_{\text{eff}}$
for Gaussian noise. Large $d_{\text{eff}}$: the formula is accurate. Small $d_{\text{eff}}$ — one
or two eigenvalues carrying most of $\operatorname{tr}\Sigma$, which is the normal situation for
transformer residual streams — the norm fluctuates, and the true $\mathbb{E}[\cos]$ sits below
the formula.

**Consequence:** use the closed form to reason about scaling and to sanity-check magnitudes. Do
not use it to produce the number you report. Get the reported number by empirical resampling
(§9), which is free of this approximation.

## 3. Split-half reliability and Spearman–Brown

Define the reliability of an estimator at sample size $n$ as $r_n = S_n/(1+S_n)$ — exactly the
quantity above. To measure it, split into disjoint halves, giving $r_{n/2}$, then extrapolate.
Under this model the Spearman–Brown formula is not an analogy, it is exact. Doubling $n$ doubles
$S$, so

$$r_{2n} = \frac{2S_n}{1 + 2S_n}
= \frac{2\left(\frac{r_n}{1-r_n}\right)}{1 + 2\left(\frac{r_n}{1-r_n}\right)}
= \frac{2 r_n}{1 + r_n}$$

using $S = r/(1-r)$. So a measured half-split cosine extrapolates to the reliability of the
full-sample direction you actually deploy:

$$r_{\text{full}} = \frac{2\,r_{\text{half-split}}}{1 + r_{\text{half-split}}}$$

The general-$k$ version, from $S \propto n$: $\;r_{kn} = k r_n / (1 + (k-1) r_n)$.

This is the step people skip. A half-split cosine underestimates the reliability of the deployed
direction, because the deployed direction used twice the data.

## 4. Disattenuation: the comparison you actually want

For two estimators with **independent** errors, targeting $\mu_A$ and $\mu_B$:

$$\mathbb{E}\big[\cos(\hat\mu_A, \hat\mu_B)\big] \;\approx\; \cos(\mu_A, \mu_B) \cdot \sqrt{r_A \, r_B}$$

Same derivation: the numerator expectation is $\mu_A^\top\mu_B$, each denominator norm inflates
by its own noise term. Two consequences.

**$\sqrt{r_A r_B}$ is a ceiling.** It is the largest cosine you could observe even if the two
estimators targeted precisely the same direction. Always report the observed cosine next to its
ceiling. An observed $0.60$ against a ceiling of $0.62$ and an observed $0.60$ against a ceiling
of $0.95$ are opposite results, and the raw number cannot distinguish them.

**The comparable quantity is the disattenuated cosine:**

$$\hat\rho = \frac{\cos_{\text{obs}}(\hat\mu_A, \hat\mu_B)}{\sqrt{r_A \, r_B}}$$

Caveats that matter in practice:

- $\hat\rho$ is a ratio of noisy quantities. It is unstable when either $r$ is small, and it can
  land above $1$ — that is sampling noise, not evidence, and it should not be clipped silently.
- Bootstrap it **over contrast pairs**, not over vector coordinates. Coordinates are not the
  sampling unit and resampling them estimates the wrong variance.
- Report $\cos_{\text{obs}}$, the ceiling, and $\hat\rho$ together. $\hat\rho$ alone hides how
  much of it was extrapolation.

## 5. Correlated errors break disattenuation

Everything above assumes $\mathbb{E}[\varepsilon_A^\top \varepsilon_B] = 0$. If both estimators
are computed from the **same** sample, that term does not vanish:

$$\mathbb{E}\langle \hat\mu_A, \hat\mu_B\rangle = \mu_A^\top\mu_B + \mathbb{E}[\varepsilon_A^\top\varepsilon_B]$$

and the observed cosine is inflated by shared noise. It is not comparable to a split-half
reliability, and dividing by $\sqrt{r_A r_B}$ does not correct it — the correction assumes the
bias it is failing to remove.

This is at its worst exactly where it is easiest to miss: when the two estimators are
algebraically close, so that a large part of both the signal and the noise is literally the same
arithmetic on the same numbers. A cosine near $1$ between two same-sample estimators is
consistent with "the same target" and equally consistent with "the same noise, plus shared
algebra," and the measurement as constructed cannot separate them.

**Fix: cross-fit.** Compute $\hat\mu_A$ on fold 1 and $\hat\mu_B$ on fold 2, then swap, then
average the two cosines. Errors are now independent by construction, and the result is
comparable to the reliabilities — which must themselves be measured at the matching $n$ (fold
size), then Spearman–Brown'd up if you want the full-sample statement.

## 6. What a noise floor has to be matched on

A noise floor answers: *what does this measurement produce when the signal is removed and
nothing else is?* Every property of the pipeline that affects the noise geometry must survive
into the control. Concretely:

**1. $n$, and the effective $n$.** Noise scales as $1/n$, so the control must use the same
number of pairs. Beyond that, if pairs share images, prompts, objects, or scenes, the errors are
correlated and the effective $n$ is smaller than the nominal count. The control must inherit the
same clustering — permute or derange *within* the same grouping structure. A control that
destroys the clustering along with the signal produces a floor that is too low, and everything
looks significant against it.

**2. The estimator's functional form.** Mean-difference, PC1, and LDA have different null
distributions, and there is no shared floor across them. Mean-difference of pure noise is roughly
isotropic. PC1 of pure noise is not: under a Marchenko–Pastur spectrum with a
Baik–Ben Arous–Péché-type threshold, the top sample eigenvector has systematic structure and a
systematically non-null overlap behaviour that depends on $d/n$. Below the BBP threshold the
sample PC1 carries essentially no information about the population one, and above it the overlap
rises sharply — so a PC1 floor is a strongly nonlinear function of $d/n$ where a mean-difference
floor is not. Compute one floor per estimator family.

**3. The covariance $\Sigma$ of the underlying activations.** This is the item most often broken.
Transformer residual streams are severely anisotropic — a handful of directions carry most of
$\operatorname{tr}\Sigma$, and there is usually a large common mean offset shared by every token.
Under such $\Sigma$, two *signal-free* estimates already agree far more than isotropic intuition
predicts, because both are dominated by the same few high-variance directions. An
isotropic-Gaussian control therefore produces a floor that is much too low.

This is the structural reason label-permutation and derangement controls are the right shape:
they reuse the actual activations and destroy only the association between condition label and
activation, so $\Sigma$ — including the anisotropy and the common offset — is preserved by
construction rather than by assumption.

**4. The full post-processing pipeline.** Per-layer unit normalization, layer slicing, token
selection and pooling, centering, concatenation across layers — each changes the effective
weighting of coordinates in the final cosine. The control has to be built from identically
processed objects. If the deployed object is per-layer normalized and then concatenated, a
control that normalizes only after concatenation weights the layers differently and its floor
describes a different measurement.

**5. The ambient dimension $d$ of the object being compared.** For isotropic vectors the null
cosine concentrates near $0$ with standard deviation $\approx 1/\sqrt{d}$. A single-layer cosine
and a 32-layer-concatenated cosine therefore have floors of very different width. Fix $d$ across
the arms before comparing anything.

**6. Sign convention.** For eigenvector-type estimators the sign is arbitrary. Align signs — or
take $|\cos|$ — identically in the real and control arms. $|\cos|$ has a strictly positive null
mean ($\mathbb{E}|\cos| \approx \sqrt{2/(\pi d)}$ in the isotropic case), so mixing conventions
across arms manufactures an effect.

If any of items 1–5 differ between the two arms, the difference in observed cosine is already
explained by the mismatch, before signal enters the discussion.

## 7. The floor is a distribution, and there are two of them

A scalar floor is not usable, because the question is whether an observed cosine lies outside the
null, and that needs the null's spread. Generate $B$ permutations or derangements — $B \geq 200$
for a stable 5th/95th percentile, more if you want a tail — and report quantiles.

Two distinct floors, answering different questions, not interchangeable:

| Floor | Statistic | Question it answers |
|---|---|---|
| Reliability floor | $\cos(\text{ctrl}_{\text{half}_1}, \text{ctrl}_{\text{half}_2})$ | What split-half cosine does a signal-free estimator produce at this $n$, $\Sigma$, $d$? |
| Alignment floor | $\cos(\text{ctrl}, \hat\mu_{\text{real}})$ | What does a signal-free direction score against the real one? |

The reliability floor is what you compare a split-half reliability against. The alignment floor
is what you compare a cross-estimator or cross-condition cosine against. Using one where the
other belongs is a common and quiet error.

## 8. Worked synthetic example

Isotropic noise for arithmetic transparency; the real anisotropic case only makes the point
sharper. Take $d = 3584$, $\Sigma = \sigma^2 I$ with $\sigma^2 = 1$, and $\|\mu\| = 1.2$.

At $n = 200$:

$$\frac{\operatorname{tr}\Sigma}{n} = \frac{3584}{200} = 17.92, \qquad
S = \frac{1.44}{17.92} = 0.0804, \qquad r_{200} = \frac{0.0804}{1.0804} = 0.074$$

Check the Spearman–Brown consistency. Halves of $n = 100$:
$\operatorname{tr}\Sigma / 100 = 35.84$, $S = 0.0402$, $r_{100} = 0.0386$. Extrapolating,
$2(0.0386)/(1.0386) = 0.0743$. Matches.

Now suppose estimator $B$ is better conditioned and has $r_B = 0.30$. Ceiling:

$$\sqrt{r_A r_B} = \sqrt{0.074 \times 0.30} = 0.149$$

So if the two estimators target *the identical direction*, the expected observed cosine is
$0.149$. Suppose you measure $0.14$. The naive reading — "$0.14$ is small, and only about eight
times the isotropic floor of $1/\sqrt{3584} = 0.017$, so these are largely different directions"
— is wrong. Disattenuated:

$$\hat\rho = \frac{0.14}{0.149} = 0.94$$

The estimators are targeting nearly the same direction. The small raw cosine was almost entirely
attenuation.

The mirror-image trap, same setup. Compare the same estimator at two sample sizes: $n = 200$
($r = 0.074$) and $n = 800$ ($\operatorname{tr}\Sigma/n = 4.48$, $S = 0.321$, $r = 0.243$), each
scored against a common reference $\hat\mu_X$. The ratio of expected observed cosines is

$$\frac{\cos(\hat\mu_B, \hat\mu_X)}{\cos(\hat\mu_A, \hat\mu_X)} \approx \sqrt{\frac{r_B}{r_A}} = \sqrt{\frac{0.243}{0.074}} = 1.81$$

An $81\%$ larger cosine, from an estimator with the identical target, produced entirely by having
four times the data. Any narrative built on the raw gap is a narrative about $n$.

## 9. Recipe

1. **Reliability.** For each estimator, at each $n$ of interest: draw $B$ disjoint half-splits,
   compute the split-half cosine, take the distribution of $r_{n/2}$, Spearman–Brown to $r_n$.
2. **Floors.** Same loop, same $n$, same estimator, same preprocessing, association destroyed by
   permutation or derangement within the grouping structure. Produce both floors from §7 as
   distributions.
3. **Cross-estimator cosine.** Cross-fit (§5) so the errors are independent. Disattenuate by
   $\sqrt{r_A r_B}$. Bootstrap over pairs for the interval.
4. **Never compare two point cosines at different $n$.** Either subsample the larger arm down to
   the smaller, or — better — plot $r$ against $n$ for both estimators and compare the curves.
   A single-$n$ comparison confounds estimator quality with how much data each arm received; the
   curve separates them, and the asymptote is the quantity of interest.
5. **Report the matched quantities explicitly:** $n$, effective $n$, $d$, estimator family,
   normalization pipeline, $B$, which floor, and the sign convention. A cosine without these is
   not interpretable by anyone, including you in three months.

### Sketch

Illustrative, not repo code. `estimator(idx)` returns an unnormalized direction from the pairs
at those indices; `control_estimator(idx, rng)` is the same function with the labels deranged
inside the grouping structure.

```python
import numpy as np

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def spearman_brown(r_half, k=2):
    return k * r_half / (1 + (k - 1) * r_half)

def split_half_reliability(estimator, n_pairs, B, rng, groups=None):
    """Distribution of full-n reliability via disjoint half-splits."""
    out = []
    for _ in range(B):
        h1, h2 = disjoint_halves(n_pairs, rng, groups)   # respect clustering
        out.append(cos(estimator(h1), estimator(h2)))
    return spearman_brown(np.array(out))

def cross_fit_cosine(est_a, est_b, n_pairs, B, rng, groups=None):
    """Independent-error cosine between two estimators on the same data."""
    out = []
    for _ in range(B):
        f1, f2 = disjoint_halves(n_pairs, rng, groups)
        out.append(0.5 * (cos(est_a(f1), est_b(f2)) + cos(est_a(f2), est_b(f1))))
    return np.array(out)

# floors: identical calls, control_estimator in place of estimator
r_a  = split_half_reliability(est_a, n, B, rng, groups)
r_b  = split_half_reliability(est_b, n, B, rng, groups)
obs  = cross_fit_cosine(est_a, est_b, n, B, rng, groups)
ceil = np.sqrt(r_a.mean() * r_b.mean())

print(f"observed {obs.mean():.3f}  ceiling {ceil:.3f}  disattenuated {obs.mean()/ceil:.3f}")
print(f"floor (alignment) p5-p95: {np.percentile(floor_align, [5, 95])}")
```

For PC1-type estimators, `cos` must be sign-aligned or replaced by `abs(cos)`, applied
identically in the real and control arms.

## 10. Failure modes, condensed

- Two point cosines at different $n$, compared to each other. Measures $n$.
- A floor from isotropic Gaussian noise instead of a permutation of the real activations.
  Underestimates the floor, because it discards the anisotropy that makes signal-free estimates
  agree.
- One floor reused across estimator families. PC1 and mean-difference have different nulls.
- A scalar floor with no spread. Cannot support a claim of exceeding it.
- Cosine between two same-sample estimators read as evidence about targets. Shared noise.
- Half-split reliability reported as the reliability of the deployed direction. Off by
  Spearman–Brown.
- Sign convention differing between real and control arms. Manufactures an effect.
- Disattenuation reported without the ceiling and without an interval. Hides the extrapolation.

## Related

- `answers/concepts/jun_19_2026/vti_rotation_strength_motivation.md`
- `STEERING_MATH_REFERENCE.md` §4 (geometry of the steering operation)
