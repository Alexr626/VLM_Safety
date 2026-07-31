# Why $\mathbb{E}[\cos(\hat d_n, \mu)] \approx \sqrt{r_n}$: what the "$\approx$" is hiding

Date: 2026-07-28
Question: "Why is the expected cosine of the estimated direction with the true direction equal to
$\mathbb{E}\langle\hat d_n,\mu\rangle$ divided by $\sqrt{\mathbb{E}\|\hat d_n\|^2}\cdot\|\mu\|$?
Why can the expectation be taken of numerator and denominator separately, and why is the
denominator a square root of an expected square?"

Companion to `cosine_vs_sample_size_mean_difference_directions.md` §2, "Cosine with the truth."
Synthetic numbers only.

---

## 0. Notation

$$a = \|\mu\|, \qquad \hat u = \mu/a, \qquad \varepsilon = \hat d_n - \mu, \qquad \operatorname{Cov}(\varepsilon) = \Sigma/n, \qquad v = \frac{\operatorname{tr}\Sigma}{n} = \mathbb{E}\|\varepsilon\|^2$$

Decompose the error into its component along the true direction and the rest:

$$z = \langle \varepsilon, \hat u\rangle, \qquad \varepsilon_\perp = \varepsilon - z\hat u, \qquad w = \|\varepsilon_\perp\|^2, \qquad \|\varepsilon\|^2 = z^2 + w$$

## 1. Separating the exact steps from the approximate ones

By definition, and pulling out $\|\mu\| = a$ which is a **deterministic constant**:

$$\cos(\hat d_n, \mu) = \frac{\langle \hat d_n, \mu\rangle}{\|\hat d_n\|\,\|\mu\|} = \frac{\langle \hat d_n, \hat u\rangle}{\|\hat d_n\|}$$

That step is exact — $\mu$ is a fixed unknown vector, not a random one, so its norm is a constant
and comes straight out of any expectation. The randomness lives entirely in $\hat d_n$.

Now there are four claims bundled into the displayed line. Two are exact identities and two are
approximations, and it is worth seeing which is which.

**Exact (i): the numerator.** $\langle \hat d_n, \hat u\rangle = \langle \mu + \varepsilon, \hat u\rangle = a + z$, so

$$\mathbb{E}\langle \hat d_n, \hat u\rangle = a + \mathbb{E}[z] = a$$

because $\mathbb{E}[\varepsilon] = 0$. Equivalently $\mathbb{E}\langle \hat d_n, \mu\rangle = a^2$.
No approximation.

**Exact (ii): the expected squared norm.**

$$\mathbb{E}\|\hat d_n\|^2 = \mathbb{E}\|\mu + \varepsilon\|^2 = \|\mu\|^2 + 2\underbrace{\mu^\top \mathbb{E}[\varepsilon]}_{=0} + \mathbb{E}\|\varepsilon\|^2 = a^2 + v$$

Also no approximation. This is bias–variance in its plainest form.

**Approximate (iii): moving the expectation through the ratio.**

$$\mathbb{E}\!\left[\frac{N}{D}\right] \;\approx\; \frac{\mathbb{E}[N]}{\mathbb{E}[D]}$$

with $N = \langle\hat d_n,\hat u\rangle$ and $D = \|\hat d_n\|$. This is **false in general.** It
requires $D$ to concentrate — to be reliably close to its own mean — and §2 is about checking
that it does.

**Approximate (iv): replacing $\mathbb{E}[D]$ with $\sqrt{\mathbb{E}[D^2]}$.**

$$\mathbb{E}\|\hat d_n\| \;\approx\; \sqrt{\mathbb{E}\|\hat d_n\|^2}$$

Also false in general. By Jensen, since $\sqrt{\cdot}$ is concave,
$\mathbb{E}[\sqrt{X}] \le \sqrt{\mathbb{E}[X]}$, with equality only if $X$ is degenerate. So this
step **overstates the denominator**, and therefore understates the cosine. The reason for making
it anyway is purely that $\mathbb{E}\|\hat d_n\|^2$ has the clean closed form in (ii) while
$\mathbb{E}\|\hat d_n\|$ does not.

Note the gap between (iii) and (iv) is $\operatorname{Var}(D)$ — the two approximations are the
same approximation, and they stand or fall together on whether $\|\hat d_n\|$ concentrates.

Granting (iii) and (iv), the arithmetic finishes itself:

$$\mathbb{E}[\cos] \approx \frac{a^2}{\sqrt{a^2+v}\cdot a} = \frac{a}{\sqrt{a^2+v}} = \sqrt{\frac{a^2}{a^2+v}} = \sqrt{r_n}$$

## 2. Why the concentration holds here

The approximations are safe when $z$ is small relative to $a$ and $w$ is small relative to its own
mean. Take the isotropic case, $\Sigma = \sigma^2 I$, and write $\tau^2 = \sigma^2/n$, so
$v = d\tau^2$:

$$z \sim \mathcal{N}(0, \tau^2) = \mathcal{N}(0,\, v/d), \qquad w \sim \tau^2\chi^2_{d-1} \;\;\Rightarrow\;\; \mathbb{E}[w] = v\tfrac{d-1}{d},\;\; \operatorname{sd}(w) \approx v\sqrt{2/d}$$

Running example: $d = 300$, $a = 1$, $v = 0.30$.

| | typical size | relative to what it is compared against |
|---|---|---|
| $z$ | $\operatorname{sd} = \sqrt{0.30/300} = 0.0316$ | $\pm 3.2\%$ of $a = 1$ |
| $w$ | $\operatorname{sd} = 0.30\sqrt{2/300} = 0.0245$ | $\pm 8.2\%$ of $\mathbb{E}[w] \approx 0.30$ |

Both random inputs sit within a few percent of their means, and $\cos$ is a smooth function of
them. Plugging in means is then accurate to second order in those small fluctuations. **The
justification is a concentration argument, and what makes it work is that $d$ is large** — not
anything about $n$, and not anything about the estimator.

### 2.1 The size of the error

Expand $f(u, w) = u/\sqrt{u^2+w}$ to second order about $(u, w) = (a, \mathbb{E}[w])$. The cross
term vanishes because $z \perp \varepsilon_\perp$ for Gaussian isotropic noise. Using
$1 - r = v/(a^2+v)$, the three corrections in units of $\sqrt{r}$ are

$$\underbrace{\frac{1-r}{2d}}_{\mathbb{E}[w] \ne v} \;\underbrace{-\;\frac{3(1-r)^2}{2d}}_{\operatorname{Var}(z)} \;\underbrace{+\;\frac{3(1-r)^2}{4d}}_{\operatorname{Var}(w)}
\;=\; \frac{1-r}{d}\left[\frac12 - \frac{3(1-r)}{4}\right]$$

So the **relative** error of $\sqrt{r_n}$ is $O(1/d)$, and maximising the bracket over
$r \in [0,1]$ bounds it:

$$\left|\frac{\mathbb{E}[\cos]}{\sqrt{r_n}} - 1\right| \;\lesssim\; \frac{1}{4d}$$

At $d = 300$ that is under $0.09\%$, essentially regardless of $r$. For the running example
($r = 0.769$): relative error $+2.5\times10^{-4}$, so

$$\mathbb{E}[\cos] \approx 0.87706 + 0.00022 = 0.87728 \qquad \text{versus} \qquad \sqrt{r} = 0.87706$$

Compare that to $\operatorname{sd}[\cos] \approx 0.0105$ from the companion note. **The
approximation error is roughly fifty times smaller than the sampling noise of the very quantity
it approximates.** Correcting it would be pointless; you cannot measure the difference.

(The sign flips at $r = 1/3$: above it the formula slightly understates $\mathbb{E}[\cos]$,
below it slightly overstates. This is a second-order expansion, so trust it in the middle of the
range and simulate at the extremes.)

For completeness, an exact answer does exist in the isotropic Gaussian case — with
$U = (a+z)/\tau$ and $W = w/\tau^2$, the quantity $t = U/\sqrt{W/(d-1)}$ is noncentral $t$ with
$d-1$ degrees of freedom and noncentrality $a/\tau$, and $\cos = t/\sqrt{t^2 + (d-1)}$. It is not
worth using. The approximation plus a simulation check is strictly more practical.

## 3. The geometric picture, which makes all of §2 obvious

This is the part to keep; the algebra above is just its bookkeeping.

**How much of the noise lands along $\mu$?** In the isotropic case,

$$\frac{\mathbb{E}[z^2]}{\mathbb{E}\|\varepsilon\|^2} = \frac{\tau^2}{d\tau^2} = \frac{1}{d}$$

At $d = 300$, one third of one percent. A random direction in high dimension is nearly orthogonal
to any fixed direction, so **essentially all of the estimation error is perpendicular to the
thing you are estimating.** So, to the accuracy established in §2:

$$\hat d_n \;\approx\; \underbrace{\mu}_{\text{length } a,\ \text{along } \hat u} \;+\; \underbrace{\varepsilon_\perp}_{\text{length } \sqrt v,\ \perp\, \hat u}$$

Two perpendicular legs. That is a right triangle, and $\|\hat d_n\| = \sqrt{a^2+v}$ is the
hypotenuse by Pythagoras — which is exactly the exact identity (ii), now with a picture attached.
The angle between $\hat d_n$ and $\mu$ is the angle at the origin, so

$$\cos\theta = \frac{\text{adjacent}}{\text{hypotenuse}} = \frac{a}{\sqrt{a^2+v}} = \sqrt{r_n}$$

Draw it: horizontal leg of length $1$ for the signal, vertical leg of length $\sqrt{0.30} = 0.55$
for the noise at $n=50$, hypotenuse $1.14$, angle $\approx 28.7°$. At $n = 500$ the vertical leg
shrinks to $0.17$ and the angle drops to $\approx 9.9°$. Same horizontal leg in both. The
estimator does not rotate; the noise leg shortens as $1/\sqrt n$.

### 3.1 Why reliability is a cosine *squared*

$$r_n = \frac{a^2}{a^2+v} = \frac{\text{adjacent}^2}{\text{hypotenuse}^2} = \cos^2\theta$$

So $r$ is a ratio of *squared lengths* — a fraction of energy — while $\cos\theta$ is a ratio of
lengths. That is precisely the $R^2$-versus-correlation relationship, and it is why $r$ behaves
like a variance-explained quantity and $\sqrt{r}$ behaves like a correlation. Once you see that
$r$ is the squared cosine, "alignment with truth is $\sqrt{r}$" stops being a fact to memorise.

### 3.2 One factor of $\sqrt{r}$ per noisy vector

Now take two independent estimates, at possibly different sample sizes:

$$\hat d^{(1)} \approx \mu + \varepsilon^{(1)}_\perp, \qquad \hat d^{(2)} \approx \mu + \varepsilon^{(2)}_\perp$$

The two perpendicular errors are independent random vectors in a $(d-1)$-dimensional space, so
they are themselves nearly orthogonal to each other — their inner product contributes about
$\sqrt{v_1 v_2 / d}$, negligible against $a^2$. Hence

$$\langle \hat d^{(1)}, \hat d^{(2)}\rangle \approx a^2, \qquad
\cos\big(\hat d^{(1)}, \hat d^{(2)}\big) \approx \frac{a^2}{\sqrt{a^2+v_1}\sqrt{a^2+v_2}}
= \underbrace{\frac{a}{\sqrt{a^2+v_1}}}_{\sqrt{r_1}}\cdot\underbrace{\frac{a}{\sqrt{a^2+v_2}}}_{\sqrt{r_2}}$$

**Every noisy vector in the comparison contributes exactly one factor of $\sqrt{r}$ — its own
$\cos\theta$ — and an exactly-known vector contributes a factor of $1$.** All three results in
§2 of the companion note are this one rule:

| Comparison | Noisy vectors | Result |
|---|---|---|
| $\cos(\hat d_n, \mu)$ | one | $\sqrt{r_n}$ |
| $\cos(\hat d_n, \hat d_n')$ | two, same $n$ | $\sqrt{r_n}\sqrt{r_n} = r_n$ |
| $\cos(\hat d_{50}, \hat d_{500})$ | two, different $n$ | $\sqrt{r_{50}}\sqrt{r_{500}}$ |

And the general case: if the two targets themselves differ, the picture tilts the shared leg by
$\cos(\mu_A,\mu_B)$ and you get $\cos(\mu_A,\mu_B)\sqrt{r_A r_B}$ — the disattenuation formula.

## 4. When the approximation stops being safe

The concentration argument used three things. Each one can fail.

**1. $d$ small, or $d_{\text{eff}}$ small.** The error bound was $O(1/d)$, so at $d_{\text{eff}} = 3$
the relative error is percent-scale rather than $10^{-4}$-scale. Substituting $d_{\text{eff}}$
for $d$ is a heuristic, not an identity, but the direction is right: **strongly anisotropic noise
degrades this approximation, exactly as it degrades everything else built on it.**

**2. $\mu$ aligned with the noise spectrum.** This one is not captured by any scalar summary and
is worth stating separately. In general

$$\operatorname{Var}(z) = \hat u^\top \frac{\Sigma}{n} \hat u$$

which is $v/d$ only under isotropy. If $\mu$ happens to lie along a top eigenvector of $\Sigma$,
$\operatorname{Var}(z)$ can be a large fraction of $v$, the "all the noise is perpendicular"
premise of §3 fails, and the right-triangle picture stops being a good approximation. If $\mu$
lies in a low-variance direction, the opposite. Neither $\operatorname{tr}\Sigma$ nor
$d_{\text{eff}}$ knows where $\mu$ points relative to the eigenbasis — you have to check
$\hat u^\top \Sigma \hat u$ directly. For directions extracted from representations that have a
large shared mean offset, this is a live concern rather than a technicality.

**3. Non-Gaussian or heavy-tailed $\varepsilon$.** The $\chi^2$ concentration of $w$ was
Gaussian-specific. Heavy tails widen $\operatorname{Var}(w)$ and the expansion degrades.

In all three cases the fix is the same and is cheap: simulate. Draw from your estimated $\Sigma$,
compute the empirical distribution of $\cos$, and compare its mean to $\sqrt{r}$. If they agree
to well within the empirical sd, the approximation is doing no harm.

## Related

- `answers/concepts/july_28_2026/cosine_vs_sample_size_mean_difference_directions.md` — §2 is
  what this note unpacks; §5 has the variance of the cosine.
- `answers/concepts/july_28_2026/sample_mean_covariance_and_trace.md` — where $v = \operatorname{tr}\Sigma/n$
  comes from.
- `answers/concepts/july_28_2026/noise_floor_and_disattenuation.md` — the same approximation in
  its general form, and $d_{\text{eff}}$.
