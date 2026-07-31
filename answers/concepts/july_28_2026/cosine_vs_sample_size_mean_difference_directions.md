# The same concept direction estimated at n=50 and n=500: what the cosine actually measures

Date: 2026-07-28
Question: "What is the difference between measuring the cosine similarity between a vector of
differences between two similar sets of data — say word embeddings rather than activations —
that differ only in a specific concept, but that are estimated using different sample sizes,
say 50 vs 500?"

Scope: concept note. All numbers are synthetic and invented for the example. No project results
are read or referenced.

---

## 1. The setup, made concrete

Take a word-embedding gender direction, the standard textbook case. You have pairs
$(w_i^A, w_i^B)$ — `(man, woman)`, `(king, queen)`, `(actor, actress)` — and embeddings
$e(\cdot) \in \mathbb{R}^{300}$. The per-pair difference is

$$x_i = e(w_i^A) - e(w_i^B)$$

and the estimator is the mean difference

$$\hat d_n = \frac{1}{n}\sum_{i=1}^{n} x_i$$

The generative model that makes the question well-posed:

$$x_i = \mu + \varepsilon_i, \qquad \mathbb{E}[\varepsilon_i] = 0, \qquad \operatorname{Cov}(\varepsilon_i) = \Sigma$$

$\mu$ is the concept direction you want. $\varepsilon_i$ is everything *else* that distinguishes
the two words of pair $i$ — `king`/`queen` also differ in which monarchies the corpus discusses,
`actor`/`actress` also differ in corpus frequency and register. So

$$\operatorname{Cov}(\hat d_n) = \frac{\Sigma}{n}$$

Note what $\Sigma$ is here. It is the **between-pair scatter of the difference vectors**, not the
covariance of the embedding space. Those are different objects and conflating them is the most
common setup error. Section 7 is about what happens when the "everything else" is not
zero-mean — that case behaves completely differently and is the one that actually bites.

## 2. Three cosines, and the $\sqrt{r}$ rule

Everything follows from two expectations. Write $v_n = \operatorname{tr}\Sigma / n$ for the total
noise energy in the estimate, and $a = \|\mu\|$:

$$\mathbb{E}\|\hat d_n\|^2 = a^2 + v_n, \qquad \mathbb{E}\langle \hat d_n, \hat d_{n'}\rangle = a^2 \;\; \text{(disjoint samples)}$$

Define the reliability $r_n = a^2/(a^2 + v_n) = S_n/(1+S_n)$ with $S_n = n a^2/\operatorname{tr}\Sigma$.

**Cosine with the truth.** This is the one people do not compute and should:

$$\mathbb{E}\big[\cos(\hat d_n, \mu)\big] \;\approx\; \frac{\mathbb{E}\langle\hat d_n,\mu\rangle}{\sqrt{\mathbb{E}\|\hat d_n\|^2}\,\|\mu\|}
= \frac{a^2}{\sqrt{a^2+v_n}\;\cdot a} = \frac{a}{\sqrt{a^2+v_n}} = \sqrt{r_n}$$

**Cosine between two independent estimates at the same $n$.**

$$\mathbb{E}\big[\cos(\hat d_n, \hat d_n')\big] \approx \frac{a^2}{a^2 + v_n} = r_n$$

**Cosine between estimates at different $n$, same target.**

$$\boxed{\;\mathbb{E}\big[\cos(\hat d_{50}, \hat d_{500})\big] \;\approx\; \frac{a^2}{\sqrt{a^2+v_{50}}\sqrt{a^2+v_{500}}} = \sqrt{r_{50}}\,\sqrt{r_{500}}\;}$$

The general statement, for estimators with independent errors and possibly different targets:

$$\mathbb{E}[\cos(\hat d_A, \hat d_B)] \approx \cos(\mu_A,\mu_B)\sqrt{r_A r_B}$$

**The relation to hold onto: alignment with truth is the square root of the two-replicate
cosine.** A split-half cosine of $0.64$ does not mean your direction is 64% of the way to
correct; it means each half aligns with the truth at $\sqrt{0.64} = 0.80$. Reading a split-half
cosine as "how good is my direction" understates it, systematically, and by a lot in the range
where these numbers usually land. This is the psychometric validity ceiling
$\rho_{XY} \le \sqrt{r_{XX}}$, in geometric clothing.

## 3. Worked numbers

Invented but plausible. $d = 300$, $\|\mu\| = 1.0$, isotropic $\Sigma = \sigma^2 I$ with
$\operatorname{tr}\Sigma = 15$. That means a single pair difference has noise norm
$\sqrt{15} \approx 3.9$ against signal norm $1.0$ — any individual pair is dominated by
idiosyncrasy, which is realistic for word pairs.

| | $v_n = 15/n$ | $S_n$ | $r_n$ | $\sqrt{r_n}=\cos$ to truth |
|---|---|---|---|---|
| $n=50$ | $0.300$ | $3.33$ | $0.769$ | $0.877$ |
| $n=500$ | $0.030$ | $33.3$ | $0.971$ | $0.985$ |

So:

- $\mathbb{E}\cos(\hat d_{50}, \hat d_{500}) = 0.877 \times 0.985 = \mathbf{0.864}$
- $\mathbb{E}\cos(\hat d_{50}, \hat d_{50}') = r_{50} = 0.769$
- $\mathbb{E}\cos(\hat d_{500}, \hat d_{500}') = r_{500} = 0.971$

**Both estimators target the identical direction.** The entire gap from $1.0$ is sampling noise.
If you had reported "the 50-pair and 500-pair gender directions agree at only $0.86$, so the
small sample is picking up something different," you would have described $n$ and called it a
finding.

## 4. What differs between the two estimates, and what does not

**Does not differ: the expected vector.** $\mathbb{E}[\hat d_{50}] = \mathbb{E}[\hat d_{500}] = \mu$.
The mean-difference estimator is unbiased *as a vector* at every $n$. There is no systematic
rotation, no drift, no direction-dependent distortion. Averaged over replicates, the two point
the same way.

**Differs: the scatter,** by a factor of $10$ in variance, $\sqrt{10}$ in typical angular error.

**Differs: the norm, systematically.** $\mathbb{E}\|\hat d_n\|^2 = a^2 + v_n$, so

$$\|\hat d_{50}\| \approx \sqrt{1.30} = 1.140, \qquad \|\hat d_{500}\| \approx \sqrt{1.03} = 1.015$$

The small-$n$ estimate is about $12\%$ **longer**, and every bit of the excess is noise energy.
Noise adds in quadrature and cannot subtract. So any pipeline that reads the raw magnitude of an
extracted direction as "how strong the concept is" will report small samples as stronger
concepts. Where the direction is unit-normalized before use, this term is discarded and does not
propagate — but it does mean raw norms are not comparable across $n$, ever.

**The subtle one: the vector is unbiased, the cosine is not.** $\cos$ is a nonlinear function of
$\hat d_n$, and Jensen bites in a fixed direction: $\mathbb{E}[\cos(\hat d_n,\mu)] = \sqrt{r_n} < 1$
for every finite $n$. The bias is always **toward the null, never away from it.** Unbiased
vector, biased cosine — these coexist and the second is what you measure.

## 5. The error is bias, not variance — which is why averaging will not save you

Decompose $\varepsilon = z\hat u + \varepsilon_\perp$ with $\hat u = \mu/\|\mu\|$, so
$z \sim \mathcal{N}(0, v/d)$ and $\|\varepsilon_\perp\|^2 \sim (v/d)\chi^2_{d-1}$. Then
$\cos = (a+z)/\sqrt{(a+z)^2 + w}$ with $w = \|\varepsilon_\perp\|^2$. Delta method about
$(z, w) = (0, v)$:

$$\frac{\partial \cos}{\partial z} = \frac{v}{(a^2+v)^{3/2}}, \qquad \frac{\partial \cos}{\partial w} = -\frac{a}{2(a^2+v)^{3/2}}$$

with $\operatorname{Var}(z) = v/d$ and $\operatorname{Var}(w) \approx 2v^2/d$. Combining:

$$\operatorname{Var}\big[\cos(\hat d_n,\mu)\big] \;\approx\; \frac{v^2}{d}\cdot\frac{v + a^2/2}{(a^2+v)^3}$$

For the example: $n=50$ gives $\text{sd} \approx 0.0105$; $n=500$ gives $\text{sd} \approx 0.0012$.

$$\cos(\hat d_{50},\mu) = 0.877 \pm 0.010, \qquad \cos(\hat d_{500},\mu) = 0.985 \pm 0.001$$

Read the scaling: **the standard deviation is $O(1/\sqrt{d})$ while the attenuation is $O(1)$ in
$d$.** Their ratio grows like $\sqrt d$. In $d=300$ the bias is roughly twelve standard
deviations. So in high dimension a single measured cosine is a *precise* measurement of a
*systematically wrong* quantity. Repeating the measurement, averaging over seeds, or adding
error bars does nothing — none of those touch a bias. Only more data per estimate, or an explicit
disattenuation, moves it.

This is the opposite of the usual small-sample intuition, where the worry is that the number
bounces around. Here it barely bounces. It is just shifted.

## 6. Where the asymmetry bites: which arm deserves more data

$\sqrt{r_{50} r_{500}}$ is a product, so the weaker arm dominates and there is a hard ceiling set
by it alone:

$$\cos(\hat d_{50}, \hat d_{\infty}) = \sqrt{r_{50}} = 0.877$$

Even against a perfectly estimated direction, the 50-pair estimate cannot exceed $0.877$. So:

| Change | Ceiling on the cosine |
|---|---|
| baseline, $50$ vs $500$ | $0.864$ |
| $500 \to 5000$ (10× the *large* arm) | $0.876$ |
| $50 \to 500$ (10× the *small* arm) | $0.957$ |
| both to $5000$ | $0.997$ |

Ten times more data on the already-good arm buys $0.012$. Ten times more on the weak arm buys
$0.093$, roughly eight times as much. The marginal value of data is concentrated entirely where
$r$ is low, because $r = S/(1+S)$ saturates. When you have a fixed budget of pair annotations,
this table is the allocation rule.

## 7. The case that actually breaks the analysis: shared contamination

Everything above assumed $\mathbb{E}[\varepsilon_i] = 0$ — that the non-concept differences are
idiosyncratic and average away. For word pairs that assumption is usually false, and this is
where the premise "differ only in a specific concept" does the heavy lifting.

Suppose there is a nuisance component **shared across pairs** — a corpus-frequency direction, a
formality direction, a "this word is rarer" direction:

$$x_i = \mu + b + \varepsilon_i, \qquad \mathbb{E}[\varepsilon_i] = 0$$

Then $\hat d_n \to \mu + b$ as $n \to \infty$. Consequences, and they are severe:

1. **More data does not help.** $b$ is not reduced by averaging; it is what the estimator
   converges to. $n=500$ is a more precise estimate of the wrong direction.
2. **The cosine between arms goes up, not down.** $b$ adds to the common component, so effective
   signal energy is $\|\mu + b\|^2 > \|\mu\|^2$, $r$ rises, and
   $\cos(\hat d_{50}, \hat d_{500})$ rises with it. Contamination *improves* every agreement
   statistic you have.
3. **No internal check can detect it.** Split-half reliability, permutation floors, matched-$n$
   comparisons, bootstrap intervals — all of them measure how well the estimator converges to
   its own limit. None of them ask whether that limit is $\mu$.

The one-line version: **reliability is not validity.** A split-half cosine of $0.97$ says the
estimator has converged. It says nothing about what it converged to. High agreement between a
50-pair and a 500-pair estimate is evidence of *precision*, and it is equally consistent with
"we nailed the concept" and "both are dominated by the same confound."

Detecting $b$ requires something outside the estimator's own sample: a held-out criterion, or a
second pair set built specifically so that the suspected confound points a different way while
the concept points the same way. If $\cos$ between directions from the two constructions stays
high, $b$ is small along the manipulated axis. If it collapses, you have found your $b$. That is
a design move, not a statistical one, and no amount of resampling substitutes for it.

This exact failure is the subject of a well-known dispute in the word-embedding literature —
Bolukbasi et al. (2016) construct a gender subspace by this kind of pair-difference method, and
Gonen & Goldberg (2019, "Lipstick on a Pig") argue that removing it leaves the bias largely
intact, i.e. that the recovered direction was not the thing it was taken to be. I have not
re-read either in this session; treat them as pointers to check rather than as summarized
claims.

## 8. Effective $n$: the 500 is probably not 500

Word pair sets are built by hand and are heavily related — `king/queen`, `kings/queens`,
`prince/princess`, `king/queen` again from a different list. Related pairs have correlated
$\varepsilon_i$. With average within-cluster error correlation $\rho$ and clusters of size $m$,
the usual design-effect approximation gives

$$n_{\text{eff}} \approx \frac{n}{1 + (m-1)\rho}$$

At $m=4$, $\rho=0.5$: $n_{\text{eff}} \approx 500/2.5 = 200$. Every $r$ above must be computed
at $n_{\text{eff}}$, not $n$, and every resampling split must be drawn **at the cluster level**,
not the pair level. Splitting `king/queen` into one half and `kings/queens` into the other makes
the halves share their nuisance, inflates the split-half cosine, and produces a reliability
estimate that is too high — which then makes the disattenuation too small. The errors compound
in the same direction.

## 9. So what comparison actually answers the question

$\cos(\hat d_{50}, \hat d_{500})$ answers almost nothing on its own, because it is a product of
three things — the two reliabilities and the true alignment — and you have measured one number.
To separate them:

1. **Measure $r$ at each $n$ independently.** Disjoint half-splits at cluster level, $B \ge 200$
   draws, Spearman–Brown up: $r_n = 2r_{n/2}/(1 + r_{n/2})$.
2. **Compare the observed cosine to its ceiling** $\sqrt{r_{50} r_{500}}$. At the ceiling, the
   two directions are the same and you are done. Materially below it, they differ for a reason
   other than noise.
3. **Subsample as the sharper test.** Draw $B$ subsets of 50 from the 500 and build the
   distribution of $\cos(\hat d_{50}^{(b)}, \hat d_{500})$. If your actual 50-pair estimate lands
   inside that distribution, its disagreement with the 500-pair estimate is fully explained by
   having 50 pairs, with no model assumptions at all. This is the cleanest available answer and
   it needs no $\Sigma$, no $\operatorname{tr}$, no delta method.
4. **Plot $r$ against $n$ rather than comparing two points.** The curve separates "this estimator
   is noisy" from "this estimator converges somewhere else," and its asymptote is the quantity
   you actually care about.
5. **Then go outside the sample** for validity, per §7. Steps 1–4 cannot do it.

## 10. Summary table

| Quantity | Value at $n{=}50$ | Value at $n{=}500$ | Fixed by more data? |
|---|---|---|---|
| $\mathbb{E}[\hat d_n]$ | $\mu$ | $\mu$ | already correct |
| $\|\hat d_n\|$ | $1.140$ | $1.015$ | yes, $\to \|\mu\|$ |
| $\cos$ to truth | $0.877$ | $0.985$ | yes |
| $\cos$ to a replicate | $0.769$ | $0.971$ | yes |
| sd of that cosine | $0.010$ | $0.001$ | yes, but it was never the problem |
| shared nuisance $b$ | present | present | **no** |

## Related

- `answers/concepts/july_28_2026/sample_mean_covariance_and_trace.md` — unpacks §1–2 above: why
  the mean has covariance $\Sigma/n$, what $\operatorname{tr}\Sigma$ is, and why only the trace
  enters the attenuation.
- `answers/concepts/july_28_2026/noise_floor_and_disattenuation.md` — the general treatment:
  disattenuation, cross-fitting, and what a permutation floor must be matched on.
- `answers/concepts/jun_19_2026/vti_rotation_strength_motivation.md`
