# Why the mean has covariance $\Sigma/n$, and what $\operatorname{tr}\Sigma$ means

Date: 2026-07-28
Question: "Why does $\operatorname{Cov}(\varepsilon_i) = \Sigma$ give
$\operatorname{Cov}(\hat d_n) = \Sigma/n$ — why is one equal to the other divided by $n$? And
explain total noise energy, with a refresher on trace."

Companion to `cosine_vs_sample_size_mean_difference_directions.md` §1–2. Synthetic numbers only.

---

## 0. The short answer, before the algebra

$\Sigma$ and $\Sigma/n$ are the covariances of **two different random vectors**:

| Random object | What it is | Covariance |
|---|---|---|
| $x_i = \mu + \varepsilon_i$ | one pair's difference vector, e.g. $e(\text{king}) - e(\text{queen})$ | $\Sigma$ |
| $\hat d_n = \frac1n\sum_i x_i$ | the average over $n$ such pairs | $\Sigma/n$ |

$\Sigma$ is a fact about the population of word pairs. It does not depend on $n$ and it does not
change when you collect more data — it is how much an individual pair scatters around $\mu$.
$\Sigma/n$ is a fact about *your experiment*: how much the number you computed scatters around
$\mu$. Averaging is a variance-reduction operation, and the $1/n$ is the whole content of it.

If that distinction is already clear, §1 is just the algebra confirming it.

## 1. Why the $1/n$

### 1.1 The scalar case you already have

For iid scalars with variance $\sigma^2$, $\operatorname{Var}(\bar x) = \sigma^2/n$. The vector
result is this, applied once per matrix entry. Nothing new happens in higher dimension.

### 1.2 Elementwise — the direct answer to "why divided by $n$"

Take entries $j, k \in \{1,\dots,d\}$. By definition,
$\operatorname{Cov}(\hat d_n)_{jk} = \operatorname{Cov}\!\big(\hat d_{n,j},\, \hat d_{n,k}\big)$, and
$\hat d_{n,j} = \mu_j + \frac1n\sum_i \varepsilon_{ij}$. The constant $\mu_j$ drops out
(covariance is translation-invariant), so

$$\operatorname{Cov}(\hat d_n)_{jk}
= \operatorname{Cov}\!\left(\frac1n\sum_{i=1}^n \varepsilon_{ij},\; \frac1n\sum_{i'=1}^n \varepsilon_{i'k}\right)
= \frac{1}{n^2}\sum_{i=1}^{n}\sum_{i'=1}^{n} \operatorname{Cov}(\varepsilon_{ij},\, \varepsilon_{i'k})$$

using bilinearity of covariance. Now the key step. The pairs are drawn independently, so for
$i \neq i'$ the vectors $\varepsilon_i$ and $\varepsilon_{i'}$ are independent and

$$\operatorname{Cov}(\varepsilon_{ij}, \varepsilon_{i'k}) = 0 \qquad \text{for } i \neq i'$$

**even when $j \neq k$.** Independence is across *pairs*, not across coordinates: coordinate $j$
of pair 1 is uncorrelated with coordinate $k$ of pair 2, but coordinate $j$ and coordinate $k$
of the *same* pair are very much correlated — that correlation is exactly $\Sigma_{jk}$.

So the double sum collapses from $n^2$ terms to the $n$ diagonal terms $i = i'$:

$$\operatorname{Cov}(\hat d_n)_{jk} = \frac{1}{n^2}\sum_{i=1}^{n} \operatorname{Cov}(\varepsilon_{ij}, \varepsilon_{ik})
= \frac{1}{n^2}\sum_{i=1}^{n} \Sigma_{jk} = \frac{1}{n^2}\cdot n\,\Sigma_{jk} = \frac{\Sigma_{jk}}{n}$$

That is the answer. $n^2$ in the denominator from squaring the $1/n$; $n$ in the numerator from
counting the surviving terms; net $1/n$. It holds for every entry $(j,k)$ simultaneously, so in
matrix form $\operatorname{Cov}(\hat d_n) = \Sigma/n$.

### 1.3 The same thing in matrix notation

Two standard facts:

$$\operatorname{Cov}(Ay) = A\operatorname{Cov}(y)A^\top, \qquad \operatorname{Cov}\Big(\sum_i y_i\Big) = \sum_i \operatorname{Cov}(y_i) \;\;\text{for independent } y_i$$

Let $S = \sum_i \varepsilon_i$. Independence gives $\operatorname{Cov}(S) = n\Sigma$. Then
$\hat d_n = \mu + \frac1n S$, and applying the first fact with $A = \frac1n I$:

$$\operatorname{Cov}(\hat d_n) = \tfrac1n I \,(n\Sigma)\, \tfrac1n I = \frac{\Sigma}{n}$$

### 1.4 Where the $n$ comes from, and where it does not

The $1/n$ came from **one place only: the cross terms vanishing**. It has nothing to do with $d$,
with the geometry of $\Sigma$, or with the size of $\mu$. High dimension does not weaken it and
anisotropy does not weaken it.

The geometric reading matters for what comes later. $\Sigma/n$ has the **same eigenvectors** as
$\Sigma$, with every eigenvalue scaled by $1/n$:

$$\Sigma = \sum_j \lambda_j u_j u_j^\top \;\;\Longrightarrow\;\; \frac{\Sigma}{n} = \sum_j \frac{\lambda_j}{n} u_j u_j^\top$$

So the noise ellipsoid keeps its orientation and its shape exactly, and shrinks by $1/\sqrt n$
uniformly along every axis. **Collecting more data does not make the noise more isotropic.** If
the per-pair noise is dominated by three directions, the noise in the mean of 10,000 pairs is
still dominated by the same three directions, just smaller. This is why "we have lots of data"
is not an argument that a permutation control can be replaced by an isotropic one.

### 1.5 Numerical check, $d = 2$

$$\Sigma = \begin{pmatrix} 4 & 1 \\ 1 & 9\end{pmatrix}, \qquad n = 4
\;\;\Longrightarrow\;\; \frac{\Sigma}{4} = \begin{pmatrix} 1 & 0.25 \\ 0.25 & 2.25\end{pmatrix}$$

- Per-coordinate sd: $(2, 3) \to (1, 1.5)$. Both divided by $\sqrt4 = 2$. ✓
- Correlation before: $1/(2\cdot3) = 0.167$. After: $0.25/(1 \cdot 1.5) = 0.167$. **Unchanged.** ✓
- $\operatorname{tr}$: $13 \to 3.25 = 13/4$. ✓

The correlation being unchanged is §1.4's shape-preservation, visible in one number.

### 1.6 What breaks it

The collapse of the double sum required $\operatorname{Cov}(\varepsilon_i, \varepsilon_{i'}) = 0$
for $i \neq i'$. If pairs are related — `king/queen` and `kings/queens` — the cross terms survive
and *add*, so

$$\operatorname{Cov}(\hat d_n) = \frac{\Sigma}{n_{\text{eff}}}, \qquad n_{\text{eff}} < n$$

Everything downstream still holds with $n_{\text{eff}}$ in place of $n$; the estimator is just
worse than its nominal sample size claims. Note the sign: positively correlated errors always
make $n_{\text{eff}}$ *smaller*, never larger, so the naive calculation is optimistic in the
direction that flatters your result.

## 2. Trace

### 2.1 Definition and the properties that earn their keep

For a square $A \in \mathbb{R}^{d\times d}$, the trace is the sum of the diagonal:

$$\operatorname{tr}(A) = \sum_{j=1}^{d} A_{jj}$$

Four properties, in order of how much they matter here:

1. **Sum of eigenvalues.** $\operatorname{tr}(A) = \sum_j \lambda_j$. This is why it summarizes a
   covariance: it totals the variance across all principal axes.
2. **Basis-invariant.** $\operatorname{tr}(P^{-1}AP) = \operatorname{tr}(A)$. Rotating your
   coordinate system does not change it. Since the choice of embedding-space basis is arbitrary,
   any quantity that *did* depend on it would be an artifact.
3. **Linear.** $\operatorname{tr}(A + B) = \operatorname{tr}A + \operatorname{tr}B$ and
   $\operatorname{tr}(cA) = c\operatorname{tr}A$. This is what lets you push it through
   expectations: $\operatorname{tr}(\mathbb{E}[A]) = \mathbb{E}[\operatorname{tr}(A)]$.
4. **Cyclic.** $\operatorname{tr}(AB) = \operatorname{tr}(BA)$ whenever both products are
   defined. Used constantly to turn an inner product into a trace and back — see §2.2.

### 2.2 Why $\operatorname{tr}\Sigma$ is the total noise energy

This is a two-line calculation and it is the whole justification for the phrase.

$$\mathbb{E}\|\varepsilon\|^2 = \mathbb{E}\left[\sum_{j=1}^d \varepsilon_j^2\right] = \sum_{j=1}^d \mathbb{E}[\varepsilon_j^2] = \sum_{j=1}^d \operatorname{Var}(\varepsilon_j) = \sum_{j=1}^d \Sigma_{jj} = \operatorname{tr}\Sigma$$

The third equality uses $\mathbb{E}[\varepsilon] = 0$, so the second moment *is* the variance.

The slicker route, worth having because you will meet it repeatedly in this literature: a scalar
equals its own trace, so $\|\varepsilon\|^2 = \varepsilon^\top \varepsilon = \operatorname{tr}(\varepsilon^\top\varepsilon)$,
and the cyclic property lets you swap the order:

$$\mathbb{E}\|\varepsilon\|^2 = \mathbb{E}\operatorname{tr}(\varepsilon^\top\varepsilon) = \mathbb{E}\operatorname{tr}(\varepsilon\varepsilon^\top) = \operatorname{tr}\big(\mathbb{E}[\varepsilon\varepsilon^\top]\big) = \operatorname{tr}\Sigma$$

**So "energy" is not a metaphor.** $\operatorname{tr}\Sigma$ is literally the expected squared
length of the noise vector — the same sense of "energy" as in signal processing, squared
magnitude. Two readings of the same quantity:

$$\operatorname{tr}\Sigma = \underbrace{\sum_j \Sigma_{jj}}_{\text{total variance summed over coordinates}} = \underbrace{\sum_j \lambda_j}_{\text{total variance summed over principal axes}} = \mathbb{E}\|\varepsilon\|^2$$

They agree because trace is basis-invariant, and the eigenbasis is just another basis.

### 2.3 Applying it to the estimate

Since $\operatorname{Cov}(\hat d_n) = \Sigma/n$ and the estimator is unbiased, the error of the
*mean* has expected squared length

$$\mathbb{E}\big\|\hat d_n - \mu\big\|^2 = \operatorname{tr}\!\left(\frac{\Sigma}{n}\right) = \frac{\operatorname{tr}\Sigma}{n} \;\equiv\; v_n$$

using linearity of trace. So $v_n$ is "total noise energy in the estimate," and it is the single
quantity that the attenuation depends on.

Running example from the companion note: $d = 300$, $\|\mu\| = 1$, $\operatorname{tr}\Sigma = 15$.

| | expected squared error | typical error norm | vs signal norm $1.0$ |
|---|---|---|---|
| one pair, $x_i$ | $15$ | $\sqrt{15} = 3.87$ | noise dominates, 4:1 |
| mean of 50 | $15/50 = 0.30$ | $0.55$ | signal dominates, 1:0.55 |
| mean of 500 | $15/500 = 0.03$ | $0.17$ | signal dominates, 1:0.17 |

Consistency check on the $1/\sqrt n$: $3.87/\sqrt{50} = 0.548 = \sqrt{0.30}$. ✓

This table is the intuition for the whole business. Any single word pair is a terrible estimate
of the gender direction — it is four times more "other stuff" than "gender." Fifty of them
averaged is already decent. That is $1/\sqrt n$ doing its work.

### 2.4 $S$ is dimensionless, and that is a useful check

$$S_n = \frac{n\|\mu\|^2}{\operatorname{tr}\Sigma} = \frac{\|\mu\|^2}{v_n} = \frac{\text{signal energy}}{\text{noise energy}}$$

Both are squared lengths, so $S$ carries no units. Rescale every embedding by $c$: then
$\mu \to c\mu$, $\Sigma \to c^2\Sigma$, and

$$S \to \frac{n c^2\|\mu\|^2}{c^2\operatorname{tr}\Sigma} = S$$

unchanged, hence $r = S/(1+S)$ unchanged. It must be, since cosine is scale-invariant and $r$ is
an expected cosine. If a derivation ever hands you an $r$ that moves when you rescale the
embeddings, there is an algebra error in it.

## 3. Why *only* the trace appears — and where the rest of $\Sigma$ comes back

This is the part worth sitting with. In the attenuation result, the full $d \times d$ matrix
$\Sigma$ collapses to one number. Why?

Because only two expectations entered the derivation:

$$\mathbb{E}\langle \hat d_1, \hat d_2\rangle = \|\mu\|^2 \quad(\text{needs only } \mathbb{E}\varepsilon = 0), \qquad
\mathbb{E}\|\hat d_n\|^2 = \|\mu\|^2 + \operatorname{tr}\Sigma/n$$

Neither one asks how the noise energy is *distributed* across directions. A noise cloud that is
a thin needle and a noise cloud that is a uniform ball inflate $\mathbb{E}\|\hat d_n\|^2$ by the
same amount if they have the same total energy. So to leading order the mean attenuation depends
on $\Sigma$ only through $\operatorname{tr}\Sigma$.

**The spectrum returns in the second moment.** Recall the participation ratio

$$d_{\text{eff}} = \frac{(\operatorname{tr}\Sigma)^2}{\operatorname{tr}(\Sigma^2)}$$

which counts how many directions the noise energy is genuinely spread over. Two covariances with
identical trace, $\operatorname{tr}\Sigma = 15$, $d = 300$:

| | eigenvalues | $\operatorname{tr}\Sigma$ | $\operatorname{tr}(\Sigma^2)$ | $d_{\text{eff}}$ |
|---|---|---|---|---|
| isotropic | $0.05$ on all 300 axes | $15$ | $300(0.05)^2 = 0.75$ | $225/0.75 = 300$ |
| concentrated | $5$ on 3 axes, $0$ elsewhere | $15$ | $3(5)^2 = 75$ | $225/75 = 3$ |

Same trace, so **the same expected cosine**: $r_{50} = 1/1.3 = 0.769$,
$\mathbb{E}\cos(\hat d_{50},\mu) = 0.877$ in both cases.

But the spread of that cosine across replicates differs by a factor of
$\sqrt{300/3} = 10$:

$$\text{isotropic: } 0.877 \pm 0.010 \qquad\qquad \text{concentrated: } 0.877 \pm 0.105$$

(Using $\operatorname{Var}[\cos] \approx \frac{v^2}{d_{\text{eff}}}\cdot\frac{v + a^2/2}{(a^2+v)^3}$
from the companion note §5. Substituting $d_{\text{eff}}$ for $d$ is a heuristic in the
anisotropic case, not an identity — the scaling is right, the constant is approximate. Verify by
simulation before relying on the second figure.)

So, stated cleanly:

> **$\operatorname{tr}\Sigma$ determines where the cosine sits on average. The rest of the
> spectrum determines how much you can trust any single measurement of it.**

In the concentrated case a lone measured cosine of $0.75$ or $0.99$ is unremarkable sampling
variation around $0.877$; in the isotropic case either would be extraordinary. Identical trace,
completely different inferential situation. That is the second reason permutation controls beat
synthetic isotropic ones: they inherit the real spectrum, not just the real total.

## Related

- `answers/concepts/july_28_2026/cosine_vs_sample_size_mean_difference_directions.md` — §1–2 are
  what this note unpacks.
- `answers/concepts/july_28_2026/noise_floor_and_disattenuation.md` — §2.1 on $d_{\text{eff}}$
  and when the ratio-of-expectations step is safe.
