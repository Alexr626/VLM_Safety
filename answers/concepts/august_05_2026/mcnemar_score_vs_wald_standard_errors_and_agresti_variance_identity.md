# McNemar's $z$, the paired Wald interval, and why they use different standard errors

**Date:** 2026-08-05
**Question it answers:** why §4 of
[`paired_accuracy_inference_and_minimum_detectable_effect.md`](./paired_accuracy_inference_and_minimum_detectable_effect.md)
and Agresti §8.1 appear to give different test statistics and different intervals, when the
variance at the top of §4 is exactly the one Agresti's standard error is built from.

Numbers are either invented or taken from Agresti's own tables (2nd ed., 2007). No project
results appear here.

---

## 0. The compressed answer

Three separate things, in decreasing order of how much they matter.

1. **The two $z$ statistics are the same statistic.** My $b,c$ are Agresti's $n_{12},n_{21}$.
   The bridge from $\text{Bin}(n^*,\tfrac12)$ to $(n_{12}-n_{21})/\sqrt{n_{12}+n_{21}}$ is two
   lines: the numerator $n_{12}-\tfrac12 n^*$ *is* $\tfrac12(n_{12}-n_{21})$, the denominator
   $\sqrt{n^*/4}$ *is* $\tfrac12\sqrt{n^*}$, and the halves cancel. My note asserted the
   binomial and then wrote the normal form without showing the step. §1 below shows it.

2. **The real disconnect — and it is real, you found it.** The test and the interval do not use
   the same standard error, on purpose. $\operatorname{Var}(\hat\Delta) = (\pi_d - \Delta^2)/n$.
   The **test** evaluates that variance under $H_0:\Delta = 0$, where the $-\Delta^2$ term
   vanishes, giving $\sqrt{n^*}/n$. The **interval** evaluates it at $\hat\Delta$, where the term
   does not vanish, giving $\sqrt{n^* - (n_{12}-n_{21})^2/n}\,/\,n$. That is the score-versus-Wald
   distinction from §1–§2 of the morning note reappearing in the paired setting. My §4 put both
   formulas on the page and never said they were different objects. §2 below.

3. **Agresti (8.2) and my $(\pi_d-\Delta^2)/n$ are algebraically identical.** (8.2) is just
   $\operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X,Y)$ for the two marginal
   indicators, and $p_{11}p_{22}-p_{12}p_{21}$ *is* their sample covariance. The concordant cells
   enter only through $p_{11}+p_{22} = 1-\hat\pi_d$, so they cancel. §3 below does the cancellation
   term by term. Your own §4 already contained (8.2) — it is the "Relation to the unpaired
   variance" line, with the covariance written as $\pi_{11}-p_Bp_S$ instead of
   $\pi_{11}\pi_{22}-\pi_{12}\pi_{21}$. Those two expressions are equal.

If that settles it, stop here and go to §5. The rest is the working.

---

## 0.5 Notation bridge

| | Agresti §8.1 | §4 of the morning note |
|---|---|---|
| rows | question 1 / classification 1 | baseline correct? |
| cols | question 2 / classification 2 | steered correct? |
| $(1,1)$ | $n_{11}$ | $a$ |
| $(1,2)$ | $n_{12}$ | $b$ |
| $(2,1)$ | $n_{21}$ | $c$ |
| $(2,2)$ | $n_{22}$ | $d$ |
| discordant total | $n^* = n_{12}+n_{21}$ | $m = b+c$ |
| estimand | $\pi_{1+}-\pi_{+1} = \pi_{12}-\pi_{21}$ | $\Delta = \pi_{01}-\pi_{10}$ |

One sign difference and nothing else: Agresti's estimand is row-minus-column, mine is
column-minus-row, so $\hat\Delta_{\text{mine}} = (c-b)/n = -(p_{1+}-p_{+1})$. The variance is
unaffected ($\Delta$ enters squared) and a two-sided test is unaffected. From here I use
Agresti's labels throughout, so $\hat\Delta := p_{1+}-p_{+1} = (n_{12}-n_{21})/n$.

Two derived quantities used constantly below:

$$\hat\pi_d = \frac{n^*}{n} \quad(\text{discordance rate}), \qquad
\hat d = \frac{n_{12}-n_{21}}{n_{12}+n_{21}} \quad(\text{net split among discordants}).$$

Note $\hat\Delta = \hat\pi_d\,\hat d$. That factorisation is what makes §2 clean.

---

## 1. Where the $z$ in (8.1) comes from

### 1.1 The conditional argument

Full model: $(n_{11},n_{12},n_{21},n_{22}) \sim \text{Multinomial}(n;\ \pi_{11},\pi_{12},\pi_{21},\pi_{22})$.

Condition on the discordant total $n^* = n_{12}+n_{21}$. Each of those $n^*$ subjects is of type
$(1,2)$ or $(2,1)$, and given that a subject is one of the two, the probability it is type $(1,2)$
is $\pi_{12}/(\pi_{12}+\pi_{21})$. Independence across subjects gives

$$n_{12} \mid n^* \;\sim\; \text{Bin}\!\left(n^*,\ \frac{\pi_{12}}{\pi_{12}+\pi_{21}}\right).$$

Under $H_0: \pi_{12}=\pi_{21}$ that success probability is exactly $\tfrac12$:

$$n_{12}\mid n^* \;\sim\; \text{Bin}(n^*,\tfrac12).$$

**Why conditioning is the right move and not a trick.** The unconditional model has three free
parameters. $H_0$ constrains one combination of them and leaves $\pi_{11}$, $\pi_{22}$, and
$\pi_{12}+\pi_{21}$ as nuisance parameters. The conditional distribution above depends on *none*
of them — it is $\text{Bin}(n^*,\tfrac12)$ whatever the nuisance parameters are. Conditioning on
$n^*$ buys an exactly known null distribution, which is why the exact McNemar test exists at all.
The concordant cells $n_{11},n_{22}$ have disappeared not by approximation but because they are
ancillary to the question.

### 1.2 The two lines Agresti compresses

$\text{Bin}(n^*,\tfrac12)$ has mean $\tfrac12 n^*$ and variance $n^*(\tfrac12)(\tfrac12) = n^*/4$.
Standardise:

$$z = \frac{n_{12}-\tfrac12 n^*}{\sqrt{n^*(\tfrac12)(\tfrac12)}}.$$

That is the left-hand form of (8.1). Now substitute $n^* = n_{12}+n_{21}$ in both places.

**Numerator.**

$$n_{12}-\tfrac12(n_{12}+n_{21}) \;=\; \tfrac12 n_{12}-\tfrac12 n_{21} \;=\; \tfrac12\,(n_{12}-n_{21}).$$

**Denominator.**

$$\sqrt{n^*\cdot\tfrac14} \;=\; \tfrac12\sqrt{n^*} \;=\; \tfrac12\sqrt{n_{12}+n_{21}}.$$

**Divide.** The $\tfrac12$'s cancel:

$$z = \frac{\tfrac12(n_{12}-n_{21})}{\tfrac12\sqrt{n_{12}+n_{21}}} = \frac{n_{12}-n_{21}}{\sqrt{n_{12}+n_{21}}}.$$

That is the right-hand form of (8.1), and it is my §4's $z = (c-b)/\sqrt{b+c}$ up to the sign
convention in §0.5. Squaring gives $\chi^2 = (n_{12}-n_{21})^2/(n_{12}+n_{21})$ on 1 df.

That is the whole gap between the two presentations of the test. The morning note stated
$c\mid m \sim \text{Bin}(m,\tfrac12)$ and then wrote down $z=(c-b)/\sqrt{b+c}$ with the word
"approximation" between them and no algebra. The algebra is the four lines above.

### 1.3 The same statistic from the unconditional model — this is the one that matters for §2

Forget conditioning. Work directly with $\hat\Delta = (n_{12}-n_{21})/n$ and the variance from the
top of §4,

$$\operatorname{Var}(\hat\Delta) = \frac{\pi_d - \Delta^2}{n}, \qquad \pi_d = \pi_{12}+\pi_{21}.$$

A **score test** evaluates the variance at the null value of the parameter. Under $H_0:\Delta=0$:

$$\operatorname{Var}_0(\hat\Delta) = \frac{\pi_d - 0^2}{n} = \frac{\pi_d}{n}.$$

The $-\Delta^2$ term is *gone*, not approximated away. Estimate the remaining nuisance $\pi_d$ by
its restricted MLE. Maximising the multinomial likelihood subject to $\pi_{12}=\pi_{21}$ gives
$\hat\pi_{12}=\hat\pi_{21}=n^*/(2n)$, hence $\hat\pi_d = n^*/n$ — the same value as the
unrestricted MLE, which is why nothing subtle happens here. So

$$\text{SE}_0 = \sqrt{\frac{n^*/n}{n}} = \frac{\sqrt{n^*}}{n},$$

$$z_{\text{score}} = \frac{\hat\Delta}{\text{SE}_0} = \frac{(n_{12}-n_{21})/n}{\sqrt{n^*}/n} = \frac{n_{12}-n_{21}}{\sqrt{n_{12}+n_{21}}}.$$

**Identical to §1.2.** McNemar's $z$ is simultaneously (i) the normal approximation to an exact
conditional binomial and (ii) the score statistic in the unconditional multinomial. That
coincidence is the reason the next section is a Wald-versus-score story rather than a
conditional-versus-unconditional one.

---

## 2. The disconnect: null variance versus estimated variance

### 2.1 Two standard errors from one variance formula

$$\operatorname{Var}(\hat\Delta) = \frac{\pi_d-\Delta^2}{n}.$$

Everything follows from *where you evaluate this*.

**Evaluate at $\Delta = 0$ (score / test).** The $-\Delta^2$ term vanishes:

$$\boxed{\ \text{SE}_0 = \frac{\sqrt{n^*}}{n} = \frac{\sqrt{n_{12}+n_{21}}}{n}\ }$$

**Evaluate at $\Delta = \hat\Delta$ (Wald / interval).** The term does not vanish. Plug in
$\hat\pi_d = n^*/n$ and $\hat\Delta = (n_{12}-n_{21})/n$:

$$\widehat{\operatorname{Var}}(\hat\Delta) = \frac1n\left[\frac{n_{12}+n_{21}}{n} - \frac{(n_{12}-n_{21})^2}{n^2}\right] = \frac{1}{n^2}\left[(n_{12}+n_{21}) - \frac{(n_{12}-n_{21})^2}{n}\right]$$

$$\boxed{\ \text{SE}_W = \frac{1}{n}\sqrt{(n_{12}+n_{21}) - \frac{(n_{12}-n_{21})^2}{n}}\ }$$

which is Agresti's displayed SE verbatim, and my §4 bottom line verbatim with $b,c$ for
$n_{12},n_{21}$. So: **you were right that Agresti's SE is $\sqrt{(\hat\pi_d-\hat\Delta^2)/n}$.**
It is. And you were right that the test does not use it. It does not, because the test is
entitled to assume the null it is testing.

This is exactly §1 versus §2 of the morning note. There, on a single proportion:
$z_{\text{Wald}} = (\hat p - \pi_0)/\sqrt{\hat p(1-\hat p)/n}$ substitutes the estimate into the
denominator; $z_{\text{score}} = (\hat p-\pi_0)/\sqrt{\pi_0(1-\pi_0)/n}$ keeps the null value
there, and inverting *that* gives Wilson. Same move, one dimension up.

### 2.2 The exact ratio, and a one-directional theorem

Divide:

$$\frac{\text{SE}_W^2}{\text{SE}_0^2} = \frac{n^* - (n_{12}-n_{21})^2/n}{n^*} = 1 - \frac{(n_{12}-n_{21})^2}{n\,n^*} = 1 - \frac{\hat\Delta^2}{\hat\pi_d}.$$

Using $\hat\Delta = \hat\pi_d\hat d$ this becomes $1 - \hat\pi_d\hat d^{\,2}$, so

$$\boxed{\ \frac{\text{SE}_W}{\text{SE}_0} = \sqrt{1-\hat\pi_d\,\hat d^{\,2}}\,, \qquad
z_{\text{Wald}} = \frac{z_{\text{McNemar}}}{\sqrt{1-\hat\pi_d\,\hat d^{\,2}}}\ }$$

Three consequences, all worth carrying:

**(a) $\text{SE}_W \le \text{SE}_0$, always.** The correction term is a subtraction of a
non-negative quantity, with equality iff $n_{12}=n_{21}$ (where both statistics are $0$ anyway).
Therefore $|z_{\text{Wald}}| \ge |z_{\text{McNemar}}|$ always, and the Wald interval is always
narrower than the interval you would get from the null SE. **The disagreement runs one way only:**
the Wald CI can exclude $0$ while McNemar fails to reject; McNemar cannot reject while the Wald CI
covers $0$. The Wald interval is anti-conservative relative to the test, in the same direction and
for the same reason as in the single-proportion case.

**(b) When does the correction bite?** $\hat\pi_d\hat d^{\,2}$ is a product. It is near zero unless
discordance is *high* **and** the discordant split is *lopsided*. A study with 2% discordance can
have $\hat d = 1$ and the two SEs still agree to three decimals. A study with 50% discordance and a
55:45 split ($\hat d = 0.1$) likewise. You need both.

**(c) A $z$ computed from the CI's SE is not McNemar's $z$, and reporting it as such is an error.**
If you write "the 95% CI is $[\,\cdot\,,\cdot\,]$, which excludes zero, so $p<0.05$ by McNemar,"
you have used the wrong denominator to make the claim. State which SE produced which number. If
you want an interval that agrees with the test by construction, you invert the *family* of score
tests of $H_0:\Delta=\Delta_0$ — the variance is then evaluated at $\Delta_0$, not at $0$ and not
at $\hat\Delta$ — which is Tango's interval. Named in §4 of the morning note, not derived; the
point here is only that "invert the test" and "plug in the estimate" are two different intervals.

### 2.3 On Agresti's own data — this is your 8.2

Heaven/hell table from your 8.2 work: $n_{11}=833$, $n_{12}=125$, $n_{21}=2$, $n_{22}=160$,
$n=1120$. So $n^*=127$, $n_{12}-n_{21}=123$, $\hat\Delta = 123/1120 = 0.109821$,
$\hat\pi_d = 127/1120 = 0.113393$, $\hat d = 123/127 = 0.968504$.

| quantity | value |
|---|---|
| $\text{SE}_0 = \sqrt{127}/1120$ | $0.0100620$ |
| $\text{SE}_W = \sqrt{127 - 123^2/1120}/1120$ | $0.0095118$ |
| ratio $\sqrt{1-\hat\pi_d\hat d^{\,2}} = \sqrt{1-0.106363}$ | $0.945324$ |
| $z_{\text{McNemar}} = 123/\sqrt{127}$ | $10.9144$ |
| $z_{\text{Wald}} = \hat\Delta/\text{SE}_W$ | $11.5458$ |

Two different $z$'s on one table, differing by 5.8%, both testing $\Delta=0$. Your (a) computed
$10.914$ and your (b) computed $0.109821 \pm 1.645(0.0095118) = [0.09417,\ 0.12547]$ — both
correct as Agresti asks for them, and they use different denominators. That is what you noticed.
For contrast, the interval built from the null SE would be
$0.109821 \pm 1.645(0.0100620) = [0.09327,\ 0.12637]$, about 5.8% wider.

Why does the correction bite here? Because $\hat d = 0.9685$ — of 127 discordant subjects, 125
went one way. Almost maximally lopsided, at 11% discordance.

**Now your 8.1**, the environment table: $n_{11}=227$, $n_{12}=132$, $n_{21}=107$, $n_{22}=678$,
$n=1144$. $n^*=239$, $n_{12}-n_{21}=25$, $\hat\Delta = 25/1144 = 0.0218531$,
$\hat\pi_d = 0.208916$, $\hat d = 25/239 = 0.104603$.

$$\hat\pi_d\hat d^{\,2} = 0.208916 \times 0.010942 = 0.0022859, \qquad \sqrt{1-\cdot} = 0.998856.$$

$\text{SE}_0 = 0.0135137$, $\text{SE}_W = 0.0134982$. $z_{\text{McNemar}} = 1.61712$,
$z_{\text{Wald}} = 1.61897$. **They agree to three significant figures.** Your $1.62$ is right
either way, and the choice of SE is invisible on this table.

The contrast between 8.1 and 8.2 is the cleanest illustration of point (b) available: same
textbook, adjacent problems, and the correction is negligible in one and 6% in the other. It is
$\hat d$ that changed — $0.105$ versus $0.969$.

### 2.4 A synthetic table where they actually disagree about the conclusion

Invented, small on purpose. $n=12$: $n_{11}=5$, $n_{12}=3$, $n_{21}=0$, $n_{22}=4$.

- $\hat\Delta = 3/12 = 0.250$, $n^* = 3$, $\hat d = 1$, $\hat\pi_d = 0.25$.
- $z_{\text{McNemar}} = 3/\sqrt3 = 1.7321$, two-sided $p = 0.0833$. **Does not reject at 0.05.**
- $\text{SE}_W = \sqrt{3 - 9/12}/12 = \sqrt{2.25}/12 = 0.1250$.
- $z_{\text{Wald}} = 0.25/0.125 = 2.000$, two-sided $p = 0.0455$. **Rejects at 0.05.**
- 95% Wald CI: $0.25 \pm 1.96(0.125) = [0.005,\ 0.495]$ — excludes $0$, barely.
- 95% interval from the null SE: $0.25 \pm 1.96(0.14434) = [-0.033,\ 0.533]$ — covers $0$.

Ratio check: $\sqrt{1-0.25(1)} = 0.8660$, and $1.7321/0.8660 = 2.000$. ✓

Note the direction: the Wald interval is the one that rejects, per (a). And note where you had to
go to manufacture the disagreement — $n^*=3$ with a $3$–$0$ split, i.e. exactly the regime where
neither normal approximation is worth anything. The exact conditional test says
$p = 2\Pr[\text{Bin}(3,\tfrac12)=3] = 2(\tfrac18) = 0.25$: nothing here at all. **The gap between
the two asymptotic denominators is widest precisely where you should be using the exact test
instead of either.** That is not a coincidence; both approximations degrade for the same reason.

A moderate case for calibration — invented, $n=200$: $n_{11}=90$, $n_{12}=60$, $n_{21}=20$,
$n_{22}=30$. $\hat\Delta = 0.20$, $n^*=80$, $\hat\pi_d = 0.40$, $\hat d = 0.50$.
$z_{\text{McNemar}} = 40/\sqrt{80} = 4.472$; $\text{SE}_W = \sqrt{72}/200 = 0.042426$,
$z_{\text{Wald}} = 4.714$; 95% CIs $[0.1168,\ 0.2832]$ (Wald) versus $[0.1123,\ 0.2877]$ (null SE).
Both overwhelming, no conflict, ~5% width difference. Typical.

---

## 3. Agresti (8.2) and $(\pi_d-\Delta^2)/n$ are the same expression

You said the SE "is exactly square root of the variance provided at the top of section 4" and asked
why the marginal-proportion form looks nothing like it. Here is the cancellation.

### 3.1 Where (8.2) comes from — read it as a variance of a difference

Let $X_i = \mathbb 1\{\text{row}=1\}$ and $Y_i = \mathbb 1\{\text{col}=1\}$ for subject $i$. Then
$p_{1+}$ and $p_{+1}$ are the sample means of $X$ and $Y$, and $\hat\Delta = \bar X - \bar Y$. The
textbook formula for the variance of a difference of two correlated means gives

$$\operatorname{Var}(\hat\Delta) = \frac1n\Big[\operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X,Y)\Big].$$

$X$ and $Y$ are Bernoulli, so $\operatorname{Var}(X) = \pi_{1+}(1-\pi_{1+})$ and
$\operatorname{Var}(Y) = \pi_{+1}(1-\pi_{+1})$. And the covariance:

$$\operatorname{Cov}(X,Y) = \pi_{11} - \pi_{1+}\pi_{+1}.$$

Now the small identity that makes Agresti's form appear. With $\pi_{11}+\pi_{12}+\pi_{21}+\pi_{22}=1$,

$$\pi_{11}-\pi_{1+}\pi_{+1} = \pi_{11} - (\pi_{11}+\pi_{12})(\pi_{11}+\pi_{21})$$
$$= \pi_{11} - \pi_{11}^2 - \pi_{11}\pi_{21} - \pi_{11}\pi_{12} - \pi_{12}\pi_{21}$$
$$= \pi_{11}\underbrace{(1-\pi_{11}-\pi_{12}-\pi_{21})}_{=\ \pi_{22}} - \pi_{12}\pi_{21}$$
$$\boxed{\ \operatorname{Cov}(X,Y) = \pi_{11}\pi_{22} - \pi_{12}\pi_{21}\ }$$

**So the cross-product difference in (8.2) is not a new object — it is the covariance of the two
marginal indicators.** Substituting it into the variance-of-a-difference formula and replacing
parameters with sample proportions gives (8.2) exactly:

$$\big[p_{1+}(1-p_{1+}) + p_{+1}(1-p_{+1}) - 2(p_{11}p_{22}-p_{12}p_{21})\big]/n.$$

Your morning note already had this. §4, "Relation to the unpaired variance":
$\operatorname{Var}(\hat\Delta) = \frac1n[\sigma_B^2+\sigma_S^2-2\operatorname{Cov}(B_i,S_i)]$ with
$\operatorname{Cov} = \pi_{11}-p_Bp_S$. Same equation, covariance written in the un-simplified form.
The note failed to say "and this is Agresti (8.2)".

### 3.2 The cancellation

Write $a=p_{11}$, $b=p_{12}$, $c=p_{21}$, $e=p_{22}$ as *proportions*, with $a+b+c+e=1$. Then
$p_{1+}=a+b$, $p_{+1}=a+c$, $1-p_{1+}=c+e$, $1-p_{+1}=b+e$, and

$$\hat\pi_d = b+c, \qquad \hat\Delta = p_{1+}-p_{+1} = b-c.$$

**Expand (8.2)'s bracket.**

$$(a+b)(c+e) = ac + ae + bc + be$$
$$(a+c)(b+e) = ab + ae + bc + ce$$
$$-2(ae - bc) = -2ae + 2bc$$

Sum. The $ae$ terms are $ae + ae - 2ae = 0$ — **they cancel identically.** The $bc$ terms are
$bc + bc + 2bc = 4bc$. What remains:

$$\text{(8.2) bracket} = ab + ac + be + ce + 4bc.$$

**Expand $\hat\pi_d - \hat\Delta^2$.** Use $b+c = (b+c)(a+b+c+e)$ to homogenise:

$$(b+c)(a+b+c+e) = ab + b^2 + bc + be + ac + bc + c^2 + ce$$
$$= ab + ac + be + ce + b^2 + c^2 + 2bc$$
$$\hat\Delta^2 = (b-c)^2 = b^2 - 2bc + c^2$$
$$\hat\pi_d - \hat\Delta^2 = ab + ac + be + ce + \big[b^2+c^2+2bc\big] - \big[b^2-2bc+c^2\big] = ab+ac+be+ce+4bc.$$

$$\boxed{\ p_{1+}(1-p_{1+}) + p_{+1}(1-p_{+1}) - 2(p_{11}p_{22}-p_{12}p_{21}) \;=\; \hat\pi_d - \hat\Delta^2\ }$$

Divide by $n$ and take the square root and you have Agresti's displayed SE. That is the identity
you suspected.

### 3.3 Where the concordant cells went

Factor the common form:

$$ab+ac+be+ce+4bc = (a+e)(b+c) + 4bc.$$

The concordant proportions $a$ and $e$ appear **only** through their sum, and $a+e = 1-(b+c) = 1-\hat\pi_d$.
So

$$\hat\pi_d-\hat\Delta^2 = \hat\pi_d(1-\hat\pi_d) + 4p_{12}p_{21},$$

a third equivalent form, and one that makes the point visible: nothing about the split between
$p_{11}$ and $p_{22}$ survives. This is the same "concordant cells drop out" fact as §4 of the
morning note and as the conditioning argument in §1.1 above, arriving for a third time by a third
route.

The mechanism of the cancellation is worth naming: the two marginal variances each generate an
$ae$ cross-term, and $-2\operatorname{Cov}$ generates $-2ae$. The covariance term exists precisely
to remove them. When you write the variance in marginal form, the concordant cells enter and then
leave; when you write it as $\pi_d - \Delta^2$ they never enter. That is the whole reason the two
expressions look unrelated.

### 3.4 Numeric check on the heaven/hell table

$n=1120$: $p_{11}=0.7437500$, $p_{12}=0.1116071$, $p_{21}=0.0017857$, $p_{22}=0.1428571$;
$p_{1+}=0.8553571$, $p_{+1}=0.7455357$.

| piece | value |
|---|---|
| $p_{1+}(1-p_{1+}) = 0.8553571 \times 0.1446429$ | $0.1237213$ |
| $p_{+1}(1-p_{+1}) = 0.7455357 \times 0.2544643$ | $0.1897122$ |
| $p_{11}p_{22} = 0.7437500\times0.1428571$ | $0.1062500$ |
| $p_{12}p_{21} = 0.1116071\times0.0017857$ | $0.0001993$ |
| $-2(p_{11}p_{22}-p_{12}p_{21})$ | $-0.2121014$ |
| **sum** | $\mathbf{0.1013321}$ |

And the other route: $\hat\pi_d - \hat\Delta^2 = 0.1133929 - (0.1098214)^2 = 0.1133929 - 0.0120608
= 0.1013321$. Identical to seven places.

Third form: $\hat\pi_d(1-\hat\pi_d) + 4p_{12}p_{21} = 0.1133929(0.8866071) + 4(0.0001993)
= 0.1005350 + 0.0007972 = 0.1013322$. ✓

Divide by $n$: $0.1013321/1120 = 9.04751\times10^{-5}$; square root $= 0.0095118$. That is your
$\text{SE}$ from 8.2(b), reached from the marginal-proportion form without ever using the count
formula.

---

## 4. One thing in your 1.8 I want to ask about before saying anything

I worked 1.8 independently from the data in your scan ($n=1170$, $344$ successes) before reading
your answer.

(a) and (b) match mine: $\hat p = 344/1170 = 0.29402$, $\text{SE}_{\text{Wald}} = 0.013320$,
$z = -15.46$, $p < 0.001$.

Two questions.

**First — and this is the same distinction as §2, in the setting you already worked.** You tested
$H_0:\pi=0.5$ using $\text{SE} = \sqrt{\hat p(1-\hat p)/n} = 0.01332$. The score version uses the
null value in the denominator: $\sqrt{0.5(0.5)/1170} = 0.014618$, giving $z = -14.09$ rather than
$-15.46$. Both are far past any threshold so the conclusion is identical, but that is a 10%
difference in $z$ from the same choice you are asking about in 8.1/8.2. Which one does Agresti's
§1.4 present for this test, and which did you intend?

**Second, on 1.8(c).** Your half-width $2.576 \times 0.0133 = 0.0343$ agrees with mine. Your
interval is written as $[0.466,\ 0.5343]$. I get $[0.2597,\ 0.3283]$. The two differ by where the
interval is centred — yours is centred at $0.5$, mine at $\hat p = 0.294$. What were you centring
on at that step, and what was the interval meant to be a statement about? (There is a legitimate
object centred at $0.5$ — the acceptance region of the test, i.e. the set of $\hat p$ values that
would not reject — and it is not the same thing as a confidence interval for $\pi$. I want to know
which one you were building before I say more.)

---

## 5. Where to go next

Per the standing rule, the check is a textbook problem, not one of mine.

**Primary: Agresti 2nd ed., problem 1.12.** Wald versus score on a single proportion. This is §2 of
this note stripped of the paired structure — one estimand, two denominators, same null. Work it
and write down explicitly, for each part, *which value of the parameter the standard error was
evaluated at*. If you can say that sentence for 1.12, you can say it for McNemar. Follow with 1.18
for the same contrast in a different configuration.

**Secondary: Agresti 2nd ed., problem 8.5.** The exact / mid-$P$ version of McNemar. It lands in
the $n^*$-small regime of my §2.4 table, where the asymptotic disagreement is widest and neither
$z$ should be trusted. Doing it after 1.12 will make §2.4 concrete rather than a curiosity.

1.13, 1.15, 1.10, and 8.6 remain open from the earlier list.

---

## 6. Summary of the corrections to the morning note

For when you reread §4:

1. The step from $c\mid m\sim\text{Bin}(m,\tfrac12)$ to $z = (c-b)/\sqrt{b+c}$ was asserted.
   §1.2 above is the missing algebra.
2. §4 gave the McNemar $z$ and the paired CI without stating that they use different standard
   errors, or why. They are score and Wald respectively; $\text{SE}_W = \text{SE}_0\sqrt{1-\hat\pi_d\hat d^{\,2}} \le \text{SE}_0$,
   so the interval is always the narrower of the two and the disagreement is one-directional.
3. §4's "Relation to the unpaired variance" line *is* Agresti (8.2), with the covariance written
   as $\pi_{11}-p_Bp_S$ rather than $\pi_{11}\pi_{22}-\pi_{12}\pi_{21}$. The note did not connect
   them. §3.1 shows the two covariance forms are equal; §3.2 shows the whole expression collapses
   to $\hat\pi_d-\hat\Delta^2$.
