# Error bounds on accuracy change: single-proportion intervals, the paired design, and the minimum detectable effect

**Date:** 2026-08-05
**Question it answers:** at $n=450$ binary items, what change in accuracy between a baseline
condition and a steered condition is distinguishable from sampling noise — and what decision
rule could have been written into the plan before any run happened.

All numbers below are invented. This note contains no project results.

---

## 0. Three different questions wearing the same clothes

The request was "a confidence interval based on the sample size of 450". That phrase names one
object but the analysis needs three, and they have different answers.

| Question | Object | Depends on |
|---|---|---|
| How precisely do I know baseline accuracy? | CI on a single proportion | $n$, $\hat p$ |
| Did steering change accuracy? | CI on the **difference**, paired | $n$, discordant counts |
| What change would I have been able to detect at all? | Minimum detectable effect | $n$, $\alpha$, power, discordance |

The first is the one asked for. It is the wrong tool for the second, and using it for the second
is the specific mistake this note is aimed at. The third is the one that belongs in a plan,
because it is the only one that can be computed before the run.

There is a fourth thing, and no amount of statistics supplies it: the **smallest effect you would
care about**. That is a claim about what you are asserting, not about the data. More in §8.

---

## 1. Accuracy as a binomial estimate, and the Wald interval

Let item $i \in \{1,\dots,n\}$, $n = 450$. Let $C_i \in \{0,1\}$ indicate a correct answer. Under
the standard model, $C_i \stackrel{iid}{\sim} \text{Bern}(p)$ and

$$X = \sum_i C_i \sim \text{Bin}(n, p), \qquad \hat p = X/n, \qquad \operatorname{Var}(\hat p) = \frac{p(1-p)}{n}.$$

By CLT,

$$\frac{\hat p - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} N(0,1).$$

The Wald interval takes the pragmatic route of substituting $\hat p$ for $p$ in the denominator
(justified by Slutsky, since $\hat p \to^P p$):

$$\boxed{\;\hat p \pm z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}\;}$$

### Half-widths at $n=450$

$\text{HW}(\hat p) = 1.96\sqrt{\hat p(1-\hat p)/450}$:

| $\hat p$ | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|
| $\pm$ HW (pts) | 4.62 | 4.53 | 4.23 | 3.70 | 2.77 | 2.01 |

**This is the single most useful number in the note.** At $n=450$ and accuracy near 0.7, a single
accuracy figure is known to about $\pm 4$ percentage points. Notice how flat the curve is: the
function $\sqrt{\hat p(1-\hat p)}$ barely moves between 0.5 and 0.8. You do not get much precision
back by being at high accuracy unless you are *very* high.

### Where Wald fails

Two failure modes, and they are not the same.

**(a) The interval leaves the parameter space, and degenerates at the boundary.** At $\hat p = 1$
the estimated standard error is exactly $0$ and the interval is the single point $[1,1]$ — a
claim of infinite precision from finite data. At $\hat p = 449/450$, $n=450$: HW $=
1.96\sqrt{0.99778 \times 0.00222/450} = 0.00435$, giving $[0.9934, 1.0021]$, which exceeds 1.

**(b) The subtler one: the two errors conspire.** The substitution $\hat p \to p$ in the
denominator is not innocent. $\hat p(1-\hat p)$ is maximized at $\hat p = 1/2$ and falls off
toward the boundaries. So when the sampling fluctuation pushes $\hat p$ *away* from $1/2$ — i.e.
in the direction of the boundary — the estimated standard error simultaneously *shrinks*. The
interval gets narrower exactly on the draws where it is most displaced from $p$. The two errors
are positively coupled rather than independent, and coverage falls below nominal systematically,
not just occasionally.

On top of this, the binomial is a lattice distribution, so coverage as a function of $n$ does not
converge monotonically — it oscillates with the position of the lattice points relative to the
interval endpoints. "$n = 450$ is large" is therefore not by itself a defence of the Wald
interval; the argument has to be about where $\hat p$ sits.

---

## 2. Wilson: invert the test instead of plugging in

The fix is to not substitute at all. Keep $p$ in the denominator and solve for the set of $p$ the
data does not reject:

$$\left|\frac{\hat p - p}{\sqrt{p(1-p)/n}}\right| \le z.$$

Square both sides and collect in $p$:

$$(\hat p - p)^2 \le \frac{z^2}{n}p(1-p)$$
$$\hat p^2 - 2p\hat p + p^2 \le \frac{z^2}{n}p - \frac{z^2}{n}p^2$$
$$p^2\left(1 + \frac{z^2}{n}\right) - p\left(2\hat p + \frac{z^2}{n}\right) + \hat p^2 \le 0.$$

This is a convex quadratic in $p$, so the solution set is the interval between its roots:

$$\boxed{\;p_{\pm} = \frac{\hat p + \dfrac{z^2}{2n} \;\pm\; z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}\;}$$

Two structural readings:

- **Centre.** $\dfrac{\hat p + z^2/(2n)}{1 + z^2/n}$ is $\hat p$ shrunk toward $1/2$. It is exactly
  the sample proportion you would get after adding $z^2/2 \approx 1.92$ successes and $1.92$
  failures. (Rounding that to 2 and 2 and then applying Wald is the Agresti–Coull interval.)
- **Radius.** The extra $z^2/(4n^2)$ under the root is what keeps the interval non-degenerate at
  $\hat p \in \{0,1\}$: at $\hat p = 1$ the first term vanishes but the second does not.

The interval is asymmetric about $\hat p$, which is correct — the sampling distribution of $\hat p$
is skewed away from the boundary, so the interval should be too.

### Where the difference bites at $n=450$

With $z = 1.96$: $z^2/n = 0.008537$, $z^2/(2n) = 0.004268$, $z^2/(4n^2) = 4.743\times10^{-6}$.

| $\hat p$ | Wald | Wilson | max endpoint gap |
|---|---|---|---|
| $0.720$ (324/450) | $[0.6785,\ 0.7615]$ | $[0.6768,\ 0.7595]$ | 0.0020 |
| $0.900$ (405/450) | $[0.8723,\ 0.9277]$ | $[0.8688,\ 0.9244]$ | 0.0035 |
| $0.980$ (441/450) | $[0.9671,\ 0.9929]$ | $[0.9624,\ 0.9894]$ | 0.0047 |
| $0.9911$ (446/450) | $[0.9824,\ 0.9998]$ | $[0.9774,\ 0.9965]$ | 0.0050 |
| $1.000$ (450/450) | $[1,\ 1]$ | $[0.9915,\ 1]$ | 0.0085 |

Read the last column. In the mid range the two intervals differ by a fifth of a percentage point —
irrelevant. The gap grows monotonically toward the boundary, and the degenerate row is where the
difference stops being cosmetic. Note also the asymmetry appearing at $\hat p = 0.98$: Wilson
extends $0.0176$ down and only $0.0094$ up.

**Practical consequence.** For a headline accuracy anywhere near the middle at $n=450$, Wald and
Wilson give the same picture and the choice does not matter. Use Wilson anyway, because the places
it matters are exactly the places you will eventually look: per-subset accuracies with small
denominators, yes-rates that collapse toward 0 or 1 under strong steering, and any adversarial
split where one class is nearly always right or nearly always wrong.

Clopper–Pearson is the third option: it inverts the *exact* binomial test rather than the normal
approximation, so its coverage is guaranteed $\ge 1-\alpha$, at the cost of being noticeably wide.
Which of the three you use is a **convention you adopt and state**, not a fact — see §9.

---

## 3. The unpaired two-proportion comparison, and why it is the wrong model here

Suppose you ignored the pairing and treated baseline and steered accuracy as two independent
samples. Then

$$\hat\Delta = \hat p_S - \hat p_B, \qquad \operatorname{Var}(\hat\Delta) = \frac{p_B(1-p_B)}{n} + \frac{p_S(1-p_S)}{n}.$$

At $p_B \approx p_S \approx 0.72$, $n=450$: each term is $4.48\times10^{-4}$, so
$\text{se} = \sqrt{8.96\times10^{-4}} = 0.02993$ and the 95% CI on the difference has half-width
$1.96 \times 0.02993 = 0.0587$ — nearly **6 percentage points**, before you have even asked about
power.

This is the wrong model because **your items are the same items**. Every AMBER discriminative item
is answered under both conditions by the same model. $B_i$ and $S_i$ are two measurements on one
unit, and they will be strongly correlated: an item that is easy at baseline is usually still easy
after a small activation perturbation. The independent-samples variance ignores that correlation
and pays for it.

This is the same structural point as §2 of `STEERING_MATH_REFERENCE.md`, one level up. There,
pairing demos before differencing removes the per-image random effect from the extraction. Here,
pairing items before differencing removes the per-item difficulty random effect from the
evaluation. Identical idea, different stage of the pipeline.

---

## 4. The paired structure: what cancels, and what carries the variance

Write the $2\times 2$ table of per-item correctness. For each item, $(B_i, S_i) \in \{0,1\}^2$:

|  | steered correct | steered wrong | row total |
|---|---|---|---|
| **baseline correct** | $a$ ($\pi_{11}$) | $b$ ($\pi_{10}$) | $n\hat p_B$ |
| **baseline wrong** | $c$ ($\pi_{01}$) | $d$ ($\pi_{00}$) | |
| **col total** | $n\hat p_S$ | | $n$ |

$b$ and $c$ are the **discordant** cells — items where the two conditions disagree. $a$ and $d$ are
concordant.

### The cancellation

$$p_B = \pi_{11} + \pi_{10}, \qquad p_S = \pi_{11} + \pi_{01}$$
$$\Delta \;=\; p_S - p_B \;=\; \pi_{01} - \pi_{10}, \qquad \hat\Delta = \frac{c-b}{n}.$$

$\pi_{11}$ cancels identically, and $\pi_{00}$ never appeared. **The parameter you care about is a
function of the discordant cells alone.** Items that both conditions get right, and items both get
wrong, carry no information about $\Delta$ whatsoever. This is the load-bearing algebraic fact of
the whole note.

### The variance

Two routes; take the short one first. $S_i - B_i \in \{-1, 0, +1\}$, and $(S_i - B_i)^2 = 1$
exactly on discordant items. So $E[(S_i-B_i)^2] = \pi_{01} + \pi_{10} =: \pi_d$, the discordance
rate, and

$$\operatorname{Var}(S_i - B_i) = E[(S_i-B_i)^2] - \Delta^2 = \pi_d - \Delta^2$$
$$\boxed{\;\operatorname{Var}(\hat\Delta) = \frac{\pi_d - \Delta^2}{n} = \frac{(\pi_{01}+\pi_{10}) - (\pi_{01}-\pi_{10})^2}{n}\;}$$

The long route confirms it. $(a,b,c,d) \sim \text{Multinomial}(n; \pi_{11},\pi_{10},\pi_{01},\pi_{00})$,
so $\operatorname{Cov}(b,c) = -n\pi_{10}\pi_{01}$ and

$$\operatorname{Var}(c-b) = n\pi_{01}(1-\pi_{01}) + n\pi_{10}(1-\pi_{10}) + 2n\pi_{01}\pi_{10} = n\left[\pi_d - \Delta^2\right].$$

**Neither $\pi_{11}$ nor $\pi_{00}$ appears.** The precision of your estimated accuracy change is
governed entirely by how often the two conditions disagree — not by $n$ alone, and not by the
marginal accuracy rates.

### Relation to the unpaired variance

$$\operatorname{Var}(\hat\Delta) = \frac{1}{n}\Big[\sigma_B^2 + \sigma_S^2 - 2\operatorname{Cov}(B_i,S_i)\Big], \qquad \operatorname{Cov}(B_i,S_i) = \pi_{11} - p_Bp_S.$$

So the paired variance is the unpaired variance minus $2\rho\sigma_B\sigma_S / n$. When the
marginals are close, $\sigma_B \approx \sigma_S = \sigma$ and

$$\operatorname{Var}(\hat\Delta) \approx \frac{2\sigma^2(1-\rho)}{n}.$$

Pairing buys a variance reduction of exactly $(1-\rho)$. At $\rho = 0.83$ that is a factor of $6$
in variance, $2.4$ in standard error. At $\rho = 0$ pairing buys nothing. At $\rho < 0$ pairing
*costs* you — and that case is not hypothetical for steering, since an intervention can break
items it used to get right while fixing others.

### McNemar's test

Under $H_0: \pi_{01} = \pi_{10}$ we have $\Delta = 0$, so the null variance drops the $\Delta^2$
term. Better: condition on the number of discordant items $m = b + c$. Given $m$, under the null
each discordant item is equally likely to have flipped either way, so

$$c \mid m \;\sim\; \text{Bin}(m, 1/2),$$

which is an **exact** null distribution requiring no asymptotics. The normal approximation to it is

$$z = \frac{c-b}{\sqrt{b+c}}, \qquad \chi^2 = \frac{(b-c)^2}{b+c} \sim \chi^2_1.$$

Look at what is *absent*: $n$. The test statistic does not contain the sample size at all. 450
items with 30 discordants gives exactly the same evidence as 4500 items with 30 discordants. Your
effective sample size for this question is $m$, not $n$.

Use the exact binomial when $m$ is small (a common rule is $m < 25$); the continuity-corrected
statistic $(|b-c|-1)^2/(b+c)$ sits between the two. Which you use is a convention to state.

### CI on the paired difference

$$\hat\Delta \pm z_{\alpha/2}\cdot \frac{1}{n}\sqrt{\,b + c - \frac{(c-b)^2}{n}\,}$$

This is the Wald form and inherits the boundary problems of §1 when $m$ is small or $|c-b|$ is
close to $m$. The score-interval analogues (Tango's interval; Newcombe's hybrid from the two
Wilson intervals) are the Wilson-grade fix for the paired difference. Named, not derived — the
point is that the same "invert the test rather than plug in" move is available here.

---

## 5. Worked example A: paired significant, unpaired not

Synthetic, AMBER-shaped. $n = 450$.

|  | steered correct | steered wrong | total |
|---|---|---|---|
| **baseline correct** | $a=316$ | $b=8$ | $324$ |
| **baseline wrong** | $c=22$ | $d=104$ | $126$ |
| **total** | $338$ | $112$ | $450$ |

$\hat p_B = 324/450 = 0.7200$, $\hat p_S = 338/450 = 0.7511$, $\hat\Delta = 14/450 = 0.0311$
(3.1 points). Discordants $m = 30$. Correlation: $\pi_{11} = 0.7022$, $p_Bp_S = 0.5408$, so
$\operatorname{Cov} = 0.1614$, $\sigma_B = 0.4490$, $\sigma_S = 0.4324$, $\rho = 0.832$.

**Marginal Wilson intervals.**
Baseline $[0.6768,\ 0.7595]$. Steered $[0.7092,\ 0.7889]$. These overlap over
$[0.7092,\ 0.7595]$ — a five-point band. Eyeballing the error bars says "no difference".

**Unpaired two-proportion test.** Pooled $\bar p = 662/900 = 0.7356$,
$\text{se}_0 = \sqrt{0.7356 \times 0.2644 \times 2/450} = 0.02940$, $z = 0.0311/0.0294 = 1.058$,
$p = 0.290$. Unpaired CI on the difference: $0.0311 \pm 0.0576 = [-0.0265,\ 0.0887]$. Contains 0.

**McNemar.** $z = 14/\sqrt{30} = 2.556$, two-sided $p = 0.0106$. Continuity-corrected:
$\chi^2 = 169/30 = 5.633$, $p = 0.0176$. Exact: $2\Pr[\text{Bin}(30,\tfrac12) \ge 22] =
2 \times 8656937/2^{30} = 2 \times 0.00806 = 0.0161$. All three reject at $\alpha = 0.05$.

**Paired CI.** $\widehat{\text{se}} = \frac{1}{450}\sqrt{30 - 196/450} = \frac{5.4373}{450} = 0.01208$.
CI $= 0.0311 \pm 0.0237 = [0.0074,\ 0.0548]$. Excludes 0.

**Summary of the disagreement:**

| | estimate | 95% CI on $\Delta$ | width | $p$ |
|---|---|---|---|---|
| unpaired | $+0.0311$ | $[-0.0265,\ 0.0887]$ | 11.5 pts | $0.290$ |
| paired | $+0.0311$ | $[\ 0.0074,\ 0.0548]$ | 4.7 pts | $0.011$ |

Same marginals, same $\hat\Delta$, factor 2.4 in interval width, and opposite conclusions. The
paired analysis knows something the unpaired one threw away: of 450 items, only 30 changed status,
and of those 30 the flips ran 22–8 in one direction. A 22–8 split is unlikely under a fair coin.
The unpaired test cannot see the 22–8 because it only ever sees 324 and 338.

---

## 6. Worked example B: unpaired significant, paired not

The reverse case exists and matters, because it is the one where pairing does not rescue you.
Marginals near $0.5$, and steering that scrambles per-item correctness.

|  | steered correct | steered wrong | total |
|---|---|---|---|
| **baseline correct** | $a=52$ | $b=164$ | $216$ |
| **baseline wrong** | $c=200$ | $d=34$ | $234$ |
| **total** | $252$ | $198$ | $450$ |

$\hat p_B = 0.480$, $\hat p_S = 0.560$, $\hat\Delta = 36/450 = 0.080$ (8 points). Discordants
$m = 364$ — **81% of items change status**. Correlation: $\pi_{11} = 0.1156$, $p_Bp_S = 0.2688$,
$\operatorname{Cov} = -0.1532$, $\rho = -0.618$.

**Unpaired.** Pooled $\bar p = 468/900 = 0.520$, $\text{se}_0 = \sqrt{0.52 \times 0.48 \times 2/450}
= 0.03331$, $z = 0.080/0.03331 = 2.402$, $p = 0.0163$. Rejects.

**McNemar.** $z = 36/\sqrt{364} = 1.887$, $p = 0.0592$. Does not reject at $\alpha = 0.05$.

An 8-point accuracy gain fails the correct test while a 3-point gain passed it in Example A. The
paired analysis is telling you that 364 items flipped and the net was only 36 — a 200–164 split,
which a fair coin produces often enough. That is a much weaker claim than "steering improved
accuracy by 8 points", and it is the honest one.

### The exact relation between the two statistics

Both tests have numerator $c-b$. The unpaired pooled-variance statistic is
$z_u = (c-b)/\sqrt{2n\bar p(1-\bar p)}$ and McNemar is $z_p = (c-b)/\sqrt{b+c}$. Therefore

$$\boxed{\;\frac{z_{\text{paired}}}{z_{\text{unpaired}}} = \sqrt{\frac{2n\bar p(1-\bar p)}{b+c}}\;}$$

Check: Example A, $2 \times 450 \times 0.7356 \times 0.2644 = 175.06$, $m = 30$, ratio
$= \sqrt{5.835} = 2.416$, and $1.058 \times 2.416 = 2.556$. ✓
Example B, $2 \times 450 \times 0.52 \times 0.48 = 224.64$, $m = 364$, ratio $= \sqrt{0.617} =
0.786$, and $2.402 \times 0.786 = 1.887$. ✓

The crossover is at $m = 2n\bar p(1-\bar p)$. And $2n\bar p(1-\bar p)$ is *exactly* the expected
discordant count if the two conditions were independent with those marginals: under independence
$\pi_d = p_B(1-p_S) + p_S(1-p_B)$, which at $p_B = p_S = \bar p$ is $2\bar p(1-\bar p)$.

**So the unpaired test is the paired test evaluated at the discordance rate you would see if
steering scrambled correctness independently of the baseline.** That is what "throwing away the
pairing" costs: you are charged the noise of an intervention that has no per-item relationship to
the baseline, whether or not yours does. Fewer discordants than independence predicts and pairing
wins; more, and it loses.

### The picture

Two panels side by side.

**Left panel — the trap.** $x$ axis categorical with two ticks, `baseline` and `steered`. $y$ axis
accuracy from 0.60 to 0.85. Plot a point at $(\text{baseline}, 0.720)$ with a vertical error bar
from 0.677 to 0.760, and a point at $(\text{steered}, 0.751)$ with a bar from 0.709 to 0.789.
Shade the horizontal band from 0.709 to 0.760 across the whole panel — that is the overlap, and it
covers about a third of the vertical range. Everything the eye does with this panel is wrong.

**Right panel — the answer.** $x$ axis a single tick, `steered − baseline`. $y$ axis from $-0.06$
to $+0.10$, with a solid horizontal reference line at $0$. Two error bars, jittered slightly apart
in $x$: the unpaired CI spanning $-0.0265$ to $0.0887$ and crossing the zero line, and the paired
CI spanning $0.0074$ to $0.0548$, entirely above it. Both are centred on the same point, $0.0311$.

The left panel plots the wrong quantity. The difference has its own sampling distribution, and it
is not recoverable by looking at two marginal intervals side by side.

### Why CI overlap is a bad rule even for independent samples

Set the pairing aside; this is a separate error. Two intervals fail to overlap when

$$\hat p_S - z\,\text{se}_S > \hat p_B + z\,\text{se}_B \iff \hat\Delta > z(\text{se}_B + \text{se}_S).$$

The difference test rejects when $\hat\Delta > z\sqrt{\text{se}_B^2 + \text{se}_S^2}$. Since
$\text{se}_B + \text{se}_S \ge \sqrt{\text{se}_B^2 + \text{se}_S^2}$ always, the overlap rule is
strictly the more demanding one. With equal standard errors the ratio is
$2\text{se}/(\sqrt2\,\text{se}) = \sqrt2$, so requiring non-overlap of 95% intervals means
requiring

$$\hat\Delta > 1.96\sqrt{2}\,\sigma_d = 2.772\,\sigma_d,$$

which is a test at $\alpha = 2\Phi(-2.772) = 0.0056$. **Non-overlapping 95% intervals is roughly a
$\alpha = 0.006$ test, not a $\alpha = 0.05$ test.** It is conservative by a factor of nine in
error rate, and you lose a lot of power for a threshold you did not choose. (The converse also
fails: overlapping intervals are entirely compatible with a significant difference, as Example A
shows.)

---

## 7. Minimum detectable effect at $n = 450$

This is the number that belongs in a plan, because it can be computed before any run.

Reject when $|\hat\Delta| > z_{\alpha/2}\sigma_0$. To have power $1-\beta$ at a true effect
$\Delta$, you need the alternative distribution to put $1-\beta$ of its mass past that threshold:

$$\frac{|\Delta| - z_{\alpha/2}\sigma_0}{\sigma_1} \ge z_\beta \;\Longrightarrow\; |\Delta| \ge z_{\alpha/2}\sigma_0 + z_\beta\sigma_1 \approx (z_{\alpha/2} + z_\beta)\,\sigma$$

taking $\sigma_1 \approx \sigma_0$. With $\alpha = 0.05$ two-sided and power $0.80$:
$z_{0.975} + z_{0.80} = 1.9600 + 0.8416 = 2.8016$.

### Unpaired

$$\text{MDE}_u = 2.8016\sqrt{\frac{2\bar p(1-\bar p)}{n}} = 2.8016\sqrt{\frac{2(0.72)(0.28)}{450}} = 2.8016 \times 0.03055 = 0.0856$$

**8.6 percentage points.** That is the effect size an unpaired analysis at $n=450$ can detect with
80% power. Anything smaller and the study is underpowered by construction.

### Paired

$\sigma \approx \sqrt{\pi_d/n}$, so

$$\boxed{\;\text{MDE}_p = (z_{\alpha/2}+z_\beta)\sqrt{\frac{\pi_d}{n}}\;}$$

At $n=450$, $\alpha=0.05$, power $0.80$:

| discordance $\pi_d$ | expected $m$ | MDE (pts) |
|---|---|---|
| 0.05 | 22 | 2.95 |
| 0.10 | 45 | 4.18 |
| 0.20 | 90 | 5.91 |
| 0.4032 | 181 | 8.56 |
| 0.60 | 270 | 10.23 |

The row at $\pi_d = 0.4032$ reproduces the unpaired MDE exactly, for the reason in §6.

**Range: 3 to 10 points at fixed $n=450$**, depending on something that is a property of the
intervention rather than of the benchmark. This is why "what accuracy change is meaningful at
$n=450$" has no single answer. Whether your design can detect a 3-point change depends on how
concordant steering is with baseline, and that is measurable from the run — it is not something
you have to assume, though for the plan you may have to bound or pilot it.

### The same thing in counts

Rejection needs $|c-b| \ge z_{\alpha/2}\sqrt{m}$. So the **smallest detectable net flip count** is
about $1.96\sqrt{m}$:

- $m = 30$ discordants → need a net of $11$ (e.g. 20–10 or wider).
- $m = 100$ → need a net of $20$ (60–40).
- $m = 364$ → need a net of $38$.

Concrete and easy to carry: with 30 items in play you need roughly a 2:1 split; with 364 in play, a
55:45 split will not do it.

### Inverting for $n$

$$n = \pi_d\left(\frac{z_{\alpha/2}+z_\beta}{\Delta}\right)^2$$

To detect a 2-point change at 80% power with $\pi_d = 0.10$: $n = 0.10 \times (2.8016/0.02)^2 =
0.10 \times 19622 = 1962$ items. At $\pi_d = 0.05$: $981$. At $\pi_d = 0.05$ and a 3-point target:
$436$ — just inside 450. That last line is the honest statement of what 450 items buys.

---

## 8. Pre-specified versus post-hoc, and what post-hoc costs

The MDE and the smallest effect worth caring about are different objects:

- **MDE** is a property of your design: $n$, $\alpha$, power, $\pi_d$. It answers "what could I
  detect."
- **SESOI** (smallest effect size of interest) is a property of your claim. It answers "what would
  I bother asserting." No statistical procedure supplies it. If you would not write "steering
  improves discriminative accuracy" for a 1-point gain, your SESOI is above 1 point, and you should
  say so before the run.

If MDE > SESOI, the experiment cannot answer the question at that $n$, and **that is knowable
before running anything**. This is what was missing from the plan, and it is a design fact rather
than a results fact.

A threshold chosen after seeing the numbers is a different object with no calibrated error rate,
for two reasons.

**(a) The effective number of comparisons is the number you would have made, not the number you
did make.** If you would have called a large gain on either model, at any of several coefficients,
"the result", then the family is all of those cells whether or not you computed them.

**(b) Selection inflates the estimate.** Suppose the true effect is $\Delta = 0.020$ with
$\text{se} = 0.012$, and you only report cells that reach significance, i.e. $\hat\Delta >
1.96 \times 0.012 = 0.0235$. Standardised threshold $a = (0.0235-0.020)/0.012 = 0.293$. Using
$E[Z \mid Z > a] = \phi(a)/(1-\Phi(a))$:

$$\phi(0.293) = 0.3822, \quad 1-\Phi(0.293) = 0.3846, \quad E[Z \mid Z>a] = 0.994$$
$$E[\hat\Delta \mid \text{selected}] = 0.020 + 0.012 \times 0.994 = 0.0319.$$

A true 2.0-point effect is reported as 3.2 points — a 60% inflation — and the power here is only
about 38%, so most of the time you see nothing and the times you do see something you overstate it.
This is the regime that low-powered selective reporting lives in, and the inflation is worst
exactly when power is lowest.

The remedy is boring and effective: write $\alpha$, power, the family, the primary cell, and the
SESOI into the plan before the run, and report every cell in the family regardless of outcome.

---

## 9. Multiplicity across models and configurations

You have $k$ = (models) $\times$ (steering configurations) cells. Say $k = 10$: two models, five
coefficient settings.

Under a global null where steering does nothing anywhere, and treating the tests as independent,

$$\Pr[\text{at least one } p < 0.05] = 1 - 0.95^{10} = 1 - 0.5987 = 0.401.$$

**A 40% chance of at least one "significant" cell from nothing.** If the question you ask is "did
any configuration work", $0.05$ per comparison is not $0.05$ for that question.

**Bonferroni.** Test each at $\alpha/k = 0.005$. Validity follows from Boole's inequality,
$\Pr[\bigcup A_i] \le \sum \Pr[A_i]$, which needs no independence assumption at all — independence
affects how *conservative* it is, not whether it is valid. Cost in MDE: $z_{0.0025} = 2.807$, so
the multiplier becomes $2.807 + 0.842 = 3.649$ versus $2.802$, a factor of $1.302$. At $\pi_d =
0.10$ and $n=450$ the paired MDE goes from $4.18$ to $5.44$ points.

**Šidák.** $1-(1-0.05)^{1/10} = 0.005116$ versus Bonferroni's $0.005$. Essentially identical at
these $\alpha$; the choice between them is not where anything is gained.

**When Bonferroni is too conservative.** When the $k$ tests are strongly positively dependent. Yours
are: the same 450 items, the same model, and neighbouring steering coefficients that produce
near-identical answer sets. If five coefficients on one model give almost the same predictions, the
family behaves like far fewer than five independent tests, and correcting at $k=10$ overcorrects —
you lose real power to guard against an error rate you were never running.

Three routes exist, and choosing among them is a design decision, so they are listed and not
ranked:

1. Define the family narrowly and say so — e.g. per model per benchmark, so $k=5$ rather than 10.
2. Pre-specify one primary cell tested at $\alpha=0.05$; everything else is secondary and reported
   with intervals but not called significant.
3. Control the false discovery rate rather than the family-wise error rate, which is less
   conservative under positive dependence.

What is not an option is computing 10 comparisons, reporting the one that came out best, and
quoting its uncorrected $p$.

---

## 10. Accuracy is not the only thing that moves

On a yes/no benchmark, accuracy can be flat while the yes-rate moves a lot. Two invented cases at
$n=450$ with 225 gold-yes and 225 gold-no items:

**Case 1 — accuracy moves, yes-rate does not.** Steering flips 20 gold-no items from "yes" (wrong)
to "no" (correct), and 20 gold-yes items from "no" (wrong) to "yes" (correct). Accuracy:
$+40/450 = +8.9$ points. Yes-rate: $-20 + 20 = 0$. Unchanged.

**Case 2 — yes-rate moves, accuracy does not.** Steering flips 20 gold-yes items from "no" to
"yes" (20 gained) and 20 gold-no items from "no" to "yes" (20 lost). Accuracy: $0$. Yes-rate:
$+40/450 = +8.9$ points.

Reporting accuracy alone cannot distinguish these, and they are different claims about the model.

The machinery transfers with no changes. Define $Y_i^B, Y_i^S \in \{0,1\}$ for "answered yes",
build the $2\times2$ table of $(Y^B, Y^S)$ ignoring the gold label, and run the same McNemar test
and the same paired CI on $\Delta_Y$. Note that this table has its **own** discordant count, and
it is generally much larger than the correctness table's: an item can change from yes to no
without changing whether it is correct only if... it cannot, actually — every yes/no flip on a
fixed gold label is also a correctness flip. The two tables have the *same* discordant items, but
the split between $b$ and $c$ differs, because the direction of a yes→no flip maps to
"correct→wrong" or "wrong→correct" depending on the gold label. That relationship is worth
working out on paper once; it is the connection between criterion shift and sensitivity change,
and it is where signal detection theory enters.

The reporting convention already in `CLAUDE.md` — report a metric with the counts it is built from
— is exactly what makes this recoverable after the fact. For this analysis the counts you need are
$a$, $b$, $c$, $d$, and to get those you need **per-item correctness under both conditions, joined
on item id**. If the run only wrote aggregate accuracies, the paired test is impossible and you are
stuck with the 8.6-point unpaired MDE. Check that the dumps carry per-item records before planning
around any of this.

---

## 11. What is a fact and what is a convention you are adopting

Keep these separate in the write-up. The first list is mathematics; the second is choices, and
each one has to be stated because someone could reasonably choose otherwise.

**Facts (consequences of the model, not choices):**
- $\Delta = \pi_{01} - \pi_{10}$; concordant cells cancel identically.
- $\operatorname{Var}(\hat\Delta) = (\pi_d - \Delta^2)/n$ — depends on discordant cells only.
- $\operatorname{Var}(\hat\Delta) \approx 2\sigma^2(1-\rho)/n$; pairing helps iff $\rho > 0$.
- $z_{\text{paired}}/z_{\text{unpaired}} = \sqrt{2n\bar p(1-\bar p)/(b+c)}$.
- Non-overlapping 95% intervals corresponds to a test at $\alpha \approx 0.006$.
- Under $H_0$, $c \mid m \sim \text{Bin}(m, 1/2)$ exactly.

**Conventions (yours to choose, and to state in the plan):**
- $\alpha = 0.05$, two-sided. Nothing derives this.
- Power $= 0.80$. Nothing derives this either.
- Wilson vs. Wald vs. Clopper–Pearson for single proportions.
- Asymptotic McNemar vs. continuity-corrected vs. exact binomial, and the $m$ threshold at which
  you switch.
- Wald-on-difference vs. Tango/Newcombe score interval for the paired CI.
- The definition of the family for multiplicity, and FWER vs. FDR.
- **The SESOI.** The one that is entirely a research judgement and cannot be delegated.

---

## 12. The recipe, compressed

1. Pull per-item correctness for both conditions, joined on item id.
2. Build the $2\times2$ table. Report $a, b, c, d$, not just the two accuracies.
3. Report each marginal accuracy with a Wilson interval — for "how well does it do", not for
   comparison.
4. Report $\hat\Delta = (c-b)/n$ with a paired CI, and McNemar's $p$ (exact if $m$ is small).
5. Never compare the two marginal intervals for overlap.
6. State the pre-specified $\alpha$, power, family, and SESOI; report every cell in the family.
7. Report the same table and test for the yes-rate.
8. State $\pi_d$ — it is what makes your MDE interpretable, and it varies by configuration.

---

## 13. Comprehension check

Do not read past the questions before working them.

**(1) Compute.** Synthetic table, $n=450$: $a = 290$, $b = 26$, $c = 40$, $d = 94$.
Find $\hat p_B$, $\hat p_S$, $\hat\Delta$, McNemar's $z$, the paired 95% CI, and the unpaired $z$.
State which tests reject at $\alpha = 0.05$. Then answer: **holding $c - b = 14$ fixed, what is the
largest $m$ that would still have given a significant McNemar result?**

**(2) Predict, without computing first.** Take Example A ($b=8$, $c=22$, $m=30$, $z=2.556$).
Suppose you halve the steering coefficient and, as a result, both $b$ and $c$ halve — to $b=4$,
$c=11$. The point estimate $\hat\Delta$ also halves. Does McNemar's $z$ halve? Predict the
direction and the factor before you compute it, then check. Explain the factor in one sentence
using the structure of the statistic.

**(3) Predict.** You keep $n=450$ but evaluate a second, independently drawn 450-item set and pool,
so $n=900$ and all four cells $a,b,c,d$ double. What happens to $\hat\Delta$, to McNemar's $z$, to
the paired CI half-width, and to the MDE? Give the factor for each.

**(4) Design, in your own numbers.** For one model and one steering configuration: state the
$\alpha$, power, family size, and SESOI you would commit to in advance. Compute the paired MDE at
$n=450$ under a $\pi_d$ you are willing to defend as a planning assumption, with the multiplicity
correction applied. Then state plainly whether $n=450$ can detect your SESOI. If it cannot, that is
the finding about the design, and it is available before the run.
