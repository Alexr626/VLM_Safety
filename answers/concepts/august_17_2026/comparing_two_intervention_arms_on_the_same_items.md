# Comparing two intervention arms scored on the same items: what McNemar answers, what it needs, and what it does not cover

**Date:** 2026-08-17
**Question it answers:** two arms, neither of them the untreated baseline, evaluated on one shared
600-item yes/no set. Is "split into TP/FP/TN/FN, then McNemar" the right procedure, and what
quantity belongs in the interval column of the reading table?

Builds on the three notes of 2026-08-05:

- [`paired_accuracy_inference_and_minimum_detectable_effect.md`](../august_05_2026/paired_accuracy_inference_and_minimum_detectable_effect.md) — **note 1**
- [`mcnemar_score_vs_wald_standard_errors_and_agresti_variance_identity.md`](../august_05_2026/mcnemar_score_vs_wald_standard_errors_and_agresti_variance_identity.md) — **note 2**
- [`stratified_and_multicondition_inference_on_paired_binary_outcomes.md`](../august_05_2026/stratified_and_multicondition_inference_on_paired_binary_outcomes.md) — **note 3**

Notation follows note 1: per item, $A_i, B_i \in \{0,1\}$ are correctness under the two arms;
$a,b,c,d$ are the cells of the paired $2\times2$ table with $b$ = A right / B wrong, $c$ = A wrong /
B right; $\hat\Delta = (c-b)/n$; $m = b+c$; $\hat\pi_d = m/n$.

**Every number below is invented by me.** No project result appears here, and I have not read
`analysis/`, `evaluation/results/`, or any run dump. The item count and the two-arm framing are
carried over from how you stated the question, as premises, and are not verified against disk.

---

## 1. The compressed answer

**Your framing is right.** Two arms on the same items is a paired binary design, and the
comparison of two accuracies from it is a McNemar-family problem, not a two-sample proportion
problem. That part of your recall of the 08/05 notes is correct.

**Your procedure, as stated, is not sufficient, and under one reading it is impossible.** "Separate
their true/false positive/negative counts, then calculate McNemar" has two readings:

- *Reading 1 — compute each arm's confusion matrix, then build McNemar from those.* **Cannot be
  done.** TP/FP/TN/FN are the **marginals** of each arm separately. McNemar needs the **joint**
  across arms — how many items arm A got right and arm B got wrong, item by item. Marginals do not
  determine a joint. §3 shows two datasets with byte-identical confusion matrices for both arms and
  McNemar $p$ values of $4\times10^{-8}$ and $0.21$.
- *Reading 2 — split the items by gold label, then build a separate paired table inside each half.*
  **Correct, and better than pooling.** But it is doing something different from what you said it
  was doing: gold-label splitting is the **stratification** axis, not the pairing axis. §4.

Those are two orthogonal axes and both are live here:

| axis | what varies | do items overlap? | machinery |
|---|---|---|---|
| **pairing** — arm A vs arm B | the intervention | same items, fully overlapping | covariance **subtracts**; McNemar; note 1 §4 |
| **stratification** — gold-yes vs gold-no | the item's true label | disjoint, no overlap | covariance is **zero**; variances add; note 3 §2 |

**Neither arm being the baseline changes less than you might expect, and one thing more than you
might expect.** McNemar is symmetric in the two conditions, so it does not care which one you call
the control — build the direct A-vs-B table and it is valid as-is. What *does* change: if the two
numbers in your table were each computed as a delta against a shared vanilla baseline, you must
**not** difference the two deltas and add their variances. They share the baseline, so they are
positively correlated and adding overstates the SE. The direct A-vs-B table sidesteps this
automatically. §5.

**What goes in the interval column:** the interval on $\hat\Delta = (c-b)/n$, built from the
discordant counts, i.e.

$$\hat\Delta \;\pm\; z_{\alpha/2}\cdot\frac1n\sqrt{\,m - \frac{(c-b)^2}{n}\,}.$$

Note the SE here is **not** the one McNemar's $z$ uses — the test uses the null SE $\sqrt m / n$,
the interval uses the estimated SE. That is note 2 §2 and it is a real distinction, not a rounding
detail. §6.

**One precondition that is not statistics and can kill the whole thing:** POPE-style benchmarks draw
several object-presence questions per image. If your 600 items are not 600 independent items but,
say, 100 images × 6 questions, then items within an image are correlated, every SE above is too
small, and every $p$ is too optimistic. §8. That is a fact about the item construction, and it is
yours to check.

Everything below is the working.

---

## 2. What must be true before any of this is computable

Before choosing a test, these have to hold. They are premise checks, not analysis.

1. **Per-item records exist for both runs, with item ids.** McNemar's inputs are $b$ and $c$, which
   only exist after a join on item id. If either run wrote only an aggregate accuracy, the paired
   test is not recoverable after the fact and you are stuck with the unpaired comparison and its
   much larger SE (note 1 §3). This is exactly what the `CLAUDE.md` reporting convention — report a
   metric with the counts it is built from — exists to protect.
2. **The item sets are identical, as id sets, not merely as counts.** "600 in both" is not "the same
   600". If the two draws differ at all, the design is partly unpaired and the pairing is only valid
   on the intersection.
3. **One response per item per arm, under fixed decoding.** If either run sampled, there is a
   decoding-noise variance component that no test below models.
4. **Same gold labels and same correctness rule in both runs.** A change in the parsing of "yes" or
   in the label file between June and July is confounded with the intervention and is invisible to
   every statistic here.
5. **Items are independent.** See §8 — for POPE-shaped item construction this is the one most likely
   to be false.

A distinct hazard that is not a statistics question and I will not answer for you: the two arms were
run **on different dates**. Anything that differs between the two runs other than the intervention —
checkpoint, prompt template, transformers version, image preprocessing, the answer parser — is
confounded with the arm. The tests below all condition on "the only thing that differs is the
intervention". Whether that holds is a claim about the two run configurations, and it is yours to
establish.

---

## 3. Why the confusion matrices are not enough — the load-bearing point

### 3.1 The marginals fix the numerator and say nothing about the denominator

$c - b = (\text{number B got right}) - (\text{number A got right}) = n(\hat p_B - \hat p_A)$.

That is a function of the two marginal accuracies alone. So the **point estimate is determined** by
the confusion matrices: you can read $\hat\Delta$ straight off them, and you already have.

But McNemar's statistic is

$$z = \frac{c-b}{\sqrt{b+c}} = \frac{c-b}{\sqrt m},$$

and $m = b+c$ is **not** a function of the marginals. Same numerator, unknown denominator. The
confusion matrices tell you the size of the effect and nothing at all about its precision.

### 3.2 The exact range of $m$ the marginals permit

Let $n_A$ = items A got right, $n_B$ = items B got right, $n$ = total. The concordant-correct cell
$a$ obeys the Fréchet bounds

$$\max(0,\ n_A + n_B - n)\ \le\ a\ \le\ \min(n_A,\ n_B),$$

and since $m = n_A + n_B - 2a$,

$$\boxed{\ |n_B - n_A|\ \le\ m\ \le\ \min\big(n_A+n_B,\ 2n - n_A - n_B\big)\ }$$

Substituting into $z$:

$$|z| = \frac{|n_B-n_A|}{\sqrt m} \ \in\ \left[\frac{|n_B-n_A|}{\sqrt{m_{\max}}},\ \sqrt{|n_B-n_A|}\right].$$

The upper end is worth carrying:

$$\boxed{\ |z_{\text{McNemar}}| \ \le\ \sqrt{n\,|\hat\Delta|}\ }$$

achieved when every flip runs the same way (perfect nesting, $b=0$ or $c=0$). At $n=600$ that means
significance is *impossible* unless $n|\hat\Delta| \ge 1.96^2 = 3.84$, i.e. at least 4 net flips,
i.e. $|\hat\Delta| \ge 0.0064$. That is a free screening bound from the marginals — but it is loose,
which is precisely why it cannot substitute for the join.

### 3.3 Worked: identical marginals, opposite conclusions

Invented. $n=600$. Arm A correct on 300 items ($\hat p_A = 0.500$), arm B correct on 330
($\hat p_B = 0.550$). So $\hat\Delta = +0.05$ and $c - b = 30$ in every case below. Fréchet range:
$a \in [30, 300]$, so $m \in [30, 570]$.

| joint | $a$ | $b$ | $c$ | $d$ | $m$ | $z = 30/\sqrt m$ | two-sided $p$ |
|---|---|---|---|---|---|---|---|
| max concordance | 300 | 0 | 30 | 270 | 30 | $5.477$ | $4.3\times10^{-8}$ |
| moderate | 240 | 60 | 90 | 210 | 150 | $2.449$ | $0.0143$ |
| min concordance | 30 | 270 | 300 | 0 | 570 | $1.257$ | $0.209$ |

**Same two accuracies. Same 5-point gain. $p$ spanning seven orders of magnitude.** The confusion
matrices are identical in all three rows and contain none of this information.

The crossing point is at $m = (30/1.96)^2 = 234$: below that the difference is significant, above it
it is not.

### 3.4 The picture for §3.3

One panel. $x$ axis: discordant count $m$, linear, from 30 to 570, labelled "items the two arms
disagree on". $y$ axis: $|z|$, from 0 to 6. Plot the curve $|z| = 30/\sqrt m$ — a decaying
power-law, steep on the left, flat on the right. Draw a solid horizontal line at $1.96$ and drop a
dotted vertical from its intersection with the curve down to the $x$ axis at $m = 234$, labelled
"threshold". Shade the region $m < 234$ lightly and label it "significant"; leave $m > 234$ unshaded.
Mark the left endpoint $m=30$ with a filled dot labelled "maximally nested" and the right endpoint
$m=570$ with a filled dot labelled "maximally scrambled". Put a caption stating that **every point on
this curve is consistent with the same pair of confusion matrices**, and that the join with per-item
ids is what tells you which point you are on.

---

## 4. The main worked example: two arms, two gold strata

Invented throughout. $n = 600$ items, $n_1 = 300$ gold-yes, $n_0 = 300$ gold-no, joined per item.

### 4.1 The two paired tables

**Gold-yes stratum** ($n_1 = 300$; on these items "correct" $\equiv$ "answered yes")

| | B correct | B wrong | total |
|---|---|---|---|
| **A correct** | $a_1 = 250$ | $b_1 = 8$ | $258$ |
| **A wrong** | $c_1 = 26$ | $d_1 = 16$ | $42$ |
| **total** | $276$ | $24$ | $300$ |

**Gold-no stratum** ($n_0 = 300$; "correct" $\equiv$ "answered no")

| | B correct | B wrong | total |
|---|---|---|---|
| **A correct** | $a_0 = 230$ | $b_0 = 30$ | $260$ |
| **A wrong** | $c_0 = 6$ | $d_0 = 34$ | $40$ |
| **total** | $236$ | $64$ | $300$ |

The two confusion matrices that fall out (this is all a marginal report would have given you):

| | TP | FN | FP | TN | accuracy |
|---|---|---|---|---|---|
| **arm A** | 258 | 42 | 40 | 260 | $518/600 = 0.8633$ |
| **arm B** | 276 | 24 | 64 | 236 | $512/600 = 0.8533$ |

### 4.2 Pooled — the headline number

$b = 38$, $c = 32$, $m = 70$.

$$\hat\Delta_{\text{pool}} = \frac{32-38}{600} = -0.01000 \quad(-1.00\ \text{points})$$
$$z_{\text{McNemar}} = \frac{-6}{\sqrt{70}} = -0.7172,\qquad p = 0.473$$
$$\text{SE}_W = \frac{1}{600}\sqrt{70 - \frac{36}{600}} = \frac{\sqrt{69.94}}{600} = 0.013938$$
$$\text{95\% CI} = -0.0100 \pm 1.96(0.013938) = [-0.0373,\ +0.0173].$$

Read straight, this is "arm B is 1.0 points worse, and that is well inside noise."

### 4.3 Per stratum — and the headline number was hiding a great deal

$$\hat\Delta_{\text{yes}} = \frac{26-8}{300} = +0.0600,\quad m_1 = 34,\quad z_1 = \frac{18}{\sqrt{34}} = +3.087,\quad p = 0.0020$$
$$\text{SE}_{W,1} = \frac{1}{300}\sqrt{34 - \frac{324}{300}} = \frac{\sqrt{32.92}}{300} = 0.019125,\quad \text{CI} = [+0.0225,\ +0.0975]$$

$$\hat\Delta_{\text{no}} = \frac{6-30}{300} = -0.0800,\quad m_0 = 36,\quad z_0 = \frac{-24}{\sqrt{36}} = -4.000,\quad p = 6.3\times10^{-5}$$
$$\text{SE}_{W,0} = \frac{1}{300}\sqrt{36 - \frac{576}{300}} = \frac{\sqrt{34.08}}{300} = 0.019459,\quad \text{CI} = [-0.1181,\ -0.0419]$$

**Each half of the benchmark moved by 6 to 8 points, in opposite directions, both with $|z|>3$.**
The pooled test saw $p = 0.47$ because the sum contrast really is near zero. Nothing is wrong with
the pooled test; it is answering a different question than the one you are asking when you say the
two arms differ.

### 4.4 The interaction: do the arms differ *differently* on the two strata?

The strata share no items, so the covariance is exactly zero and variances **add** (note 3 §2.2):

$$\hat\theta = \hat\Delta_{\text{yes}} - \hat\Delta_{\text{no}} = 0.0600 - (-0.0800) = +0.1400$$
$$\text{SE}(\hat\theta) = \sqrt{0.019125^2 + 0.019459^2} = \sqrt{7.4445\times10^{-4}} = 0.027284$$
$$z_\theta = \frac{0.1400}{0.027284} = 5.131,\qquad p = 2.9\times10^{-7},\qquad \text{CI} = [+0.0865,\ +0.1935].$$

### 4.5 The signal-detection reading of the same table

$$H_A = \tfrac{258}{300} = 0.860 \to H_B = \tfrac{276}{300} = 0.920 \quad (\Delta H = +0.060)$$
$$F_A = \tfrac{40}{300} = 0.1333 \to F_B = \tfrac{64}{300} = 0.2133 \quad (\Delta F = +0.080)$$

Both rates up: arm B answers "yes" more readily. Checks against note 3 §3.2:

- Youden $J$: $0.7267 \to 0.7067$, $\Delta J = -0.020 = 2\hat\Delta_{\text{pool}} = 2(-0.010)$ ✓
- Yes-rate: $298/600 = 0.4967 \to 340/600 = 0.5667$, change $+0.0700 = \tfrac12\hat\theta$ ✓

So the accuracy delta $-1.0$ points and the interaction $+14.0$ points are the two coordinates of one
2-D displacement $(\Delta H, \Delta F) = (+0.060, +0.080)$, rotated 45°. **Accuracy alone is one
linear functional of that displacement and cannot recover the other coordinate at any $n$** — note 3
§3.4. That is an identification failure, not a precision failure, and it is the reason a table with
one accuracy column per arm cannot answer "how do these two arms differ".

### 4.6 Summary of the example

| quantity | value | $z$ | $p$ |
|---|---|---|---|
| $\hat\Delta_{\text{pool}}$ (accuracy) | $-0.0100$ | $-0.717$ | $0.473$ |
| $\hat\Delta_{\text{yes}}$ | $+0.0600$ | $+3.087$ | $0.0020$ |
| $\hat\Delta_{\text{no}}$ | $-0.0800$ | $-4.000$ | $6.3\times10^{-5}$ |
| $\hat\theta$ (interaction) | $+0.1400$ | $+5.131$ | $2.9\times10^{-7}$ |
| yes-rate change | $+0.0700$ | — | — |
| $(\Delta H, \Delta F)$ | $(+0.060, +0.080)$ | — | — |

### 4.7 The same example, showing §3 bites here too

Hold both confusion matrices in §4.1 fixed and vary only the joint. The Fréchet extremes:

| stratum | joint | $b$ | $c$ | $m$ | $z$ | $p$ |
|---|---|---|---|---|---|---|
| gold-yes | as given | 8 | 26 | 34 | $+3.087$ | $0.0020$ |
| gold-yes | max concordance ($a_1=258$) | 0 | 18 | 18 | $+4.243$ | $2.2\times10^{-5}$ |
| gold-yes | min concordance ($a_1=234$) | 24 | 42 | 66 | $+2.216$ | $0.0267$ |
| gold-no | as given | 30 | 6 | 36 | $-4.000$ | $6.3\times10^{-5}$ |
| gold-no | max concordance ($a_0=236$) | 24 | 0 | 24 | $-4.899$ | $9.6\times10^{-7}$ |
| gold-no | min concordance ($a_0=196$) | 64 | 40 | 104 | $-2.353$ | $0.0186$ |

An order of magnitude in $p$ on the gold-yes stratum, from identical confusion matrices. The
$\hat\Delta$ column would be unchanged in every row.

### 4.8 The picture for §4

One panel, extending note 3 §9.1 with these numbers.

- $x$ axis, four categorical ticks: `gold_yes`, `gold_no`, then a **vertical dotted separator**, then
  `yes − no`. The separator matters: the third quantity is a contrast, on a different scale from the
  first two, and without it the eye reads it as a third stratum.
- $y$ axis: change in accuracy (arm B minus arm A), in proportion units, $-0.16$ to $+0.22$.
- A **solid** horizontal reference line at $0$.
- A **dashed** horizontal line at $\hat\Delta_{\text{pool}} = -0.0100$, with a thin shaded band
  across the full width for its 95% CI $[-0.0373, +0.0173]$. Label it "pooled accuracy change".
- Point + 95% bar at `gold_yes`: $+0.0600$, $[+0.0225, +0.0975]$.
- Point + 95% bar at `gold_no`: $-0.0800$, $[-0.1181, -0.0419]$.
- Point + 95% bar at `yes − no`: $+0.1400$, $[+0.0865, +0.1935]$.

What the panel shows in one glance: the pooled band hugs zero, and the two stratum points sit outside
it in opposite directions with intervals that do not reach it. Caption it with the sentence the
figure cannot show: **overlap of the two stratum bars would not test their difference** — that is the
third tick's job, and note 1 §6 has why.

---

## 5. Two arms, neither a baseline: what actually changes

### 5.1 McNemar does not privilege a control

Nothing in §1.1 of note 2 — the conditional argument, $n_{12}\mid n^* \sim \text{Bin}(n^*,\tfrac12)$
— refers to one condition as baseline. Swapping the roles of A and B swaps $b$ and $c$, flips the
sign of $\hat\Delta$ and of $z$, and changes nothing two-sided. So: build the direct A-vs-B table,
run McNemar, done. The null it tests is

$$H_0:\ \text{arm A and arm B have equal marginal accuracy on these items,}$$

which says nothing about either against an untreated baseline. Rejecting A-vs-B while failing to
reject A-vs-baseline is entirely coherent and is not a contradiction.

### 5.2 The trap: differencing two published deltas

Suppose your table's two entries were each formed as $\hat\Delta_A = \widehat{\text{acc}}_A -
\widehat{\text{acc}}_{\text{base}}$ and $\hat\Delta_B$ likewise, each with its own SE from its own
McNemar-vs-baseline analysis. The estimand you want is

$$\hat\Delta_B - \hat\Delta_A = \widehat{\text{acc}}_B - \widehat{\text{acc}}_A,$$

the baseline cancels **exactly**, so the difference-of-differences *is* the direct difference. There
is never a reason to compute it the long way. The danger is only in the variance:

$$\operatorname{Var}(\hat\Delta_B - \hat\Delta_A) = \operatorname{Var}(\hat\Delta_B) + \operatorname{Var}(\hat\Delta_A) - 2\operatorname{Cov}(\hat\Delta_A,\hat\Delta_B),$$

and the covariance is not zero. With per-item difference scores $D_{iA} = A_i - \text{base}_i$ and
$D_{iB} = B_i - \text{base}_i$,

$$\operatorname{Cov}(D_{iA}, D_{iB}) = \operatorname{Cov}(A_i, B_i) - \operatorname{Cov}(A_i, \text{base}_i) - \operatorname{Cov}(B_i, \text{base}_i) + \operatorname{Var}(\text{base}_i),$$

and the $+\operatorname{Var}(\text{base}_i)$ term is shared by both arms — the same baseline noise is
inside both deltas (note 3 §7.1). For two nearby interventions on one model this is strongly
positive.

**Consistency check that closes the loop.** $D_{iB} - D_{iA} = B_i - A_i$ identically, so

$$\operatorname{Var}(\hat\Delta_B - \hat\Delta_A) = \frac{\pi_d^{AB} - \Delta_{AB}^2}{n},$$

which is exactly note 1 §4 applied to the **direct** A-vs-B table. The direct table gives the right
answer with no covariance bookkeeping at all. That is the practical recommendation.

### 5.3 Numbers, tied to §4

Take $V = \operatorname{Cov}(D_{iA}, D_{iB})$ with invented diagonal $V_{AA} = 0.100$,
$V_{BB} = 0.090$ (arm A discordant with baseline on about 60 of 600 items, arm B on about 54). The
off-diagonal is then *forced* by §4's direct table, since $V_{AA} + V_{BB} - 2V_{AB}$ must equal
$\hat\pi_d^{AB} - \hat\Delta_{AB}^2 = 70/600 - 0.01^2 = 0.116567$:

$$V_{AB} = \tfrac12(0.190 - 0.116567) = 0.036717,\qquad \operatorname{corr} = \frac{0.036717}{\sqrt{0.100\times0.090}} = 0.387.$$

| route | SE of $\hat\Delta_B - \hat\Delta_A$ |
|---|---|
| correct: $\sqrt{(V_{AA}+V_{BB}-2V_{AB})/n} = \sqrt{0.116567/600}$ | $0.013938$ |
| naive, adding variances: $\sqrt{0.190/600}$ | $0.017795$ |

The correct value is identical to §4.2's $\text{SE}_W$, as it must be. The naive route is $1.28\times$
too large here — **conservative**, in the same direction and for the same reason as note 3 §7.3.

The penalty grows as the two arms resemble each other. If A and B disagreed on only 12 of 600 items
($\hat\pi_d^{AB} = 0.02$), the correct SE is $\sqrt{0.02/600} = 0.005774$ while the naive one is
unchanged at $0.017795$ — $3.08\times$ too large. Two nearly identical interventions are exactly the
case where you most need the pairing and where discarding it costs most.

### 5.4 If the two runs had different baselines

Everything in §5.2 assumed one shared baseline. If the June run's vanilla arm and the July run's
vanilla arm are different objects — different code, prompt, or parse — then

$$\hat\Delta_B - \hat\Delta_A = (\widehat{\text{acc}}_B - \widehat{\text{acc}}_A) - (\widehat{\text{acc}}_{\text{base},B} - \widehat{\text{acc}}_{\text{base},A}),$$

and the second bracket does not vanish. You now have a genuine difference-in-differences with two
more paired tables in it, and the estimand is "how much more did the intervention help in July than
in June", which is a different claim from "arm B beats arm A". **Which estimand your table's rows are
supposed to carry is a design question, and it is yours.** I am naming the fork, not resolving it.

---

## 6. What goes in the interval column, and what it licenses

### 6.1 The quantity

The interval belongs on the **delta**, not on each arm's accuracy. Reason, from note 1 §6: for arms
scored over the same items, the interval on the difference depends on the covariance between them,
and the covariance is not recoverable from the two marginal intervals. Two marginal Wilson intervals
side by side are compatible with essentially any interval on the difference, and comparing them for
overlap is a test at $\alpha \approx 0.006$ rather than $0.05$.

### 6.2 Which SE — the distinction that is easy to lose

One variance formula, two evaluation points (note 2 §2.1):

$$\operatorname{Var}(\hat\Delta) = \frac{\pi_d - \Delta^2}{n}$$

- **Test (score).** Evaluate at $\Delta = 0$; the $-\Delta^2$ term vanishes:
  $\text{SE}_0 = \sqrt m / n$. This is McNemar's denominator.
- **Interval (Wald).** Evaluate at $\hat\Delta$; the term survives:
  $\text{SE}_W = \frac1n\sqrt{m - (c-b)^2/n}$.

with $\text{SE}_W = \text{SE}_0\sqrt{1 - \hat\pi_d \hat d^{\,2}} \le \text{SE}_0$ always, so the
interval is always the narrower of the two and the disagreement is one-directional.

On §4's pooled table: $\hat\pi_d\hat d^{\,2} = (70/600)(6/70)^2 = 0.000857$, so the two SEs agree to
four figures and the choice is invisible. On §4's gold-yes stratum: $(34/300)(18/34)^2 = 0.0318$, so
$\text{SE}_W$ is $1.6\%$ narrower — still cosmetic. The correction only bites when discordance is
high *and* the discordant split is lopsided (note 2 §2.2b).

**Do not write "the CI excludes zero, so $p<0.05$ by McNemar."** That uses one denominator to make a
claim about a statistic computed with the other. State which is which. If you want an interval that
agrees with the test by construction, invert the family of score tests — Tango's interval — named in
note 1 §4, not derived.

### 6.3 What to report alongside it

Per the `CLAUDE.md` convention that a ratio be reported with the counts it is built from, the minimum
honest record of a paired comparison is:

$$n,\quad a,\ b,\ c,\ d,\quad m = b+c,\quad \hat\Delta,\quad \text{SE (and which one)},\quad \text{CI},\quad p\ \text{(and which test)}.$$

$m$ in particular. Note 1 §7 showed the paired MDE is $(z_{\alpha/2}+z_\beta)\sqrt{\pi_d/n}$, so
without $\hat\pi_d$ the interval cannot be compared against the pre-run "smallest difference
separable from noise" commitment that motivated the column in the first place. At $n=600$, $\alpha
= 0.05$ two-sided, power $0.80$:

| $\pi_d$ | expected $m$ | paired MDE |
|---|---|---|
| 0.05 | 30 | $2.56$ pts |
| 0.10 | 60 | $3.62$ pts |
| 0.20 | 120 | $5.12$ pts |
| 0.40 | 240 | $7.24$ pts |

Same $n$, a factor of nearly three in what is detectable, driven by a property of the intervention
pair rather than of the benchmark.

### 6.4 What the interval does not cover

The only randomness modelled is **over items**, conditional on: this model, this extracted direction,
this decoding, this prompt. Three things it therefore says nothing about:

1. **Direction-estimation variance.** The steering direction was estimated from a demo set with its
   own finite-sample error (the July 28 notes on cosine-vs-$n$ and the noise floor are that
   component). Two arms using directions estimated from different demo draws differ partly for that
   reason, and no McNemar interval covers it. Resampling the demo set is the only route, and it is a
   different experiment.
2. **Run-to-run variance.** There is no replication at the run level, so "this would happen again" is
   not licensed.
3. **Whether the size is worth asserting.** Rejecting $H_0$ says $\Delta \ne 0$. Whether $-1.0$
   points, or $+14.0$ points of interaction, supports the claim you want to make is the SESOI
   question of note 1 §8 and the interpretation question after it. Both are outside the test and
   both are yours.

---

## 7. If the table has more than one row

Two structurally different cases, and they need different machinery.

**Rows are disjoint item subsets** (gold-yes/gold-no; or POPE's random / popular / adversarial
negative-sampling regimes). Then the per-row effects are independent, variances add for any contrast
across rows, and §4.4 is the template. Two further consequences from note 3 §2.4:

- The pooled effect is $\hat\Delta_{\text{pool}} = \sum_s \lambda_s \hat\Delta_s$ with
  $\lambda_s = n_s/n$, and its stratified variance is $\sum_s \lambda_s^2 \operatorname{Var}(\hat\Delta_s)$.
- That is **not** the unstratified $(\hat\pi_d - \hat\Delta^2)/n$. Stratifying narrows the main-effect
  interval by exactly the between-stratum term $\lambda(1-\lambda)(\Delta_1-\Delta_0)^2$, i.e. by
  exactly the amount the strata disagree. In §4's numbers: stratified SE
  $= \sqrt{0.25(3.6578 + 3.7867)\times10^{-4}} = 0.013641$ against unstratified $0.013938$, a $2.1\%$
  narrowing. Free, and larger the more the strata diverge.

**Rows are additional arms on the same items.** Then they are correlated through the shared items and
variances do not add. Stack into $\hat{\boldsymbol\Delta} \in \mathbb R^k$, estimate $V$ as the sample
covariance of the per-item difference-score vectors, and every question is one quadratic form
$n(C\hat{\boldsymbol\Delta})'(C\hat VC')^{-1}(C\hat{\boldsymbol\Delta})$ — note 3 §7. And the
multiplicity discipline applies: with $k$ arms you have $\binom k2$ pairwise contrasts, "the best of
$k$ had $p=0.03$" is not a $p=0.03$ finding, and the omnibus-then-post-hoc structure or a
pre-specified primary contrast is what buys back a calibrated statement (note 3 §8).

---

## 8. The precondition most likely to be violated: clustered items

POPE-family benchmarks are constructed by drawing several object-presence questions per image. If
your 600 items come from, say, 100 images at 6 questions each, then $\{D_i\}$ are not 600
independent draws — items from one image share the image, the caption, and whatever the model got
right or wrong about that scene. Every SE in this note assumed independence across items.

### 8.1 The size of the error

For a mean of $n = Gk$ observations in $G$ clusters of size $k$, with within-cluster correlation
$\rho_c$ of the per-item difference score,

$$\operatorname{Var}(\hat\Delta)_{\text{clustered}} = \operatorname{Var}(\hat\Delta)_{\text{iid}} \times \underbrace{\big[1 + (k-1)\rho_c\big]}_{\text{design effect}}.$$

Invented, at $k = 6$:

| $\rho_c$ | DEFF | SE inflation |
|---|---|---|
| $0.05$ | $1.25$ | $\times 1.118$ |
| $0.10$ | $1.50$ | $\times 1.225$ |
| $0.20$ | $2.00$ | $\times 1.414$ |
| $0.30$ | $2.50$ | $\times 1.581$ |

Applied to §4's gold-yes stratum at $\rho_c = 0.20$: $\text{SE}$ goes $0.019125 \to 0.027046$, the
CI widens from $[0.0225, 0.0975]$ to $[0.0070, 0.1130]$, and $z$ falls from $3.087$ to $2.183$
($p: 0.0020 \to 0.029$). The conclusion survives here but a marginal one would not. Note that the
$p$-values themselves are the thing that moves most.

### 8.2 The two fixes

**Cluster-robust (sandwich) variance for a mean.** Let $T_g = \sum_{i \in g} D_i$ be the cluster
total. Then

$$\widehat{\operatorname{Var}}(\hat\Delta) = \frac{G}{G-1}\cdot\frac{1}{n^2}\sum_{g=1}^{G}\big(T_g - n_g\hat\Delta\big)^2 .$$

Note the structure: it treats the $G$ cluster totals as the independent units, so the effective
sample size is $G$, not $n$. Reduces to the iid formula when $k=1$.

**Cluster bootstrap.** Resample **images** with replacement $G$ times, rebuild the item set from the
drawn images, recompute $\hat\Delta$; repeat $10^4$ times; take the $2.5$/$97.5$ percentiles. This is
the route I would default to for a paired binary outcome, because it needs no distributional
assumption and handles unequal cluster sizes without extra work. The unit of resampling is the
image, not the item — that is the whole content of the method.

Either way, McNemar's $p$ as computed in §4 is no longer valid, because its conditional argument
assumes independent discordant items. The exact conditional test has no clustered analogue; the
bootstrap interval or the cluster-robust $z$ replaces it.

**Whether this applies to your 600 is a question about how the item set was drawn, and it is yours to
check** — count distinct images among the item ids.

---

## 9. Facts and conventions, kept apart

**Facts — consequences of the model and the design:**

- $c-b = n(\hat p_B - \hat p_A)$ is determined by the marginals; $m = b+c$ is not. Two confusion
  matrices fix $\hat\Delta$ and say nothing about its SE.
- Fréchet: $|n_B-n_A| \le m \le \min(n_A+n_B,\ 2n-n_A-n_B)$, hence
  $|z_{\text{McNemar}}| \le \sqrt{n|\hat\Delta|}$.
- McNemar is symmetric in the two conditions; there is no privileged baseline.
- $\hat\Delta_B - \hat\Delta_A = \widehat{\text{acc}}_B - \widehat{\text{acc}}_A$ exactly when the
  baseline is shared; the estimands coincide and only the variance route differs.
- Adding the variances of two deltas that share a baseline overstates the SE (positive covariance),
  i.e. it is conservative.
- Disjoint strata ⟹ zero covariance ⟹ variances add for both the sum and the difference contrast.
- Sum contrast = accuracy change; difference contrast = yes-rate change; pooled accuracy is one
  linear functional of $(\Delta H, \Delta F)$ and cannot identify the other coordinate at any $n$.
- $\text{SE}_W = \text{SE}_0\sqrt{1-\hat\pi_d\hat d^{\,2}} \le \text{SE}_0$; test and interval use
  different denominators by design.
- Clustered items inflate the true variance by $1+(k-1)\rho_c$; the effective sample size is the
  number of clusters.

**Conventions — yours to choose and to state:**

- $\alpha$, power, the family for multiplicity, and the SESOI. Still yours, still underivable.
- Wald vs score/Tango for the interval; asymptotic vs continuity-corrected vs exact for the test, and
  the $m$ threshold for switching.
- The scale the interaction is defined on: difference of proportions vs log-odds. Not scale-free, and
  can change sign (note 3 §6.2).
- Whether "accuracy" is the benchmark's own $\lambda$ or balanced ($\lambda = \tfrac12$).
- Cluster-robust SE vs cluster bootstrap, and whether to cluster at all.
- **Which estimand the table's rows carry** — direct arm-vs-arm, or difference-in-differences against
  two possibly-different baselines (§5.4).

---

## 10. Where to go next

Per the standing rule, the check is a textbook problem from a named source, not one of mine. All
three of these are already assigned in `learning/review_queue.md` and not yet worked.

**Primary: Agresti, *An Introduction to Categorical Data Analysis*, 2nd ed. (2007), problem 8.6**
(p. 267), Table 7.19, matched pairs stratified by gender. Part **(d)** is §4.4 of this note in
Agresti's own words: "Explain how you could construct a 90% confidence interval for the difference
between males and females in their differences of proportions of support for a particular item.
(Hint: The gender samples are independent.)" Gender plays the role of the gold label. Work (d)
before rereading §4.4, and write down explicitly which variances add and why. Parts (a)–(c) are
note 1 and note 3 §6 on the same table.

**Second: 8.12** (p. 269), matched triplets — alcohol, cigarettes, marijuana — comparing three
marginal "yes" proportions on the same subjects. This is the structure you actually have: **several
conditions, none of them a control**, on one shared item set. Part (b) sets up the three-way table
that makes the generalized CMH procedure apply, i.e. Cochran's Q, i.e. note 3 §7.2.

**Third, if $m$ turns out small in any stratum: 8.5**, the exact / mid-$P$ McNemar. §4's strata have
$m \approx 35$, above the usual $m<25$ switch point, but a finer stratification would drop below it
fast.

---

## 11. Comprehension check

Do 8.6(d) first. Before you start it, answer these three in writing.

**(1)** In §3 I claimed that two confusion matrices determine $\hat\Delta$ exactly and determine
nothing about its standard error. **State in one sentence which piece of information the join on item
id supplies that the two confusion matrices do not**, and then say what the largest possible
$|z_{\text{McNemar}}|$ is, in terms of $n$ and $\hat\Delta$ alone, and what has to be true of the
flips for that bound to be attained.

**(2)** In §5.3 the naive SE was too large, and in §4.4 the two stratum variances added with no
correction at all. Both are comparisons of two paired effects. **Name the one structural difference
between the two designs that produces this**, and say which of the two makes an independent-samples
SE exact and which makes it conservative.

**(3)** Predict before computing. Take §4's numbers and suppose you had instead evaluated on 1200
items — the same 600 plus 600 more drawn the same way, with all eight cell counts doubling. Give the
factor by which each of these changes: $\hat\Delta_{\text{pool}}$, $m$, $z_{\text{McNemar}}$, the
paired CI half-width, $\hat\theta$, $z_\theta$, and the MDE. Then say which of the seven do **not**
change, and why the answer for $z_\theta$ is the same as the answer for $z_{\text{McNemar}}$ even
though the two statistics have different variance structures.

Also still open from 08/05 and not answered: the two questions about your Agresti 1.8 in note 2 §4 —
which SE you intended in 1.8(b), and what the interval in 1.8(c) was centred on.
