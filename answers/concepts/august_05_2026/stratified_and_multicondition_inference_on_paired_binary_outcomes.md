# Stratified and multi-condition inference on paired binary outcomes: gold-label interaction, signal detection, and $k$ correlated conditions

**Date:** 2026-08-05
**Question it answers:** McNemar tests one paired condition against one baseline on one item set. What
tests answer (a) "does the intervention affect gold-yes and gold-no items differently?" and
(b) "do several conditions, each paired against the *same* baseline, differ from each other?"

Builds on
[`paired_accuracy_inference_and_minimum_detectable_effect.md`](./paired_accuracy_inference_and_minimum_detectable_effect.md)
(hereafter **note 1**) and
[`mcnemar_score_vs_wald_standard_errors_and_agresti_variance_identity.md`](./mcnemar_score_vs_wald_standard_errors_and_agresti_variance_identity.md)
(**note 2**). Notation is note 1's: per item, $B_i,S_i\in\{0,1\}$ are correctness at baseline and
under the intervention; $a,b,c,d$ are the cells of the $2\times2$ table with $b$ = baseline right /
steered wrong, $c$ = baseline wrong / steered right; $\hat\Delta=(c-b)/n$; $m=b+c$;
$\hat\pi_d=m/n$.

Every number below is invented by me or taken from Agresti's own tables. Section and problem
numbers are from Agresti, *An Introduction to Categorical Data Analysis*, 2nd ed. (2007), verified
against the text, not recalled. **No project result appears here.**

---

## 0. First: the thread left open yesterday

Note 2 §4 asked two things about your 1.8 and you have not answered either. Before the new
material:

1. In 1.8(b) you tested $H_0:\pi=0.5$ with $\text{SE}=\sqrt{\hat p(1-\hat p)/n}$. Which SE does
   Agresti §1.3.2 use for that test, and which did you intend?
2. In 1.8(c) your interval reads $[0.466,\ 0.5343]$ and mine reads $[0.2597,\ 0.3283]$. The
   half-widths agree; the centres do not. Yours is centred at $0.5$, mine at $\hat p=0.294$.
   **What were you centring on, and what was the interval a statement about?**

I am not going to say which is right until you answer, because the two candidate objects — a
confidence interval for $\pi$, and the acceptance region of the test of $\pi=0.5$ — are exactly the
score-versus-Wald distinction that §6 below rests on again. If the answer is "I meant the
acceptance region", the note-2 material lands differently than if the answer is "I meant a CI".

---

## 1. The compressed answer

Your two questions map onto two objects, and both are contrasts of things you already know how to
compute.

**Question A — "does steering affect gold-yes and gold-no differently?"**
Compute $\hat\Delta_{\text{yes}}$ and $\hat\Delta_{\text{no}}$ separately, each by note 1's machinery
on its own stratum. The gold-yes and gold-no items are **disjoint**, so the two estimates are
independent and

$$\operatorname{Var}\!\big(\hat\Delta_{\text{yes}}-\hat\Delta_{\text{no}}\big)
=\operatorname{Var}(\hat\Delta_{\text{yes}})+\operatorname{Var}(\hat\Delta_{\text{no}}).$$

Variances **add**. Test the interaction with a $z$ on that. This is Agresti problem 8.6(d) exactly,
with gold label playing the role of gender.

**Question B — "do the layer windows differ from each other?"**
The windows share the baseline and the item set, so the $\hat\Delta_j$ are **correlated**, and
variances do **not** add. Stack them into $\hat{\boldsymbol\Delta}\in\mathbb R^k$, estimate the
$k\times k$ covariance from the per-item difference scores, and use quadratic forms:
$W=n\hat{\boldsymbol\Delta}'\hat V^{-1}\hat{\boldsymbol\Delta}\sim\chi^2_k$ for "any window does
anything", and $Q_{\text{het}}=n(C\hat{\boldsymbol\Delta})'(C\hat VC')^{-1}(C\hat{\boldsymbol\Delta})
\sim\chi^2_{k-1}$ for "the windows differ from each other", with $C$ the differencing matrix. The
omnibus form of this for binary matched conditions has a name — **Cochran's Q** — and Agresti gives
it at §8.2.5, p. 252, as the generalized Cochran–Mantel–Haenszel test of §6.4.2 applied to a
$T\times2\times n$ table.

**And one fact worth more than either test.** With balanced strata, the two contrasts of
$(\hat\Delta_{\text{yes}},\hat\Delta_{\text{no}})$ are not arbitrary:

$$\underbrace{\tfrac12\big(\hat\Delta_{\text{yes}}+\hat\Delta_{\text{no}}\big)}_{\text{change in accuracy}},
\qquad
\underbrace{\tfrac12\big(\hat\Delta_{\text{yes}}-\hat\Delta_{\text{no}}\big)}_{\text{change in yes-rate}} .$$

The sum contrast *is* the accuracy change. The difference contrast *is* the yes-rate change. That is
the identity note 1 §10 said was "worth working out on paper once", and it means pooled accuracy is
structurally blind to the asymmetry you are asking about: it is one of the two coordinates, and you
are asking about the other one.

Everything below is the working.

---

## 2. Two disjoint strata: the variance algebra

### 2.1 Setup

Partition the $n$ items by gold label: $n_1$ gold-yes items, $n_0$ gold-no items, $n_1+n_0=n$,
$\lambda=n_1/n$. The partition is fixed by the benchmark, not random, and **no item is in both
strata**. Within stratum $s\in\{1,0\}$ build the paired correctness table $(a_s,b_s,c_s,d_s)$ and

$$\hat\Delta_s=\frac{c_s-b_s}{n_s},\qquad
\operatorname{Var}(\hat\Delta_s)=\frac{\pi_{d,s}-\Delta_s^2}{n_s},$$

which is note 1 §4 applied twice. Nothing new yet.

### 2.2 Why the covariance is exactly zero

Write $D_i=S_i-B_i\in\{-1,0,+1\}$, the per-item difference score. Then
$\hat\Delta_s=\frac{1}{n_s}\sum_{i\in s}D_i$, and

$$\operatorname{Cov}\big(\hat\Delta_{\text{yes}},\hat\Delta_{\text{no}}\big)
=\frac{1}{n_1n_0}\sum_{i\in\text{yes}}\sum_{j\in\text{no}}\operatorname{Cov}(D_i,D_j)=0,$$

because $i\ne j$ always — the index sets do not intersect — and items are independent. This is the
load-bearing fact for Question A and it is a fact about the *design*, not an approximation.

Therefore, for any constants $u,v$,

$$\boxed{\;\operatorname{Var}\big(u\hat\Delta_{\text{yes}}+v\hat\Delta_{\text{no}}\big)
=u^2\frac{\pi_{d,1}-\Delta_1^2}{n_1}+v^2\frac{\pi_{d,0}-\Delta_0^2}{n_0}\;}$$

In particular the **sum** contrast and the **difference** contrast have *the same* standard error:
$u=v=1$ and $u=1,v=-1$ give the same expression. What differs between the two questions is the size
of the effect you are hunting, not the precision you hunt it with.

### 2.3 Contrast with the within-stratum comparison

Inside one stratum, the comparison is baseline vs steered on the *same* items, and note 1 §4 gives

$$\operatorname{Var}(\hat\Delta_s)=\frac{1}{n_s}\Big[\sigma_{B,s}^2+\sigma_{S,s}^2
-2\operatorname{Cov}(B_i,S_i)\Big]
\;\approx\;\frac{2\sigma^2(1-\rho)}{n_s}.$$

Here the covariance **subtracts** — that is the entire benefit of pairing, and it is why a 3-point
accuracy change can be significant at $n=450$.

So the two comparisons are structurally opposite:

| comparison | shares items? | covariance term | effect on variance |
|---|---|---|---|
| baseline vs steered, within a stratum | yes, all of them | $-2\operatorname{Cov}$ | shrinks it (when $\rho>0$) |
| gold-yes effect vs gold-no effect | no, none of them | $0$ | variances add, full stop |

You cannot pair across the strata and there is nothing to recover. That is the price of the
interaction question, and §5 quantifies it.

### 2.4 The pooled effect, and a free lunch

The pooled estimate decomposes exactly:

$$\hat\Delta_{\text{pool}}=\frac{(c_1-b_1)+(c_0-b_0)}{n}
=\lambda\hat\Delta_{\text{yes}}+(1-\lambda)\hat\Delta_{\text{no}},$$

$$\operatorname{Var}(\hat\Delta_{\text{pool}})
=\lambda^2\operatorname{Var}(\hat\Delta_{\text{yes}})+(1-\lambda)^2\operatorname{Var}(\hat\Delta_{\text{no}}).$$

This is **not** in general equal to the unstratified $(\hat\pi_d-\hat\Delta^2)/n$ from note 1. The
unstratified formula treats the gold label as random; the stratified one conditions on it, as the
design says you should. By the law of total variance,

$$\underbrace{\operatorname{Var}(D_i)}_{\text{unstratified}}
=\underbrace{E_s\!\left[\operatorname{Var}(D_i\mid s)\right]}_{\text{stratified, within}}
+\underbrace{\operatorname{Var}_s\!\left(E[D_i\mid s]\right)}_{\text{between-stratum}},$$

and the between-stratum term is $\lambda(1-\lambda)(\Delta_1-\Delta_0)^2$ — the squared interaction.
So **stratifying tightens the main-effect interval by exactly the amount the strata disagree.** When
the two strata move together it buys nothing; when they move in opposite directions it buys a lot.
Numbers in §4.4.

---

## 3. What the two contrasts mean: the signal-detection parameterization

### 3.1 The reparameterization

On a yes/no benchmark, per-item correctness and per-item answer are the same information read
through the gold label. Define, for a fixed condition,

$$H=\Pr[\text{answers yes}\mid\text{gold yes}]\ \ (\textbf{hit rate}),\qquad
F=\Pr[\text{answers yes}\mid\text{gold no}]\ \ (\textbf{false-alarm rate}).$$

Then, mechanically:

$$\text{accuracy on gold-yes}=H,\qquad \text{accuracy on gold-no}=1-F,$$
$$\text{pooled accuracy}=\lambda H+(1-\lambda)(1-F),\qquad
\text{yes-rate}=\lambda H+(1-\lambda)F .$$

This is Agresti §2.1.3's sensitivity/specificity vocabulary: $H$ is sensitivity, $1-F$ is
specificity. Differencing between conditions,

$$\Delta_{\text{yes}}=\Delta H,\qquad \Delta_{\text{no}}=-\Delta F.$$

The minus sign is the whole story. **Accuracy on the gold-no stratum runs backwards in $F$**, so a
change that pushes the model toward "yes" shows up as $+$ in one stratum and $-$ in the other.

### 3.2 The two contrasts, at $\lambda=1/2$

$$\Delta_{\text{acc}}=\tfrac12(\Delta_{\text{yes}}+\Delta_{\text{no}})=\tfrac12(\Delta H-\Delta F)
=\tfrac12\,\Delta J,$$
$$\Delta_{\text{yes-rate}}=\tfrac12(\Delta_{\text{yes}}-\Delta_{\text{no}})=\tfrac12(\Delta H+\Delta F),$$

where $J=H-F$ is Youden's index — the vertical distance from the operating point to the chance
diagonal in ROC space. So:

- The **sum** contrast is (half) the change in $H-F$: how far the operating point moved *away from
  the diagonal*. This is the sensitivity-flavoured coordinate.
- The **difference** contrast is (half) the change in $H+F$: how far the operating point slid
  *along* the ROC curve. This is the criterion-flavoured coordinate.

They are the two orthogonal directions in $(H,F)$ space, rotated 45°.

### 3.3 The two mechanisms, and what each does to the strata

Take the equal-variance Gaussian signal-detection model as a device for making the two hypotheses
concrete: $H=\Phi(d'/2-c)$, $F=\Phi(-d'/2-c)$, with $d'$ sensitivity and $c$ criterion.

**Pure criterion shift** ($c$ decreases, $d'$ fixed): both $\Phi$ arguments increase, so $\Delta H>0$
and $\Delta F>0$. Then $\Delta_{\text{yes}}=\Delta H>0$ and $\Delta_{\text{no}}=-\Delta F<0$: **the
strata move in opposite directions.** In the sum they cancel — exactly when $\Delta H=\Delta F$,
approximately otherwise. In the difference they reinforce.

**Pure sensitivity change** ($d'$ increases, $c$ fixed at 0): $\Delta H>0$ and $\Delta F<0$, so
$\Delta_{\text{yes}}>0$ and $\Delta_{\text{no}}=-\Delta F>0$: **the strata move together.** In the sum
they reinforce; in the difference they cancel.

| | $\Delta H$ | $\Delta F$ | $\Delta_{\text{yes}}$ | $\Delta_{\text{no}}$ | sum contrast (accuracy) | difference contrast (yes-rate) |
|---|---|---|---|---|---|---|
| criterion loosens toward "yes" | $+$ | $+$ | $+$ | $-$ | $\approx 0$ | large $+$ |
| criterion tightens toward "no" | $-$ | $-$ | $-$ | $+$ | $\approx 0$ | large $-$ |
| sensitivity up | $+$ | $-$ | $+$ | $+$ | large $+$ | $\approx 0$ |
| sensitivity down | $-$ | $+$ | $-$ | $-$ | large $-$ | $\approx 0$ |

### 3.4 Why pooled accuracy cannot separate them

The state of the system after the intervention is a **two-dimensional** displacement
$(\Delta H,\Delta F)$. Pooled accuracy is the single linear functional

$$\Delta_{\text{acc}}=\lambda\,\Delta H-(1-\lambda)\,\Delta F,$$

i.e. one projection of a 2-D vector onto one direction. Its level sets are lines: at $\lambda=1/2$,
every $(\Delta H,\Delta F)$ with $\Delta H-\Delta F=$ const gives the *same* accuracy change. A
criterion shift with $\Delta H=\Delta F=0.14$ and total inertness with $\Delta H=\Delta F=0$ lie on
the same level line. No amount of $n$ recovers the lost coordinate, because it was never measured —
this is an identification failure, not a precision failure. That is a stronger statement than
"accuracy is noisy". It is why the answer to "does steering affect gold-yes and gold-no items
differently" cannot be extracted from a pooled accuracy number, however many items you run.

**The picture.** ROC unit square, $F$ on the $x$ axis, $H$ on the $y$ axis, chance diagonal
$H=F$ drawn as a thin dashed line from $(0,0)$ to $(1,1)$. Put the baseline operating point at
$(0.207, 0.783)$. Draw two arrows from it:

- one to $(0.350, 0.923)$ — *parallel to the diagonal*, i.e. sliding along an iso-sensitivity
  curve. This is the criterion shift.
- one to $(0.133, 0.850)$ — *up and to the left, perpendicular-ish to the diagonal*, away from it.
  This is the sensitivity change.

Now overlay the iso-accuracy lines: at $\lambda=1/2$ these are the lines of slope $+1$, i.e.
*parallel to the diagonal*. The criterion arrow lies **along** an iso-accuracy line; the sensitivity
arrow **crosses** them. That single picture is the whole section: accuracy measures displacement
perpendicular to the diagonal and is exactly blind to displacement along it.

### 3.5 The loose end from note 1 §10, closed

Note 1 §10 observed that the correctness table and the yes-answer table have the *same* discordant
items but different $b/c$ splits, and left the reason open. Here it is: on a gold-yes item,
"answered yes" and "correct" are the same event, so a flip contributes with the same sign to both
tables; on a gold-no item they are complementary, so a flip contributes with **opposite** signs.
Hence

$$\hat\Delta_{\text{correct}}\ \text{sums the strata},\qquad
\hat\Delta_{\text{yes-rate}}\ \text{differences them},$$

which is §3.2. One practical consequence: **you can run the interaction test without ever
stratifying** — build the yes-answer $2\times2$ table on all $n$ items, ignoring gold labels, and
McNemar it. At $\lambda=1/2$ that is the difference contrast up to the factor 2. Stratifying is
still worth doing, because it gives you the two stratum effects separately and their intervals,
and because at $\lambda\ne1/2$ the correspondence picks up weights.

---

## 4. Worked synthetic example

$n=600$ invented items, $n_1=n_0=300$, $\lambda=1/2$. Three cases, same baseline behaviour
throughout (473/600 correct at baseline), engineered to move in different ways.

### 4.1 Case *criterion shift*: strata move in opposite directions, pooled accuracy does not move

**gold-yes stratum** ($n_1=300$; correct $=$ answered yes)

| | steered right | steered wrong | total |
|---|---|---|---|
| baseline right | $a_1=232$ | $b_1=3$ | 235 |
| baseline wrong | $c_1=45$ | $d_1=20$ | 65 |
| total | 277 | 23 | 300 |

**gold-no stratum** ($n_0=300$; correct $=$ answered no)

| | steered right | steered wrong | total |
|---|---|---|---|
| baseline right | $a_0=190$ | $b_0=48$ | 238 |
| baseline wrong | $c_0=5$ | $d_0=57$ | 62 |
| total | 195 | 105 | 300 |

**Per stratum.**

$$\hat\Delta_{\text{yes}}=\frac{45-3}{300}=+0.14000,\qquad m_1=48,\qquad
z_1=\frac{42}{\sqrt{48}}=+6.062,\ p=1.3\times10^{-9}$$
$$\hat\Delta_{\text{no}}=\frac{5-48}{300}=-0.14333,\qquad m_0=53,\qquad
z_0=\frac{-43}{\sqrt{53}}=-5.906,\ p=3.5\times10^{-9}$$

Both enormous, opposite signs.

**Pooled.** $b=b_1+b_0=51$, $c=c_1+c_0=50$, $m=101$:

$$\hat\Delta_{\text{pool}}=\frac{50-51}{600}=-0.001667,\qquad
z=\frac{-1}{\sqrt{101}}=-0.0995,\qquad p=0.92 .$$

**A 0.17-point move and a McNemar $p$ of 0.92 — while each half of the benchmark moved 14 points
with $|z|\approx6$.** This is the case the whole note exists for. Nothing is wrong with the pooled
test; it is answering the sum contrast, and the sum contrast really is zero here.

**Interaction.** Using §2.2 with the Wald (estimate-plugged-in) variance of note 2 §2.1:

$$\widehat{\operatorname{Var}}(\hat\Delta_{\text{yes}})=\frac{0.16-0.14^2}{300}
=\frac{0.140400}{300}=4.6800\times10^{-4},\quad \text{SE}=0.021633$$
$$\widehat{\operatorname{Var}}(\hat\Delta_{\text{no}})=\frac{0.176667-0.143333^2}{300}
=\frac{0.156122}{300}=5.2041\times10^{-4},\quad \text{SE}=0.022812$$
$$\hat\theta=\hat\Delta_{\text{yes}}-\hat\Delta_{\text{no}}=+0.283333,$$
$$\text{SE}(\hat\theta)=\sqrt{4.6800\times10^{-4}+5.2041\times10^{-4}}=\sqrt{9.8841\times10^{-4}}=0.031439$$

$$z_\theta=\frac{0.283333}{0.031439}=9.012,\qquad
\text{95\% CI}=0.283333\pm1.96(0.031439)=[0.2217,\ 0.3450].$$

**Signal-detection reading.** $H_B=235/300=0.7833\to H_S=277/300=0.9233$ ($\Delta H=+0.1400$);
$F_B=62/300=0.2067\to F_S=105/300=0.3500$ ($\Delta F=+0.1433$). Both rates up: criterion loosened
toward "yes". Youden $J$: $0.5767\to0.5733$, $\Delta J=-0.0033=2\Delta_{\text{acc}}$ ✓.
Yes-rate: $297/600=0.4950\to382/600=0.6367$, change $+0.14167=\tfrac12\hat\theta$ ✓.

### 4.2 Case *sensitivity change*: strata move together, interaction null

**gold-yes:** $a_1=230,\ b_1=5,\ c_1=25,\ d_1=40$. **gold-no:** $a_0=232,\ b_0=6,\ c_0=28,\ d_0=34$.

$$\hat\Delta_{\text{yes}}=\frac{20}{300}=+0.066667,\qquad \hat\Delta_{\text{no}}=\frac{22}{300}=+0.073333$$

Pooled: $b=11$, $c=53$, $m=64$, $\hat\Delta_{\text{pool}}=42/600=+0.07000$,
$z=42/\sqrt{64}=5.250$, $p=1.5\times10^{-7}$. Strongly significant.

Interaction: $\hat\theta=-0.006667$;
$\widehat{\operatorname{Var}}_{\text{yes}}=(0.100000-0.004444)/300=3.18519\times10^{-4}$,
$\widehat{\operatorname{Var}}_{\text{no}}=(0.113333-0.005378)/300=3.59852\times10^{-4}$,
$\text{SE}(\hat\theta)=\sqrt{6.78371\times10^{-4}}=0.026046$,

$$z_\theta=\frac{-0.006667}{0.026046}=-0.256,\qquad p=0.80,\qquad
\text{95\% CI}=[-0.0577,\ +0.0444].$$

Signal detection: $H:0.7833\to0.8500$ ($+0.0667$), $F:0.2067\to0.1333$ ($-0.0733$). $H$ up, $F$
down — sensitivity increase. $\Delta J=+0.1400=2(0.07)$ ✓. Yes-rate $0.4950\to0.4917$, essentially
flat ✓.

### 4.3 Case *both*: pooled accuracy moves **and** the strata are asymmetric

**gold-yes:** $a_1=231,\ b_1=4,\ c_1=40,\ d_1=25$. **gold-no:** $a_0=218,\ b_0=20,\ c_0=14,\ d_0=48$.

$\hat\Delta_{\text{yes}}=36/300=+0.12$, $\hat\Delta_{\text{no}}=-6/300=-0.02$.
Pooled: $b=24$, $c=54$, $m=78$, $\hat\Delta_{\text{pool}}=+0.05$, $z=30/\sqrt{78}=3.397$, $p=0.00068$.
Interaction: $\hat\theta=+0.14$; $\widehat{\operatorname{Var}}_{\text{yes}}=(0.146667-0.0144)/300
=4.4089\times10^{-4}$, $\widehat{\operatorname{Var}}_{\text{no}}=(0.113333-0.0004)/300=3.7644\times10^{-4}$,
$\text{SE}=0.028589$, $z_\theta=4.897$, $p=9.7\times10^{-7}$.

Both coordinates are non-null and both are detected. $\Delta H=+0.12$, $\Delta F=+0.02$: sensitivity
up *and* criterion loosened. The point of including this case: the two contrasts are separately
identified and a significant main effect does not imply a null interaction or vice versa.

### 4.4 The free lunch from §2.4, checked

| | stratified SE of $\hat\Delta_{\text{pool}}$ | unstratified SE (note 1 §4) | narrowing |
|---|---|---|---|
| criterion-shift case | $\sqrt{0.25(4.680+5.2041)\times10^{-4}}=0.015719$ | $\frac{1}{600}\sqrt{101-1/600}=0.016750$ | 6.2% |
| sensitivity case | $\sqrt{0.25(3.18519+3.59852)\times10^{-4}}=0.013023$ | $\frac{1}{600}\sqrt{64-42^2/600}=0.013024$ | 0.005% |

Exactly as §2.4 predicts. When the strata disagree violently, conditioning on the (fixed by design)
stratum sizes removes a real variance component; when they agree, there is nothing to remove.

### 4.5 The summary table

| | crit. shift | sens. change | both |
|---|---|---|---|
| $\hat\Delta_{\text{yes}}$ | $+0.1400$ | $+0.0667$ | $+0.1200$ |
| $\hat\Delta_{\text{no}}$ | $-0.1433$ | $+0.0733$ | $-0.0200$ |
| $\hat\Delta_{\text{pool}}$ (accuracy) | $-0.0017$ | $+0.0700$ | $+0.0500$ |
| pooled McNemar $z$ | $-0.10$ | $5.25$ | $3.40$ |
| $\hat\theta$ (interaction) | $+0.2833$ | $-0.0067$ | $+0.1400$ |
| $z_\theta$ | $9.01$ | $-0.26$ | $4.90$ |
| $(\Delta H,\Delta F)$ | $(+.140,+.143)$ | $(+.067,-.073)$ | $(+.120,+.020)$ |

Read the second and fifth rows together. They are the two coordinates, and they move independently.

---

## 5. What the interaction costs in power

Assume balanced strata $n_1=n_0=n/2$ and a common discordance rate $\pi_d$ in both. Under $H_0$ the
$\Delta^2$ terms drop (note 2 §2.1), so

$$\operatorname{Var}(\hat\Delta_s)=\frac{\pi_d}{n/2}=\frac{2\pi_d}{n},\qquad
\operatorname{Var}(\hat\Delta_{\text{pool}})=\tfrac14\!\left(\tfrac{2\pi_d}{n}+\tfrac{2\pi_d}{n}\right)=\frac{\pi_d}{n},$$
$$\operatorname{Var}(\hat\theta)=\frac{2\pi_d}{n}+\frac{2\pi_d}{n}=\frac{4\pi_d}{n}.$$

$$\boxed{\ \text{SE}(\hat\theta)=2\,\text{SE}(\hat\Delta_{\text{pool}})
=\sqrt2\,\text{SE}(\hat\Delta_s)\ }$$

**A factor of 2 in SE, hence a factor of 4 in $n$**, to detect an interaction of the same numerical
size as a main effect. This is the standard "interactions need four times the sample" fact, and here
it comes from two sources multiplying: each stratum has half the items, and the two errors add
rather than cancel.

At $n=600$, $\pi_d=0.10$, $\alpha=0.05$ two-sided, power $0.80$ (multiplier
$z_{0.975}+z_{0.80}=2.8016$, note 1 §7):

| quantity | SE | MDE |
|---|---|---|
| $\hat\Delta_s$ (one stratum, $n_s=300$) | $\sqrt{0.10/300}=0.018257$ | $5.12$ pts |
| $\hat\Delta_{\text{pool}}$ ($n=600$) | $\sqrt{0.10/600}=0.012910$ | $3.62$ pts |
| $\hat\theta=\hat\Delta_{\text{yes}}-\hat\Delta_{\text{no}}$ | $2\sqrt{0.10/600}=0.025820$ | $7.23$ pts |

**One subtlety that is easy to get backwards, so state it carefully.** "The interaction test is
weaker" is true on the $\theta$ scale and false on the per-stratum scale:

- To detect $\theta$ of a given size you need 4× the $n$ you would need to detect a pooled
  $\bar\Delta$ of that same size. (Row 2 vs row 3 above: 3.62 vs 7.23.)
- But an asymmetry of $\pm\delta$ (gold-yes $+\delta$, gold-no $-\delta$) has $\theta=2\delta$, so
  its $z$ is $2\delta/(2\sqrt{\pi_d/n})=\delta/\sqrt{\pi_d/n}$ — **identical** to the $z$ for a main
  effect of $+\delta$ in both strata. Detecting "both up 3.6 points" and detecting "one up 3.6, the
  other down 3.6" are equally hard.

Both statements are consequences of the same algebra; which one is relevant depends on whether the
effect you would care about is stated as a $\theta$ or as a per-stratum shift. That is a SESOI
question (note 1 §8), i.e. yours, not the statistics'.

---

## 6. The model-based route: an interaction coefficient instead of a contrast of tests

The contrasts above are fine and fully sufficient. The model route buys you (i) one parameter with
one $p$-value instead of a hand-assembled contrast, (ii) a place to put covariates, and (iii) an
extension to $k$ conditions without recombination. Agresti's chapter 8 §8.2 is the matched-pairs
entry point.

### 6.1 Marginal (population-averaged) model — the identity link makes the identification exact

Agresti §8.2.1, p. 247 writes the matched-pairs marginal model with an identity link as
$P(Y_1=1)=\alpha+\delta$, $P(Y_2=1)=\alpha$, so that $\delta$ is exactly the difference in marginal
proportions and $H_0:\delta=0$ is McNemar's null. Extend it with the stratum indicator
$g_i\in\{0,1\}$ (1 = gold-yes) and the condition indicator $t\in\{0,1\}$ (1 = steered):

$$P(Y_{it}=1)=\alpha+\beta t+\gamma g_i+\theta\,(t\,g_i).$$

Read off the four cells:

| | $t=0$ | $t=1$ | difference |
|---|---|---|---|
| $g=0$ (gold-no) | $\alpha$ | $\alpha+\beta$ | $\beta$ |
| $g=1$ (gold-yes) | $\alpha+\gamma$ | $\alpha+\beta+\gamma+\theta$ | $\beta+\theta$ |

So $\beta=\Delta_{\text{no}}$ and $\beta+\theta=\Delta_{\text{yes}}$, hence

$$\boxed{\ \theta=\Delta_{\text{yes}}-\Delta_{\text{no}}\ }$$

**The interaction coefficient *is* the contrast of §2**, not an analogue of it. Fitted by moment
methods / GEE with an independence working correlation, $\hat\theta$ is numerically the $\hat\theta$
of §4. The *standard error* agrees with §2.2 only if you use the robust (sandwich) variance, because
the naive model-based SE assumes the two observations on an item are independent, which is precisely
the assumption pairing exists to avoid. That machinery is Agresti chapter 9 (§9.2, GEE); the intro
text presents GEE as methodology and software rather than deriving the sandwich, and I am naming
that honestly rather than pretending the derivation is in there.

### 6.2 Conditional (subject-specific) model — a closed form worth having

Agresti model (8.4), p. 249, puts an item-specific intercept in:

$$\operatorname{logit}P(Y_{it}=1)=\alpha_i+\beta t .$$

The $\alpha_i$ absorb per-item difficulty. There are $n$ of them, so ordinary ML is inconsistent;
conditional ML eliminates them, and Agresti reports (p. 250) the closed form

$$\hat\beta_{\text{cond}}=\log\frac{c}{b},\qquad
\widehat{\operatorname{Var}}(\hat\beta_{\text{cond}})=\frac1b+\frac1c$$

(the variance is problem 8.11; Agresti writes it as $n_{12}/n_{21}$ in his row/column orientation —
note 2 §0.5 has the bridge). Note that $b$ and $c$ are the only inputs: the same "concordant pairs
cancel" fact as McNemar, now visible as the conditional likelihood dropping every item whose two
responses agree.

Add the stratum interaction. Since $g_i$ is constant within an item, it is absorbed by $\alpha_i$
and only the interaction survives:

$$\operatorname{logit}P(Y_{it}=1)=\alpha_i+\beta t+\delta\,(t\,g_i).$$

Conditional ML factorises over the two strata (disjoint items again), giving
$\hat\beta=\log(c_0/b_0)$ and $\hat\beta+\hat\delta=\log(c_1/b_1)$, so

$$\boxed{\ \hat\delta=\log\frac{c_1/b_1}{c_0/b_0}=\log\frac{c_1b_0}{b_1c_0},\qquad
\widehat{\operatorname{Var}}(\hat\delta)=\frac{1}{b_1}+\frac{1}{c_1}+\frac{1}{b_0}+\frac{1}{c_0}\ }$$

A ratio of the two discordant-flip ratios, and a variance you can do in your head. On §4's cases:

| case | $\hat\delta=\log\frac{c_1b_0}{b_1c_0}$ | SE | $z$ |
|---|---|---|---|
| criterion shift | $\log\frac{45\cdot48}{3\cdot5}=\log144=4.970$ | $\sqrt{\tfrac13+\tfrac1{45}+\tfrac1{48}+\tfrac15}=0.7592$ | $6.55$ |
| sensitivity change | $\log\frac{25\cdot6}{5\cdot28}=\log1.0714=0.0690$ | $\sqrt{\tfrac15+\tfrac1{25}+\tfrac16+\tfrac1{28}}=0.6651$ | $0.10$ |
| both | $\log\frac{40\cdot20}{4\cdot14}=\log14.286=2.659$ | $\sqrt{\tfrac14+\tfrac1{40}+\tfrac1{20}+\tfrac1{14}}=0.6296$ | $4.22$ |

Same three conclusions as §4. But look at the middle row against §4.2: on the identity scale
$\hat\theta=-0.0067$ (gold-no moved *more*), and on the log-odds scale $\hat\delta=+0.069$ (gold-yes
moved more). **The interaction changed sign under a change of link.** Both are null here so nothing
breaks, but the lesson is general and it matters:

> "Is there an interaction?" is **not a scale-free question.** An interaction can be zero on the
> difference-of-proportions scale and non-zero on the log-odds scale, and vice versa. Which scale
> you test on is a convention you adopt and state.

There is a related non-equivalence: the conditional $\hat\beta$ and the marginal $\hat\beta$ of
§6.1 estimate *different parameters*, not the same one two ways — the odds ratio is not
collapsible. Agresti §10.1.4 and §10.2.7 are where that is discussed; the third route, treating
$\alpha_i$ as random effects (Agresti §10.1.2, "A Logistic GLMM for Binary Matched Pairs", p. 299),
gives a third parameterisation again.

---

## 7. $k$ conditions against one shared baseline

### 7.1 The multivariate paired setup

Item $i=1,\dots,n$; baseline correctness $B_i$; window $j=1,\dots,k$ correctness $S_{ij}$. Define the
per-item difference-score **vector**

$$\mathbf D_i=(D_{i1},\dots,D_{ik})',\qquad D_{ij}=S_{ij}-B_i\in\{-1,0,+1\}.$$

Then $\hat{\boldsymbol\Delta}=\bar{\mathbf D}$, and by the multivariate CLT

$$\boxed{\ \hat{\boldsymbol\Delta}\ \dot\sim\ N_k\!\left(\boldsymbol\Delta,\ \frac{V}{n}\right),
\qquad V=\operatorname{Cov}(\mathbf D_i)\ }$$

$V$ is estimated by the ordinary sample covariance matrix of the $n$ vectors $\mathbf D_i$ — no new
machinery, and it requires only the per-item correctness records joined on item id (note 1 §10's
data requirement, now needed for $k+1$ conditions rather than 2).

Diagonal: $V_{jj}=\pi_{d,j}-\Delta_j^2$, i.e. exactly note 1 §4 per window ✓.

Off-diagonal, expanded:

$$V_{jl}=\operatorname{Cov}(S_{ij},S_{il})-\operatorname{Cov}(S_{ij},B_i)-\operatorname{Cov}(S_{il},B_i)+\operatorname{Var}(B_i).$$

The last term is $+\operatorname{Var}(B_i)$ with a plus sign and it is shared by **every** pair
$(j,l)$: the same baseline noise is inside every $\hat\Delta_j$. That is the structural reason the
windows are positively correlated, and it does not go away by running more windows.

### 7.2 Two quadratic forms

**Omnibus — "does any window do anything?"** $H_0:\boldsymbol\Delta=\mathbf 0$:

$$W=n\,\hat{\boldsymbol\Delta}'\hat V^{-1}\hat{\boldsymbol\Delta}\ \sim\ \chi^2_k .$$

This is the Wald form. Its score sibling, for binary matched conditions, is **Cochran's Q**: Agresti
§8.2.5, p. 252, "For a matched set of $T$ observations and a response scale having $I=2$, a
generalized CMH test of conditional independence (Section 6.4.2) can be applied to a $T\times2\times
n$ table. The test statistic for that case is sometimes called Cochran's Q." So the $k$-condition
omnibus **is** in your book — it is reached through §6.4.2 (generalized Cochran–Mantel–Haenszel
tests, p. 194) rather than given its own section, and problem 8.12 is the worked instance (matched
*triplets* — alcohol, cigarettes, marijuana — with part (b) asking you to set up exactly the
three-way table that makes the generalized CMH procedure apply). §8.3.1, p. 253, gives the closely
related score statistic $W_0=n\,\mathbf d'\hat V_0^{-1}\mathbf d$ for marginal homogeneity of an
$I\times I$ square table, with the note that at $I=2$ it collapses to McNemar's $\chi^2$ — same
quadratic-form structure, transposed problem.

Note carefully what Cochran's Q's null is: *all* $T$ marginal proportions equal, i.e. baseline and
every window alike. Rejecting it does not say the windows differ from each other.

**Heterogeneity — "do the windows differ from each other?"** $H_0:\Delta_1=\dots=\Delta_k$. Take the
$(k-1)\times k$ differencing matrix

$$C=\begin{pmatrix}1&-1&0&\cdots\\0&1&-1&\cdots\\ &&\ddots\end{pmatrix},$$

so $H_0$ is $C\boldsymbol\Delta=\mathbf0$, and

$$Q_{\text{het}}=n\,(C\hat{\boldsymbol\Delta})'\big(C\hat VC'\big)^{-1}(C\hat{\boldsymbol\Delta})
\ \sim\ \chi^2_{k-1}.$$

This is the answer to "is the fluctuation across windows more than sampling error?" — direction 5 of
your question. It is the correct analogue of a heterogeneity test; it is **not** the meta-analysis
heterogeneity statistic $\sum w_j(\hat\Delta_j-\bar\Delta)^2$, which assumes independent studies and
is wrong here.

### 7.3 Synthetic numbers

Invented. $k=3$ windows, named by what they are: `early_window`, `mid_window`, `late_window`.
$n=600$ items, shared baseline.

$$\hat{\boldsymbol\Delta}=(0.050,\ 0.030,\ -0.010)',\qquad
\hat V=\begin{pmatrix}0.120&0.070&0.045\\0.070&0.098&0.050\\0.045&0.050&0.150\end{pmatrix}$$

Marginal SEs $\sqrt{V_{jj}/n}$: $0.014142$, $0.012780$, $0.015811$. Correlations:
$\operatorname{corr}(\hat\Delta_1,\hat\Delta_2)=0.070/\sqrt{0.120\cdot0.098}=0.646$,
$(\hat\Delta_1,\hat\Delta_3)=0.335$, $(\hat\Delta_2,\hat\Delta_3)=0.412$. Strongly positive, as §7.1
says they must be.

**Omnibus.** $\det\hat V=8.4555\times10^{-4}$ and

$$\hat V^{-1}=\begin{pmatrix}14.429&-9.757&-1.076\\-9.757&18.893&-3.371\\-1.076&-3.371&8.113\end{pmatrix},
\qquad \hat V^{-1}\hat{\boldsymbol\Delta}=\begin{pmatrix}0.43948\\0.11265\\-0.23606\end{pmatrix}$$
$$\hat{\boldsymbol\Delta}'\hat V^{-1}\hat{\boldsymbol\Delta}=0.0277138,\qquad
W=600(0.0277138)=16.63\ \text{on}\ \chi^2_3,\qquad p=8.4\times10^{-4}.$$

**Heterogeneity.** $C\hat{\boldsymbol\Delta}=(0.020,\ 0.040)'$ and

$$C\hat VC'=\begin{pmatrix}0.078&-0.023\\-0.023&0.148\end{pmatrix},\qquad
(C\hat VC')^{-1}=\begin{pmatrix}13.436&2.088\\2.088&7.081\end{pmatrix}$$
$$Q_{\text{het}}=600(0.0200455)=12.03\ \text{on}\ \chi^2_2,\qquad p=0.0024 .$$

So: something is going on, and the windows are not interchangeable.

**Why the correlation must not be ignored.** Compare `early_window` with `mid_window`:
$\hat\Delta_1-\hat\Delta_2=0.020$.

| treatment | SE | $z$ | $p$ |
|---|---|---|---|
| correct (uses $V_{11}+V_{22}-2V_{12}=0.078$) | $\sqrt{0.078/600}=0.011402$ | $1.754$ | $0.079$ |
| naive, treating the two windows as independent ($V_{11}+V_{22}=0.218$) | $\sqrt{0.218/600}=0.019062$ | $1.049$ | $0.294$ |

The naive SE is **1.67× too large**. The direction matters and is worth memorising: because the
correlation induced by the shared baseline is *positive*, treating windows as independent is
**conservative** for window-vs-window comparisons — you will under-detect differences between
windows. It is the mirror image of note 1 §3, where ignoring item-level pairing was also
conservative, for the same reason (a positive covariance you threw away).

### 7.4 The full $k\times2$ array

Your two questions compose. The full parameter set is $\{\Delta_{js}\}$ for window $j=1..k$ and
stratum $s\in\{\text{yes},\text{no}\}$, a $k\times2$ array, with covariance structure

$$\hat V=\begin{pmatrix}\hat V^{(\text{yes})}&0\\0&\hat V^{(\text{no})}\end{pmatrix}
\quad\text{— block diagonal, because the strata share no items,}$$

and each block dense $k\times k$, because within a stratum the windows share the baseline. Then:

- window main effects: contrasts within a block;
- stratum main effect (= yes-rate shift): $\sum_j$ weights on $\Delta_{j,\text{yes}}-\Delta_{j,\text{no}}$;
- window × stratum interaction ("do different windows shift the criterion by different amounts?"):
  a $(k-1)$-dimensional contrast, tested by the same quadratic form on $C$ applied within-and-across
  blocks, on $\chi^2_{k-1}$.

Every one of these is $n(C\hat{\boldsymbol\Delta})'(C\hat VC')^{-1}(C\hat{\boldsymbol\Delta})$ with a
different $C$. The single most useful thing to take from §7 is that once you have $\hat{\boldsymbol\Delta}$
and $\hat V$, every question you can phrase as a linear contrast is one line of linear algebra, and
you never have to ask again which test applies.

---

## 8. Omnibus first, then post-hoc — and why

You have $k$ windows, 2 strata, and 2 contrasts, plus models. The multiplicity problem of note 1 §9
is now much worse, and one specific failure mode is the one you are closest to:

> Looking at the plot, picking the window with the largest apparent effect, and quoting its McNemar
> $p$.

That is note 1 §8's selection problem with $k$ in place of the number of configurations, and the
inflation is the same: conditional on selection, the estimate is biased upward, worst when power is
lowest. With $k$ windows and a global null, the expected maximum $|z|$ over $k$ correlated tests
grows roughly like $\sqrt{2\log k}$ for independent tests and more slowly under positive
correlation — either way, "the best of 8 windows had $p=0.03$" is not a $p=0.03$ finding.

The omnibus-then-post-hoc structure exists precisely to buy back a calibrated statement:

1. **Omnibus.** Test $W$ (or Cochran's Q) at $\alpha$. One test, $k$ df, no selection. If it does
   not reject, you have no licence to name a best window — that is the whole function of the gate.
2. **Post-hoc, if the omnibus rejects.** Then examine individual windows and pairwise contrasts,
   with a correction (Bonferroni over the $\binom{k}{2}$ or over the $k$ vs-baseline comparisons; or
   FDR if you prefer, per note 1 §9). The correlation among windows makes Bonferroni conservative
   here, and that is a known cost, not a bug.

The gate is not a magic guarantee — a two-stage procedure controls the error rate of the *family* it
was defined over, and "which windows I would have looked at" is still a judgement you make in
advance. Pre-specifying a primary window and treating the rest as secondary, as in note 1 §9 option
2, remains available and is often cleaner.

One additional structural point specific to layer windows: **they are ordered**. If a monotone or
single-peaked pattern in layer index is the hypothesis, a 1-df trend contrast (a linear or
quadratic $C$ against layer index, in the sense of Agresti §2.5's ordinal trend tests) is far more
powerful than the $(k-1)$-df omnibus, because it spends its degrees of freedom on the shape you
expect. That contrast must be chosen *before* seeing the plot, or it is selection again.

---

## 9. The plots

### 9.1 Two-strata difference-CI panel

Extends note 1 §6's right-hand panel. One panel.

- $x$ axis, three categorical ticks: `gold_yes`, `gold_no`, then — after a vertical dotted separator
  — `yes − no`.
- $y$ axis: change in accuracy, points, from $-0.20$ to $+0.36$.
- A **solid** horizontal reference line at $0$.
- A **dashed** horizontal line at the pooled $\hat\Delta_{\text{pool}}=-0.0017$, with a thin shaded
  band across the whole panel for its 95% CI $[-0.0345,\ +0.0312]$. Label it "pooled accuracy
  change".
- Point + 95% bar at `gold_yes`: $+0.1400$, $[0.0976,\ 0.1824]$.
- Point + 95% bar at `gold_no`: $-0.1433$, $[-0.1880,\ -0.0986]$.
- Point + 95% bar at `yes − no`: $+0.2833$, $[0.2217,\ 0.3450]$.

What the picture shows in one glance: the pooled band hugs zero, and the two stratum points sit far
outside it, in opposite directions, with intervals that do not come near the band. The separator
matters — the third tick is a contrast, on a different scale from the first two, and putting it in
the same axis without the separator invites reading it as a third stratum.

Draw the same panel for the sensitivity case (§4.2) beside it: there the two stratum points sit
*together*, both above zero, the pooled band sits with them, and the `yes − no` bar straddles zero.
The two panels side by side are the criterion/sensitivity distinction in one figure.

### 9.2 $k$ windows against a shared baseline

Two panels, side by side.

**Left — forest plot.** $y$ axis: window names, **ordered by layer index**, top to bottom
(`early_window`, `mid_window`, `late_window`). Ordering by effect size is the selection trap made
visual and should not be done. $x$ axis: $\hat\Delta_j$ in points, solid vertical line at $0$. One
point + 95% CI per window from $\sqrt{\hat V_{jj}/n}$: $0.050\pm0.0277$, $0.030\pm0.0251$,
$-0.010\pm0.0310$. A dashed vertical line at the pooled/precision-weighted mean. Annotate in the
corner: $W=16.63$ ($\chi^2_3$, $p=8.4\times10^{-4}$) and $Q_{\text{het}}=12.03$ ($\chi^2_2$,
$p=0.0024$).

**Right — correlation heatmap.** The $k\times k$ correlation matrix of $\hat{\boldsymbol\Delta}$,
values printed in the cells ($0.646$, $0.335$, $0.412$ off-diagonal). Its only job is to stop the
reader treating the left panel's bars as independent.

Caption the pair with the sentence that the figure cannot show: **overlap or non-overlap of the bars
in the left panel does not test any pairwise difference.** That is note 1 §6's overlap fallacy, and
correlation makes it worse in a direction the eye cannot correct for — positively correlated
estimates have differences *more* precise than their individual bars suggest.

---

## 10. Facts and conventions, kept apart

**Facts — consequences of the model and the design:**

- The gold strata are disjoint ⟹ $\operatorname{Cov}(\hat\Delta_{\text{yes}},\hat\Delta_{\text{no}})=0$
  ⟹ variances add for *both* the sum and the difference contrast, which therefore have equal SEs.
- $\hat\Delta_{\text{pool}}=\lambda\hat\Delta_{\text{yes}}+(1-\lambda)\hat\Delta_{\text{no}}$ exactly.
- At $\lambda=\tfrac12$: accuracy change $=\tfrac12(\Delta H-\Delta F)$ and yes-rate change
  $=\tfrac12(\Delta H+\Delta F)=\tfrac12(\Delta_{\text{yes}}-\Delta_{\text{no}})$. The two contrasts
  are the two coordinates.
- Pooled accuracy is one linear functional of the 2-D displacement $(\Delta H,\Delta F)$ and cannot
  identify the other coordinate at any $n$.
- Stratifying narrows the main-effect SE by exactly $\lambda(1-\lambda)(\Delta_1-\Delta_0)^2/n$ of
  variance.
- Under balance and equal $\pi_d$: $\text{SE}(\hat\theta)=2\,\text{SE}(\hat\Delta_{\text{pool}})
  =\sqrt2\,\text{SE}(\hat\Delta_s)$; 4× the $n$ for an interaction of the same numerical size.
- Marginal identity-link model: the interaction coefficient equals
  $\Delta_{\text{yes}}-\Delta_{\text{no}}$ identically.
- Conditional logistic: $\hat\delta=\log\frac{c_1b_0}{b_1c_0}$, variance $\sum 1/\text{cell}$.
- $k$ conditions on a shared baseline are positively correlated through $\operatorname{Var}(B_i)$;
  the independent-samples SE for a window-vs-window difference is too large, i.e. conservative.
- $\hat{\boldsymbol\Delta}\ \dot\sim\ N_k(\boldsymbol\Delta,V/n)$ with $V=\operatorname{Cov}(\mathbf D_i)$;
  every linear-contrast question is one quadratic form.

**Conventions — yours to choose and to state:**

- **The scale the interaction is defined on.** Difference of proportions vs difference of log odds.
  These are different questions and can disagree in sign (§6.2). Not a detail.
- Which stratum weighting defines "accuracy": the benchmark's $\lambda$, or $\lambda=\tfrac12$
  (balanced accuracy). They differ whenever the benchmark is unbalanced.
- Marginal (GEE) vs conditional (conditional ML) vs random-effects (GLMM) — three different
  parameters, not three estimators of one.
- Omnibus-gate-then-post-hoc vs pre-specified primary window; and the family for multiplicity.
- Wald vs score vs exact within each stratum (note 2), and the $m$ threshold for switching.
- Whether to spend degrees of freedom on a shape (trend contrast across layers) or on the general
  alternative.
- The SESOI, separately for the main effect and for the interaction. Still yours, still
  underivable.

---

## 11. What these tests do not answer

Every test above answers one question: **is this pattern distinguishable from sampling noise over
items, given this model and this item set?** That is narrower than what a claim usually asserts, in
three specific ways.

1. **A criterion-shift signature is not a mechanism.** $\Delta H>0,\ \Delta F>0$ is consistent with
   many stories about what the intervention did. The test discriminates the *statistical* hypothesis
   "the two strata moved differently" from "they did not"; it does not discriminate among the causes
   of the former.
2. **The only randomness modelled is over items.** The model is fixed, the direction is fixed, the
   decoding is (presumably) fixed. There is no replication at the model level, so "this would happen
   again" is not a claim any of these $p$-values supports. Note 1's finite-sample material on
   estimated directions is the other half of that, and it is a different variance component.
3. **Significant asymmetry $\ne$ the asymmetry you named.** Rejecting $H_0:\theta=0$ says $\theta\ne0$.
   Whether $\theta=0.28$ is large enough to support a claim is the SESOI question, and whether it
   supports *your* claim is the interpretation question. Both are outside the test.

The route that gets closer to a criterion-free statement is ROC/AUC, and it requires something the
tests above do not: a **graded** score per item rather than a hard yes/no, so that the operating
point can be swept. For a yes/no VLM that means the yes-vs-no logit, and the natural scale is
log-odds rather than probability, because probability differences compress near 0 and 1 while
log-odds differences do not. Agresti §5.1.7 (ROC curves, p. 143) and §2.1.3 (sensitivity and
specificity, p. 23) are the entry points; the AUC-as-concordance-index identity is at §5.1.8. Named,
not derived — it is a separate session and it needs the graded scores to exist on disk first.

---

## 12. Where to go next

Per the standing rule, the check is a textbook problem, not one of mine.

**Primary: Agresti 2nd ed., problem 8.6** (p. 267), which refers to Table 7.19 on opinions about
measures to deal with AIDS, treated as matched pairs on opinion, **stratified by gender**. I checked
the text: the structure is exactly yours, with gender in the role of gold label.

- (a) and (b) are note 1's material inside one stratum: McNemar, then a 90% CI on the difference of
  proportions.
- (c) is §6 of this note: the odds ratio $\exp(\beta)$ under Agresti's marginal model (8.3) and
  under the conditional model (8.4), on the same data, so the two parameters are visibly different
  numbers.
- **(d) is the interaction, and it is the reason I am pointing you here:** "Explain how you could
  construct a 90% confidence interval for the difference between males and females in their
  differences of proportions of support for a particular item. (Hint: The gender samples are
  independent.)" That hint is §2.2 of this note, in Agresti's words. Work (d) *before* rereading
  §2, and write down explicitly which variances add and why.

**Then: 8.12** (p. 269), for the $k$-condition material. It views Table 7.3 as matched *triplets*
(alcohol, cigarettes, marijuana) and asks you to compare three marginal "yes" proportions on the
same subjects; part (b) asks you to set up the three-way table so a generalized CMH procedure
(§6.4.2) tests marginal homogeneity. That is Cochran's Q, i.e. §7.2 of this note, on real data with
$k=3$. Do this one second — it will be much easier once 8.6(d) has made the covariance bookkeeping
concrete.

**Optional third: 8.8** (p. 268), the two-period crossover on migraine treatments, where part (b)
asks why a test of independence on the order-stratified discordant table tests equality of success
rates. It is the same "stratify a paired design and test across strata" move in a different
disguise, and it is a good check that you can recognise the structure without the labels.

Still outstanding from yesterday and not yet worked: **1.12** and **1.18** (Wald vs score on a single
proportion — do these first if the §0 question about 1.8(c) turns out to be the score/Wald confusion),
**8.5** (exact / mid-$P$ McNemar), and 1.10, 1.13, 1.15.

**One gap I will not fill.** Agresti's intro text gives Cochran's Q only by reference, in a starred
subsection, with no worked derivation and no exercise that computes it directly. If you want the
$k$-condition machinery worked properly — the quadratic form, its degrees of freedom, and the
post-hoc structure — I do not have a textbook in hand that I can cite for it at the right level.
Agresti's larger *Categorical Data Analysis* (the graduate text, not this one) covers marginal
models for repeated categorical responses in much more depth and is the obvious candidate, but I
have not verified its contents and will not cite a chapter I have not checked. **Name a source you
want to use and I will work from it**; I am not going to write practice problems for that material
myself.

---

## 13. Comprehension check

Not a substitute for 8.6(d) — do that. But before you start it, answer these two in your own words,
in writing:

**(1)** In §2.2 the covariance between $\hat\Delta_{\text{yes}}$ and $\hat\Delta_{\text{no}}$ is
exactly zero, and in §7.1 the covariance between $\hat\Delta_j$ and $\hat\Delta_l$ for two layer
windows is strictly positive. Both are comparisons of two paired effects. **State the one structural
difference between the two designs that produces this, in one sentence**, and then say which of the
two cases makes an independent-samples SE conservative and which makes it exact.

**(2)** Predict, then check against §5. Suppose you doubled the benchmark to $n=1200$ but kept it
unbalanced at $n_1=200$ gold-yes and $n_0=1000$ gold-no, with $\pi_d=0.10$ throughout. Does the MDE
on the interaction $\theta$ improve relative to the balanced $n=600$ case? Give the factor before
computing it, then compute $\text{SE}(\hat\theta)=\sqrt{\pi_d/n_1+\pi_d/n_0}$ and say what the
result implies about which stratum's size is the binding constraint.

And the two questions in §0 are still open.
