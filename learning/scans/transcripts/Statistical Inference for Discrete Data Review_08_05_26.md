# Statistical Inference for Discrete Data Review — 08/05/26

Transcription of `learning/scans/Statistical Inference for Discrete Data Review_08_05_26.pdf` (9 pages).

---

## Page 1

Bootstrapping:

1) Randomly sample subset of observed data.

2) Calc. mean of random sample

3) Repeat 1+2 until you have many means

95% confidence interval — Interval that spans 95% of bootstrapped means.

- p-value of anything outside of confidence interval is $<0.05 \implies$ statistical significance.

t-test — Test of significance of difference b/t two independent samples from the same population.

Confidence interval for normally-distr. data:

$$CI = \bar{x} \pm z \cdot \frac{s}{\sqrt{n}}$$

Confidence interval for binomial data — For:

$i \in \{1, \dots, n\}$, $n = 450$, and assuming $C_i \overset{iid}{\sim} \text{Bern}(p)$

---

## Page 2

[illegible — partial line cut off at top of page]

$$x = \sum_i C_i \sim \text{Bin}(n, p), \quad \hat{p} = \frac{x}{n}, \quad \text{Var}(\hat{p}) = \frac{p(1-p)}{n}$$

in this case, $p$ is fixed, $\hat{p} = \frac{x}{n}$ is an estimator, i.e. RV.

Wald Statistic — $z = \dfrac{\hat{\beta} - \beta_0}{se}$, for $se = \sqrt{p(1-p)/n}$

Wald Confidence Interval — Set of $\beta_0$ values for which:

$$|z| < 1.96 \quad (95\% \text{ confidence})$$

Ex. of test using Wald for Binomial Param:

$H_0: \pi = 0.50$, $H_a: \pi \neq 0.50$, $\hat{p} = 0.90$ for $n = 10$:

Wald test:

$$SE = \sqrt{0.9(0.1)/10} = 0.095$$

$$\implies z = (0.9 - 0.5)/0.095 = 4.22$$

---

## Page 3

$\implies p < 0.001$

### Sample Size for Comparing Two Proportions

To determine sample size to test whether "success" prob.'s of two groups $\pi_1$ and $\pi_2$ are identical, we must specify prob. $\beta_0$ of failing to detect difference between groups.

- $\beta_0 = P(\text{rejecting } H_0 \text{ at level } \alpha)$ for p-val $\leq \alpha$:

  $$\alpha = P(\text{Type I error}), \quad \beta = P(\text{Type II error}).$$

- Power $= 1 - \beta$

### Models for Matched Pairs

Two-way contingency table — A summary of matched-pair observations displaying category counts for both classifications.

- Ex — Opinions Relating to Environment

[illegible — line cut off at bottom of page; first characters read as "Pa_ hi_h__", continues as the table header on page 4]

---

## Page 4

| Pay higher taxes | Cut Living Standards: yes | no | Total |
|---|---|---|---|
| yes | 227 | 132 | 359 |
| no | 107 | [unclear: 678] | 785 |
| Total | 334 | 810 | 1144 |

For $\pi_{i,j} = P(\text{outcome } i \text{ for } Q1, \text{ outcome } j \text{ for } Q2)$:

- Marginal homogeneity — $\pi_{1+} - \pi_{+1} = \pi_{12} - \pi_{21}$

- Because samples are from similar populations, (i.e. opinions of same subjects on questions regarding environment), [unclear: $n_{2+}$] $\approx n_{+1}$ i.e., the # of people answering yes to either question is similar (359 vs. 334).

  $\implies$ marginal populations are correlated.

  - Hence, best way to test difference in populations is to test existence of "marginal homogeneity."

    $H_0: \pi_{1+} = \pi_{+1}$, or $H_0: \pi_{12} = \pi_{21}$

### Estimating Differences of Proportions

---

## Page 5

[illegible — partial line cut off at top of page]

Let $p_{i,j} = n_{i,j}/n$,

$$\mathbb{E}[p_{1+} - p_{+1}] = \pi_{1+} - \pi_{+1}:$$

$\text{Var}(p_{1+} - p_{+1})$:

$$\left[ p_{1+}(1 - p_{1+}) + p_{+1}(1 - p_{[\text{unclear: } +1]}) - 2(p_{11}p_{22} - p_{12}p_{21}) \right]/n$$

with

$$SE = \sqrt{\text{Var}(p_{1+} - p_{+1})} = \sqrt{(n_{12} + n_{21}) - (n_{12} - n_{21})^2/n} \Big/ n$$

---

## Page 6

### Review Problems

All problems from Agresti "An Intro to Categorical Data Analysis"

1.8) a) $n = 1170$, [circled:] $\hat{p} = \dfrac{344}{1170} = 0.294$

b) $H_0: \pi = 0.5$, $H_A: \pi \neq 0.5$:

$$SE = \sqrt{\hat{p}(1 - \hat{p})/n} = \sqrt{(0.294)(0.706)/1170}$$

$$= 0.0133$$

$$\implies z = \frac{(0.294) - (0.5)}{0.0133} = -15.49$$

[circled:] $\implies p < 0.001$

A p-value of $< 0.001$ suggests that, with 95% confidence, we can say that the true proportion of people willing to accept cuts in their standard of living is not 0.5.

c) For 99% confidence, $z = 2.576$:

$\implies$ Half interval (HI) $= 2.576(0.0133)$

---

## Page 7

$$= 0.0343$$

[circled:] $\implies p \in [0.466, 0.5343]$

8.1)

| Pay higher taxes | Cut Living Standards: yes | no | Total |
|---|---|---|---|
| yes | 227 | 132 | 359 |
| no | 107 | [unclear: 678] | 785 |
| Total | 334 | 810 | 1144 |

$$z = \frac{n_{12} - n_{21}}{\sqrt{n_{12} + n_{21}}} \approx \frac{132 - 107}{\sqrt{132 + 107}}$$

$$= \frac{25}{\sqrt{239}} \approx 1.62$$

$\implies p = 0.106 \not< 0.05$

$\implies$ The difference between $\hat{\pi}_{12}$ and $\hat{\pi}_{21}$ is not significant, meaning their

---

## Page 8

is no evidence to suggest a difference in the population of those who believe in cutting living standards v.s. paying higher taxes to preserve the environment.

8.2) a) $H_0: \pi_{1+} = \pi_{+1}$, $H_A: \pi_{1+} \neq \pi_{+1}$

$$z = \frac{n_{12} - n_{21}}{\sqrt{n_{12} + n_{21}}} = \frac{125 - 2}{\sqrt{125 + 2}} \approx 10.914$$

[circled:] $\implies p < 0.001 \implies$ Reject null hypothesis

b) $$SE = \sqrt{(n_{12} + n_{21}) - (n_{12} - n_{21})^2/n} \Big/ n$$

$$= \sqrt{(125 + 2) - (125 - 2)^2/1120} \Big/ 1120$$

$$= [\text{unclear: } 0.009951], \quad p_{1+} - p_{+1} = \frac{958}{1120} - \frac{835}{1120}$$

$$= 0.110$$

$$\implies 0.110 \pm 1.645([\text{unclear: } 0.00951]) = [0.0944, 0.1256]$$

---

## Page 9

We can say with 90% confidence that the true difference in the proportion of those that believe in heaven v.s. those that believe in hell is within $[0.0944, 0.1256]$

8.4) Both tests are meant to examine whether a difference exists in populations that are represented using two dependent samples.
