# Reading list: the machinery behind the cosine/reliability notes

Date: 2026-07-28
Question: "Where can I read more about these concepts so the explanations become clearer? It's
been a while since I've done these derivations."

Organised by **which step in the derivations it unlocks**, not by subject. Each entry says what
to read it for and roughly how long.

**Verification status.** Entries marked ✓ had their existence and URL confirmed by web search on
2026-07-28. Entries marked ○ are from memory and were **not** verified in this session — check
before citing them anywhere. Chapter and section numbers are given only where I am confident;
where a number is absent, find it by the chapter title.

---

## Tier 0 — the two chapters that do the most work

If you read nothing else, read these. Between them they cover the concentration argument that
every "$\approx$" in the three notes rests on. Roughly four hours total.

**✓ Blum, Hopcroft & Kannan, *Foundations of Data Science*, Chapter 2, "High-Dimensional Space."**
Free PDF: [cs.cornell.edu/jeh/book.pdf](https://www.cs.cornell.edu/jeh/book.pdf) (also mirrored at
[home.ttic.edu/~avrim/book.pdf](https://home.ttic.edu/~avrim/book.pdf)).

Read it for: why two independent random vectors in $\mathbb{R}^d$ are nearly orthogonal, why the
norm of a random vector concentrates tightly around its mean, and why "most of the mass is near
the equator." That last fact **is** the right-triangle picture in
`why_expected_cosine_is_sqrt_reliability.md` §3 — the claim that essentially all estimation error
lands perpendicular to $\mu$ is this chapter's content, restated. Written for CS readers, prose-
first, minimal measure theory. This is the single best starting point for your specific gap.

**✓ Vershynin, *High-Dimensional Probability*, Chapter 3, "Random Vectors in High Dimensions."**
Free PDF and lecture videos: [math.uci.edu/~rvershyn/teaching/hdp/hdp.html](https://www.math.uci.edu/~rvershyn/teaching/hdp/hdp.html).

Read it for: the concentration-of-the-norm theorem, which is the formal statement of why
$\mathbb{E}\|X\| \approx \sqrt{\mathbb{E}\|X\|^2}$ and how large the gap is. That is
approximation (iv) in the same note — the Jensen step you asked about. Harder than BHK and worth
attempting second, once BHK has given you the intuition the theorems are formalising. Chapters 1
and 2 (concentration inequalities) are the prerequisites if Chapter 3 reads as too dense.

---

## Gap A — covariance algebra of random vectors

The step you asked about first: $\operatorname{Cov}(\varepsilon_i) = \Sigma \Rightarrow \operatorname{Cov}(\hat d_n) = \Sigma/n$,
and the rule $\operatorname{Cov}(Ay) = A\operatorname{Cov}(y)A^\top$.

**○ Johnson & Wichern, *Applied Multivariate Statistical Analysis*, Chapters 2–3.** Chapter 2 is
the matrix algebra refresher; Chapter 3 does random vectors, mean vectors, covariance matrices,
and linear combinations. This is the most direct match to the gap — it derives exactly the rules
I used, at exactly the level of a stats BS returning to the material. Any edition is fine.

**○ Mardia, Kent & Bibby, *Multivariate Analysis*, Chapters 1–2.** Same content, terser, more
mathematical. Prefer if Johnson & Wichern feels padded.

**✓ Petersen & Pedersen, *The Matrix Cookbook*.** PDF:
[doc.ic.ac.uk/~ahanda/referencepdfs/matrix_cookbook.pdf](https://www.doc.ic.ac.uk/~ahanda/referencepdfs/matrix_cookbook.pdf).
Not a book to read — a lookup table. Keep it open. The trace identities I used
($\operatorname{tr}A = \sum_i \lambda_i$, $\operatorname{tr}(AB) = \operatorname{tr}(BA)$,
linearity) are all in its Basics section, and its expectation/covariance section has the
$A\Sigma A^\top$ rules.

## Gap B — trace, eigenvalues, basis invariance

**✓ Axler, *Linear Algebra Done Right*, 4th edition — the chapter on trace and determinant.**
Free, open access (CC BY-NC): [linear.axler.net/LADR4e.pdf](https://linear.axler.net/LADR4e.pdf),
also on [Springer](https://link.springer.com/book/10.1007/978-3-031-41026-0).

Read it for: trace defined properly, proved equal to the sum of eigenvalues, and proved
invariant under change of basis. Axler builds the whole book without determinants, which means
the trace results are derived from operator structure rather than from index manipulation — that
is exactly the perspective that makes "trace does not depend on your choice of coordinates" feel
inevitable rather than surprising. Skim earlier chapters; go straight to the trace material and
back-fill.

## Gap C — the delta method, Jensen, and expectations of ratios

Approximations (iii) and (iv): why $\mathbb{E}[N/D] \approx \mathbb{E}[N]/\mathbb{E}[D]$ is
sometimes acceptable, and how to compute the error term.

**○ Casella & Berger, *Statistical Inference*, 2nd ed., Chapter 5** — the delta method
subsection (I believe §5.5.4; confirm from the table of contents). First- and second-order delta
method, worked. This is the standard reference and probably the one you already own.

**○ van der Vaart, *Asymptotic Statistics*, Chapter 3, "Delta Method."** The rigorous treatment,
including the multivariate version. Read this if you want the conditions stated precisely rather
than assumed. Considerably harder than Casella & Berger.

**○ Wasserman, *All of Statistics*.** Has the delta method and multivariate moments, compressed.
Good as a single-volume refresher for someone who has seen it all before and wants the shortest
path back.

## Gap D — reliability, attenuation, disattenuation

The $\sqrt{r_A r_B}$ ceiling and the correction $\hat\rho = \cos_{\text{obs}}/\sqrt{r_A r_B}$ are
century-old psychometrics, and the psychometric literature has thought harder about their failure
modes than the ML literature has.

**○ Spearman (1904), "The proof and measurement of association between two things."** The
original attenuation correction. Short, and the reasoning is entirely transparent.

**○ Lord & Novick, *Statistical Theories of Mental Test Scores* (1968).** The canonical treatment
of classical test theory: true score, reliability, the validity ceiling
$\rho_{XY} \le \sqrt{r_{XX}}$, and Spearman–Brown. Dense but definitive. Read the chapters on
reliability and on the attenuation correction, not the whole book.

**✓ Nili, Wingfield, Walther, Su, Marslen-Wilson & Kriegeskorte (2014), "A Toolbox for
Representational Similarity Analysis," *PLoS Computational Biology* 10(4): e1003553.**
[journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003553](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003553)

Read it for the **noise ceiling** construction. Neuroscience hit precisely our problem —
comparing similarity structures estimated with different amounts of noise — and the noise ceiling
is their answer. It is the same object as $\sqrt{r_A r_B}$, arrived at independently, with the
practical estimation details worked out. The closest existing methodological analogue to what the
notes describe.

## Gap E — random matrix theory, for the PC1 claims

Background for the claim in `noise_floor_and_disattenuation.md` §6.2 that a PC1 null is not a
mean-difference null. **None of the following were verified this session** — treat as pointers.

**○ Potters & Bouchaud, *A First Course in Random Matrix Theory*.** Written for practitioners
rather than probabilists. Marchenko–Pastur and the spiked model are covered without requiring a
measure-theory background.

**○ Benaych-Georges & Nadakuditi (2011), "The eigenvalues and eigenvectors of finite, low rank
perturbations of large random matrices."** The overlap formula: how much a sample eigenvector
aligns with the population one as a function of signal strength and $d/n$, including the
threshold below which the answer is zero.

**○ Johnstone (2001)** on the largest eigenvalue in the spiked covariance model, and **○ Paul
(2007)** on eigenvector asymptotics in the same model. Original sources for the BBP threshold
behaviour.

This tier is optional. It matters only if you end up reporting a PC1-based direction and need to
defend its floor.

## Gap F — the same problem, already argued about in the embedding literature

Useful because the arguments are conducted on objects you can inspect, at a scale you can rerun.

**✓ Antoniak & Mimno (2018), "Evaluating the Stability of Embedding-based Word Similarities,"
*TACL* 6:107–119.** [aclanthology.org/Q18-1008](https://aclanthology.org/Q18-1008/) ·
[code](https://github.com/maria-antoniak/word-embedding-stability)

Measures how much embedding-derived quantities move under bootstrap resampling of the training
corpus, finds nearest-neighbour distances are highly sensitive to small corpus changes, and that
the sensitivity is worse for smaller corpora. Their recommendation is to average over bootstrap
samples rather than trust a single model. This is a worked instance of the reliability question
on the exact object class we discussed.

**✓ Wendlandt, Kummerfeld & Mihalcea (2018), "Factors Influencing the Surprising Instability of
Word Embeddings," NAACL.** [arxiv.org/abs/1804.09692](https://arxiv.org/pdf/1804.09692)

**✓ "Assessing the Reliability of Word Embedding Gender Bias Measures"** —
[arxiv.org/abs/2109.04732](https://arxiv.org/pdf/2109.04732). Surfaced by title and abstract in
search; I have **not** read it. From the title it sits directly on the intersection of gender
directions and reliability, which is the running example in
`cosine_vs_sample_size_mean_difference_directions.md`. Worth checking first, with the caveat that
I cannot vouch for its contents.

**○ Bolukbasi, Chang, Zou, Saligrama & Kalai (2016), "Man is to Computer Programmer as Woman is
to Homemaker?"** and **○ Gonen & Goldberg (2019), "Lipstick on a Pig."** The construct-validity
dispute referenced in §7 of the companion note — whether the pair-difference gender direction is
the thing it is taken to be. Unverified this session.

---

## Suggested order

| # | What | Why now | Time |
|---|---|---|---|
| 1 | BHK Ch. 2 | Makes the right-triangle picture obvious rather than asserted | ~2 h |
| 2 | Johnson & Wichern Ch. 3 | Answers the $\Sigma/n$ question at its own level | ~2 h |
| 3 | Axler, trace chapter | Basis invariance, trace as sum of eigenvalues | ~1 h |
| 4 | Casella & Berger, delta method | Lets you derive the error term yourself | ~1 h |
| 5 | Nili et al. 2014 | Shows the ceiling deployed on real, noisy data | ~1 h |
| 6 | Vershynin Ch. 3 | Formalises what BHK gave you as intuition | ~3 h |
| 7 | Antoniak & Mimno | The applied instance | ~1 h |

Items 1–4 are the ones that will change how the notes read. Items 5–7 are for when you need to
defend a choice rather than understand one. Gap E is optional and only becomes relevant if a PC1
direction ends up in the writeup.

## Related

- `answers/concepts/july_28_2026/sample_mean_covariance_and_trace.md` — Gaps A and B
- `answers/concepts/july_28_2026/why_expected_cosine_is_sqrt_reliability.md` — Tier 0 and Gap C
- `answers/concepts/july_28_2026/noise_floor_and_disattenuation.md` — Gaps D and E
