# Review queue

Concept gaps surfaced during research work, tracked here rather than carried in memory or left
in whatever chat they came up in. Appended by the tutor when a gap surfaces mid-task; updated by
Alex or the tutor as items move. No required count, no interval, no gate — this is a visibility
instrument, not enforcement.

**Status values:** queued (identified, not started) / attempted (worked, not yet discussed) /
reviewed (discussed with the tutor against a cited source, status closed).

Watching an explainer video is not a status change. It's fine as orientation before doing
problems; it does not move an item from queued to attempted.

---

## Open

### Vectors, inner products, and orthogonality

Dot product and its relationship to the angle between vectors; orthogonal decomposition of a
vector with respect to another (parallel and perpendicular components); orthonormal bases and
what coordinates relative to one mean; spans and subspaces; basis change generally. Foundational
under both the PCA work and the steering-geometry work, and general enough that it's worth
reviewing as its own topic rather than only in service of one project method.

Sources:
- Strang, *Introduction to Linear Algebra* — computational, worked-example-heavy, wide chapter
  coverage of exactly this material with problem sets. The standard first choice if it's been a
  while.
- Axler, *Linear Algebra Done Right*, 4th ed. — proof-first, no determinants until late. Free:
  linear.axler.net/LADR4e.pdf. Harder going but the trace/eigenvalue chapters in particular pay
  off for later interpretability reading (why a trace or an eigenvalue doesn't depend on your
  choice of basis). Better as a second pass than a first.

### PCA and dimensionality reduction

What a principal component is, centered vs. uncentered PCA and why the distinction matters,
what "explained variance" means and how it's computed, and the basic behavior of PCA at small
sample size relative to dimension. Videos are a reasonable orientation but this needs actual
problems — computing components by hand on a small worked matrix, not just watching the
geometric intuition.

Sources:
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, the
  unsupervised-learning / PCA chapter. Already in your rotation for other topics, has exercises,
  and is pitched at the right level for a first pass.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, the corresponding
  chapter — more mathematical treatment of the same material, useful once ISL's version feels
  too easy.

### Rotation and interpolation geometry

Constructing an orthonormal basis for the plane spanned by two vectors; parameterizing points on
a circle by angle; linear (nlerp) vs. spherical (slerp) interpolation between two directions and
why they differ; what a chord step is versus an angular step.

Sources:
- Dunn & Parberry, *3D Math Primer for Graphics and Game Development*, the orientation/rotation
  chapters covering quaternions and interpolation — end-of-chapter exercises. Written for a
  different audience (graphics/games) than the rest of this list, but it's the standard place
  slerp/nlerp actually get worked as problems rather than asserted.

### Random vectors, covariance, and estimator variance

Covariance of a random vector; how covariance transforms under a linear map
(Cov(Ay) = A Cov(y) A^T); why an average of n iid draws has covariance shrinking like Sigma/n;
basic matrix trace identities used to manipulate these (trace as sum of eigenvalues, cyclic
property). This is the layer underneath "why does a mean-difference direction get more reliable
with more samples," which comes up repeatedly in this line of work.

Sources:
- Johnson & Wichern, *Applied Multivariate Statistical Analysis* — matrix-algebra refresher
  chapter followed directly by random vectors and covariance; pitched at a returning-stats-BS
  level, any edition fine.
- Petersen & Pedersen, *The Matrix Cookbook* — not a book to work problems from, a reference to
  keep open while working them. Free PDF, widely mirrored.

### Inference on proportions and paired binary outcomes

Binomial estimation and confidence intervals on a single proportion (Wald and Wilson, where they
diverge and why); the difference between an unpaired two-sample comparison and a paired design
on the same items; McNemar's test; minimum detectable effect as something computed before a run
rather than read off after one; correcting for testing multiple configurations at once. This
whole cluster goes back further than "since undergrad" for the specific paired/McNemar material
— treat it as closer to new than rusty.

Sources:
- Agresti, *An Introduction to Categorical Data Analysis* — the standard reference for exactly
  this: proportions, confidence intervals, McNemar's test, paired data. Has problem sets.
- Casella & Berger, *Statistical Inference* — for the more general hypothesis-testing and
  estimation machinery underneath (what a confidence interval actually promises, power,
  minimum-detectable-effect-style calculations), if the foundations under Agresti's applied
  treatment also need rebuilding.

### Reliability and comparing noisy estimates (lower priority)

Split-half reliability and disattenuation; the idea of a noise ceiling when comparing two noisy
measurements of the same underlying thing. Relevant to interpreting how stable an extracted
direction or a measured effect is, but more specialized than the clusters above and fine to
defer until those are underway.

Sources:
- Nili, Wingfield, Walther, Su, Marslen-Wilson & Kriegeskorte (2014), "A Toolbox for
  Representational Similarity Analysis," PLoS Computational Biology 10(4): e1003553 — the
  noise-ceiling construction, worked in a setting (neuroscience representational similarity)
  that maps closely onto the comparison-of-noisy-estimates problem here.
- Lord & Novick, *Statistical Theories of Mental Test Scores* — the classical-test-theory
  treatment of reliability and the attenuation correction, if the concept needs building from
  further back than the paper above assumes.

### Random matrix behavior of PCA at small n (optional, defer)

Why explained-variance ratios inflate when n is small relative to dimension; Marchenko-Pastur as
the relevant background. Only matters if a PC1-based direction ends up load-bearing in an actual
result — don't prioritize this over the clusters above.

Sources:
- Potters & Bouchaud, *A First Course in Random Matrix Theory* — written for practitioners,
  covers Marchenko-Pastur and the spiked model without requiring a measure-theory background.

## Reviewed

*(none yet)*
