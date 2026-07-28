# Abstract — living draft

Rewrite this after every run. Date each version and keep the old ones below, newest first.

The point is not to have a good abstract. The point is that the gap between what you want to
claim and what your files support becomes visible weekly rather than at submission. When a
sentence here stops matching a number in `evaluation/results/`, one of the two has to change,
and noticing which takes about ten seconds.

Rules: no claim appears here without a file that supports it, named in the margin. If you
cannot name the file, the sentence is an aspiration and belongs in the parked section.

---

## Draft — YYYY-MM-DD

<150-200 words. Motivation, what was done, what was found, what it means.>

Claims and their evidence:
- <claim> — <path to the file that supports it>

Parked, wanted but unsupported:
- <claim you would like to make and what would license it>

---

## Draft — 2026-07-28

Current honest position, for reference. Steering with a per-layer mean-difference direction
extracted from a truthful-versus-hallucinated caption contrast produces a decision-criterion
shift toward "yes" in Qwen2.5-VL-7B, concentrated in late layers, roughly four times larger
under rotational application than additive. The shift is unmodulated by a leading clause in
the user prompt. LLaVA-1.5-7B shows no effect outside noise.

Claims and their evidence:
- criterion shift, all four cells same sign — pope30_yes and pope30_no 2x2 summaries
- rotation larger than additive on Qwen — same, layers 18-27, beta 0.9
- unmodulated by clause — pope30_no, +0.088 in both neutral and toward-yes
- LLaVA null — both subsets, all cells inside bootstrap interval width

Parked, wanted but unsupported:
- any sycophancy claim; the completed 2x2 reads as a null on it
- any mechanism claim resting on norm preservation; Section 4 still open
- generalisation beyond n=30, one seed, one benchmark subset
