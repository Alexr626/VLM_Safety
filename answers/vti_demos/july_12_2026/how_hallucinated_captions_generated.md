# How VTI authors generated hallucinated demo captions

**Question:** Is there indication from the VTI paper or codebase about how hallucinated captions were generated? Are `co_objects` / `uncertain_objects` used specifically to generate `h_value`?

## Short answer

Yes (paper). The fields `co_objects` and `uncertain_objects` are **generation-time scaffolding** from the LURE-style (Zhou et al., 2023) pipeline: GPT-3.5 proposes co-occurring objects; captions are then rewritten (via LLaVA-1.5) so that randomly sampled co-occurring and uncertain objects are woven into a hallucinated description. The released `demos.jsonl` already contains the finished pairs; **neither the authors’ VTI code nor our reimplementation reads those two lists at direction-extraction time** — only `value` / `h_value` (and the image / question) matter for PCA.

## Paper (arxiv:2410.15778)

Main text (§ textual shifting): clean captions \(x\) are curated; a GPT model produces hallucinated \(\tilde{x}\), following Zhou et al. (2023) (LURE / “Analyzing and Mitigating Object Hallucination…”).

Appendix (hallucinated-caption construction), paraphrased:

1. Curate a small training set (paper says **50** examples; the public JSONL has **100** rows).
2. Prompt **GPT-3.5** (in-context) to list objects that **frequently co-occur** with objects in a given description → stored as `co_objects`.
3. Generate / refine descriptions with **LLaVA-1.5**, **incorporating a randomly selected word from the co-occurring list and another from the uncertain-objects list**.
4. Prompts deferred to Zhou et al. (2023); Figure 9 illustrates an example.

So `co_objects` ≈ GPT-proposed plausible-but-often-absent objects; `uncertain_objects` ≈ uncertainty / hedging tokens used to inject less grounded mentions. They are **inputs to caption synthesis**, not inputs to the VTI intervention math.

## Codebase

- Authors’ repo: `VTI/experiments/data/hallucination_vti_demos.jsonl` (same schema as our `data/vti/demos.jsonl`).
- `get_demos` / textual VTI only use `image`, `question`, `value`, `h_value`. **No references to `co_objects` or `uncertain_objects` outside the JSONL.**
- Our `evaluation/interventions/vti/directions.py` likewise only forwards `h_value` vs `value`.

## Empirical consistency with the demos file

- Most `h_value`s look like edited `value`s with injected objects (e.g. orange → hotdog + scoreboard + baseball bat).
- Substring checks: ~88/100 demos have at least one `co_objects` phrase in `h_value` but not in `value`.
- Demo `000000370505` even appends: `(Note: uncertain_objets list was not used in this output caption)` — leftover generation metadata, confirming the lists were optional generation hints.

## Implication for this project

Treat `co_objects` / `uncertain_objects` as **documentation of how the paired captions were built**. They are useful for qualitative review (which fake objects were targeted) but are **not** part of direction extraction or steering.
