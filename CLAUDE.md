# VLM sycophancy and decision-state geometry — shared project context

This file is shared context for every agent in this repo. It contains facts, not roles.
Role constraints live in `.claude/agents/`. Read `WORKFLOW.md` for how the roles fit together.

## The division of labour this repo enforces

**Anything whose output is a belief Alex will have to defend is Alex's work.
Anything whose output is an artifact is an agent's work.**

Belief (Alex, always): what results mean; whether a conclusion is supported; how much a
result answers the hypothesis; what open questions remain; what to run next; appraisal of
literature (what is genuinely new versus a known recipe on a new dataset); experimental
design.

Artifact (agents, freely): retrieval and accurate summary of papers and files; explanation
and derivation of concepts; code; runs; plots; expansion of an approved design into a
repo-grounded implementation plan; factual reads of result files.

No agent in this repo produces an interpretation of a result, ranks candidate experiments,
or proposes a hypothesis. If a prompt asks for one of those, the correct response is a
question, not an answer. This is not a stylistic preference; it is the purpose of the setup.

## Project

Re-implementation and extension of VTI-style activation steering on vision-language models,
on discriminative hallucination benchmarks. The steering direction is a per-layer
mean-difference vector extracted from a truthful-versus-hallucinated caption contrast,
applied either additively or as a rotation. `steer.py` unit-normalizes each layer's direction
slice at application time, for both variants — extracted magnitude does not reach the model.

Current evidence state, as of 2026-07-28, stated plainly so that no agent overstates it:

- The deployed direction is mean-dominated. Cosine with the raw CAA mean-difference is at or
  above 0.999; PC1 accounts for under 1% of direction norm on Qwen2.5-VL and under 9% on
  LLaVA. Treat it as per-layer CAA mean-difference steering.
- On the POPE-30 2x2 (gold-yes and gold-no subsets, neutral and counter-leading clause,
  n=30 per cell), Qwen rotation @ mlp layers 18-27 raises the probability of "yes" in all
  four cells. Accuracy rises where gold is yes and falls where gold is no. On the gold-no
  subset the clause changes the effect by zero.
- That pattern is a decision-criterion shift. The sycophancy hypothesis — that steering makes
  the model agree with a misleading user assertion — is **not currently supported**, and the
  completed 2x2 reads as a null on it.
- Rotation produces a substantially larger criterion shift than addition on Qwen. LLaVA shows
  effects at or inside noise across all cells. Two other Qwen variants appear inert.
- Everything above is n=30, single seed, one benchmark subset.

No agent should describe the project's headline claim as sycophancy induction. The response-bias
result is the finding on the table; whether a sycophancy effect exists anywhere is open.

## Competencies Alex owns and must not delegate

Three, identified because objections to the paper land on them:

1. Signal detection reasoning on discriminative benchmarks — separating a shift in decision
   criterion from a change in discrimination; recognising when a design cannot distinguish two
   explanations; floor and ceiling compression on a probability scale.
2. Finite-sample behaviour of estimated directions — what cosine two legitimate extractions of
   the same quantity produce at a given n, and how that scales.
3. The geometry of the steering operation — Section 4 of `STEERING_MATH_REFERENCE.md`, still
   marked PENDING with two explicit confusion flags. This is the mechanism the paper claims.

The Tutor may teach these. No agent may apply them to project results on Alex's behalf.

## Infrastructure

Compute: lambdab2, 4x RTX A6000 48GB, sm_86, driver 535 / CUDA 12.2, shared — check
`nvidia-smi` and pick a free GPU with `CUDA_VISIBLE_DEVICES`. RunAI (H100s, NFS-backed) for
larger batch jobs; required for activation extraction, because device_map sharding silently
breaks hook-based interventions.

Storage: `/data` for datasets, weights, logs. `HF_HOME` and `WANDB_DIR` redirected there.

Environment: conda env `vlm_hallucination_mitigation`. torch cu121 (cu124 fallback).
flash-attn intentionally absent — LLaVA and Qwen fall back to sdpa, InternVL needs
`use_flash_attn=False`. sdpa/eager is preferred for interpretability anyway. numpy,
transformers, and tokenizers pins are load-bearing; do not bump them casually.

Repo: `src/model.py` (`create_wrapper`), `src/dataset.py` (`BENCHMARK_REGISTRY`),
`src/extraction.py` (`ActivationCache`, SVD), `src/mediation.py`, `src/paths.py`,
`steer.py`, `directions_v2.py`, `metrics.py`, `run_eval.py`.

Models: `llava-hf/llava-1.5-7b-hf`, `Qwen/Qwen2.5-VL-7B-Instruct` primary;
Qwen-VL-Chat and Qwen2-VL-7B as inert baselines.

Benchmarks: POPE, AMBER, CHAIR, HallusionBench, MMHal-Bench. Counting infrastructure
(FSC-147, MAE/RMSE) does **not** exist in this repo. Do not assume it does.

Reporting primitives: parsed outcome, `score_p_yes_raw`, `score_p_no_raw`. `p_yes_norm` and
`answer_mass` are banned from all reporting.

Demo set: `demosv2`, 555 finals, content hash `9a44f4afde0324b5`, COCO train2014.

No W&B logging exists in this project. Per-run metric summaries under `evaluation/results/`
and `diagnostic_experiments/*/dumps/` are the record.

## Confidentiality

Experiments use public benchmarks only. No internal Nokia schema, code, or customer data
enters this repo or any web search. If internal data ever enters the pipeline, it and any
derived results must not be logged to an external service.

## Calibrating to Alex

CS master's student, BS in CS and Statistics, graduating December 2026, returning Bell Labs
intern, no publications yet. Comfortable with derivations, probabilistic reasoning, transformer
internals, PyTorch hooks and custom modules, activation steering on VLMs, the HuggingFace stack,
LangGraph at production scale. PCA and representation-analysis linear algebra: solid.
Slerp and rotation geometry: in progress. LLM-scale RL practice: none. Telecom: general
familiarity, define acronyms.

Go straight to substance on transformer internals, steering mechanics, statistics, and standard
supervised DL. Show derivations where load-bearing. Do not unpack attention or hooks.

Two documented tendencies the roles exist to counter: firming tentative observations into stated
facts across turns, and accepting agent-supplied interpretation without checking. Both have
recurred during this project. Neither is a knowledge gap.

## Cross-cutting

Direct, concise language. No filler, no self-promotion, no apology loops, no emoji, no
exclamation marks. Say "I cannot verify this" rather than guessing. Never invent numbers,
citations, identifiers, or architecture. When reasoning about methods conflicts with what the
code does, the code is reality.
