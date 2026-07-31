# Implementation plan — MMHal-Bench (judge-agnostic) + CHAIR scoring

**For:** Cursor implementation agent (Remote-SSH, lambdab2, repo open).
**Author:** research/analysis agent, via Romanus.
**Goal:** turn the two placeholder scorers (`mmhal_bench` → `reference_match`,
`chair` → `chair_pending`) into working metrics, so every intervention in
`ALL_INTERVENTIONS` can be evaluated on both benchmarks the same way it is on POPE.
No new intervention code, no model code. Loaders for both already exist and are
registered; this is scorer + judge-plumbing + eval-loader work only.

**Scope discipline:** do not touch `src/model.py`, the interventions, or the POPE
path. The runner (`evaluation/runners/eval_runner.py`) is already benchmark-agnostic
through `compute_metric_records(records, benchmark)` — the integration point is that
dispatch plus the two benchmarks' eval loaders and scorers.

---

## Step 0 — Verify-in-repo before writing scorer logic (do this first, report back)

These four facts determine the scorer code. The plan branches on them explicitly
below; resolve them by reading the repo, not by assuming. Paste findings into
`RESEARCH_LOG.md` (or back to Romanus) before implementing the scorers.

**0a. `load_mmhal_bench` returned-sample shape.** Open `src/dataset.py`. For a
loaded MMHal sample, confirm which of these are present (in the top-level dict,
or inside `raw`/`metadata`):
- the **question / prompt** text (`text`),
- the **ground-truth answer** (the reference answer),
- the **question category** (the 8 MMHal types: attribute, adversarial,
  comparison, counting, relation, environment, holistic, other),
- the **image-content description** (the human-written gt description that the
  official MMHal GPT-4 judge prompt consumes — distinct from the gt answer).

The official judge prompt needs all four. `response_template.json` from the HF
source typically carries question/gt-answer/category/image-content; confirm the
loader surfaces them rather than dropping them. **If the image-content
description is not surfaced, that is the one blocker** — note it; the judge can
fall back to image+question only, but flag the deviation (it changes judge
behavior and breaks strict VTI comparability).

**0b. `load_chair` exposes COCO `image_id`.** Confirm each CHAIR sample carries
the COCO `image_id` (in `raw` or `metadata`). The CHAIR scorer needs it to look
up ground-truth objects in the instance annotations. If absent, the loader must
be extended to include it (small change, note it as new).

**0c. COCO instance annotations + synonym map on disk.** `download_chair.py`
pulls "COCO val2014 + annotations". Confirm `instances_val2014.json` (the 80-class
**instance** annotations, not captions) is present under `data/coco/annotations/`.
Run `ls data/coco/annotations data/chair data/mmhal-bench`. CHAIR also needs the
standard **synonym list** mapping caption words → the 80 COCO classes; confirm
whether the repo already vendors one (search for `synonym`, `chair`, `coco_classes`).
If not present, it must be added (see Step 2c — do not hand-roll a partial list).

**0d. Current `compute_metric_records` dispatch + MMHal/CHAIR eval-loader stubs.**
Open `evaluation/classifiers/metrics.py` and `evaluation/benchmarks/`. Confirm how
`compute_metric_records` branches per benchmark and what the existing
`reference_match` / `chair_pending` stubs do, so the new scorers slot into the
same dispatch and return the documented `metric_summary.json` shape.

---

## Part A — MMHal-Bench, judge-agnostic

### A1. Judge abstraction (new module)

Create `evaluation/classifiers/judges.py` with a thin provider-agnostic interface:

```python
class Judge(Protocol):
    name: str                      # e.g. "gpt-4", "claude-...", "gemini-..."
    def score(self, prompt: str) -> str: ...   # returns raw judge text

def get_judge(spec: str) -> Judge:
    # spec examples: "openai:gpt-4", "anthropic:<model>", "vertex:<model>",
    #                "mock" (offline/dry-run)
```

Requirements:
- **Provider selected at runtime** via `--judge` CLI arg / env, not hardcoded.
  Default to `"mock"` so a dry run never makes network calls or needs a key.
- Each real provider reads its key from env (`OPENAI_API_KEY`, etc.); if the key
  is missing, raise a clear error naming the env var — do not silently fall back.
- Record the resolved judge `name` into the metric summary (see A4) so every
  MMHal result is self-describing about who judged it.
- **Confidentiality (IMPLEMENTATION.md):** MMHal-Bench images/text are public, so
  external judging is allowed here. Add a one-line guard/comment that this judge
  path is for **public benchmarks only**; internal Nokia data must never be routed
  to an external judge without explicit instruction.

### A2. Judge prompt + parser

Port the **official MMHal judge prompt** (the one used to produce the 0–6 score and
the binary hallucination flag). Inputs per the standard template: question,
model response, ground-truth answer, image-content description, category.

- Keep the prompt text in a constant in `judges.py` or a sibling
  `mmhal_judge_prompt.py`; do not paraphrase the official rubric (the score scale
  semantics matter for VTI comparability).
- Parser: extract the integer **score 0–6** and derive the **hallucination flag**
  per the official convention (rating below the standard threshold = hallucination;
  confirm the exact cutoff from the source template in Step 0 and hardcode it with
  a comment, do not guess silently).
- Be robust to judge chattiness: regex the rating out of a possibly verbose reply;
  on parse failure, mark the sample `unparsed_judge=True` and count it (do not
  crash the run, do not score it as 0 — surface it like POPE's `n_unparsed`).

### A3. Scorer: `score_mmhal_records`

Add to `evaluation/classifiers/metrics.py`, wired into `compute_metric_records`
for benchmark key `mmhal_bench`, replacing the `reference_match` stub.

Signature mirrors the POPE scorer's place in the dispatch:

```python
def score_mmhal_records(records, *, judge: Judge) -> dict
```

Per record: build the judge prompt (A2) from the stored question / response /
gt-answer / image-content / category, call `judge.score`, parse to (rating,
hallucination_flag).

Return (matches documented MMHal summary shape, extended):

```python
{
    "metric": "mmhal_judge",            # was "reference_match"
    "judge_name": str,                  # resolved judge identity
    "avg_score": float,                 # mean rating 0–6, the headline number
    "hallucination_rate": float,        # fraction flagged
    "n_total": int,
    "n_unparsed_judge": int,
    "by_category": {                    # the 8 MMHal types
        cat: {"avg_score": float, "hallucination_rate": float, "n_total": int}
    },
}
```

**`by_category` is the deliverable that matters for the second manager** — it
isolates the **counting** subset. Keep the category key spelling exactly as the
loader surfaces it (verify in 0a; the canonical key is usually `counting`).

### A4. Runner / CLI wiring

- `run_eval.py`: add `--judge` (default `"mock"`); thread it through
  `run_evaluation(...)` to the scorer. `run_evaluation` already takes per-benchmark
  knobs (`pope_split`, `amber_task`); add `judge` in the same style.
- The judge is **only** invoked for benchmark `mmhal_bench`; no other path calls it.
- `print_comparison_table`: render MMHal as one column showing `avg_score`
  (and optionally hallucination-rate); leave POPE's per-split columns untouched.
- Generation for MMHal is the same greedy `generate_vl` path the runner already
  uses; **only the scorer differs**. Set `max_new_tokens` to the MMHal-appropriate
  length (the benchmark expects free-form answers, not yes/no — confirm the
  source's default; likely 256, which is already the runner default).

### A5. Caveats to bake in as comments / log notes (do not silently absorb)

- **n is small.** 96 items total, ~12 per category. The counting subset alone
  cannot carry a counting claim — it is a directional probe. Have the scorer emit
  `n_total` per category so the ~12 is visible in every summary and never
  over-read downstream.
- **Judge variance / comparability.** Absolute `avg_score` is only comparable to
  VTI's reported numbers if judged with the same GPT-4-class model. Under any other
  judge, treat scores as **internal-only** (intervention-vs-baseline within this
  repo). `judge_name` in the summary is what makes this auditable.

---

## Part B — CHAIR (rule-based, no judge)

CHAIR is the bias-robust counterpart to POPE and the metric that actually
adjudicates "rotation improves grounding" vs "rotation just adds words." It needs
no judge — it is deterministic string-matching against COCO instance annotations.

### B1. Caption generation path

CHAIR scores **free-form captions**, not yes/no answers. Confirm the CHAIR eval
loader drives caption generation:
- The intervention interface is
  `InterventionBase.generate(wrapper, image, question, max_new_tokens, caption=None)`.
  For CHAIR the "question" is the caption prompt. Per IMPLEMENTATION.md,
  `generate_vl(image, question)` passes the question through verbatim and does
  **not** auto-substitute `_caption_prompt`; only `generate_caption()` uses the
  fixed `"Describe this image in detail."`.
- **Decision needed (B-DEC-1):** for the intervention path to steer a *caption*,
  the CHAIR eval loader should pass the caption prompt **as the question** to
  `intervention.generate(...)` (so steering hooks are active), rather than calling
  `wrapper.generate_caption()` (which bypasses interventions). Use the standard
  CHAIR prompt as that question string. Confirm this matches how the CHAIR eval
  loader is stubbed; if the stub calls `generate_caption`, change it to route
  through the intervention with the caption prompt as `question`. Flag if this
  assumption is wrong.
- `max_new_tokens`: CHAIR needs enough length for a full caption (use the COCO/CHAIR
  convention, typically 64–128; confirm against the source. Note: caption length
  directly affects CHAIR — see B4 — so this value must be **fixed across baseline
  and all interventions** and recorded, exactly like Policy A for resolution).

### B2. Ground-truth object lookup

From Step 0c. Load `instances_val2014.json` once (cache in module scope — it is
large), build `image_id -> set(coco_class_name)` from the instance annotations.
Use the COCO category id → name map from the same file. Each CHAIR sample's
`image_id` (Step 0b) keys into this.

### B3. Caption → mentioned-objects parser

Standard CHAIR procedure:
- Tokenize the generated caption (lowercase, basic normalization).
- Map words/phrases to the 80 COCO classes via the **synonym list** (Step 0c).
  Use the canonical CHAIR synonym map (e.g. the `chair.py` `synonyms.txt` from the
  Rohrbach et al. CHAIR release) — do **not** hand-roll a partial list, since
  coverage gaps silently deflate hallucination counts.
- Produce, per caption, the set of mentioned COCO objects.

### B4. Scorer: `score_chair_records`

Add to `metrics.py`, wired into `compute_metric_records` for `chair`, replacing
`chair_pending`.

```python
def score_chair_records(records) -> dict
```

Per caption: `mentioned = parse(caption)`, `gt = gt_objects[image_id]`,
`hallucinated = mentioned - gt`.

Compute:
- **CHAIR_i** = sum(|hallucinated|) / sum(|mentioned|)   (instance-level)
- **CHAIR_s** = (#captions with ≥1 hallucinated) / (#captions)  (sentence-level)

Return:

```python
{
    "metric": "chair",
    "chair_i": float,        # lower is better
    "chair_s": float,        # lower is better
    "n_total": int,
    "avg_objects_mentioned": float,   # coverage proxy (see below)
    "avg_caption_len_chars": float,   # length control (see below)
}
```

### B4-CRITICAL. Length / coverage controls (do not omit)

This is the part that makes CHAIR meaningful **for this specific project**, given
that the rotation intervention changes caption length per-model with opposite sign
(LLaVA shorter, Qwen2.5 longer):

- **`avg_objects_mentioned`** and **`avg_caption_len_chars`** must be in every
  summary. CHAIR_s can be gamed down by terse captions (mention 3 sure objects,
  hallucinate nothing); CHAIR_i can rise simply because a longer caption names more
  objects. Without these two columns, a "CHAIR improvement" under rotation is
  uninterpretable — it may just be re-measuring the known length signature.
- Because length confounds CHAIR in **both** directions, fix `max_new_tokens`
  across baseline and interventions (B1) and record it. When the results come back,
  the analysis agent will read CHAIR_i/s **jointly with** `avg_objects_mentioned`,
  not in isolation. Add a one-line comment in the scorer saying so.

### B5. Runner / CLI wiring

- No judge, no new CLI flag beyond what exists. CHAIR runs through the same
  `run_eval.py --benchmarks chair` path.
- `print_comparison_table`: render CHAIR as `chair_s/chair_i` (percentages),
  lower-is-better — make the direction explicit in the column header or a legend
  so it is not misread as accuracy.

---

## Validation (before any sweep)

1. **Dry run, mock judge, tiny limit.** `--benchmarks mmhal_bench --judge mock
   --limit 8 --interventions no_intervention` on one model (e.g.
   `llava-hf/llava-1.5-7b-hf`). Confirm: judge prompt assembles, parser handles the
   mock reply, `by_category` populates, summary schema matches A3, `n_unparsed_judge`
   countable. No network calls.
2. **CHAIR smoke, tiny limit.** `--benchmarks chair --limit 8 --interventions
   no_intervention` on the same model. Confirm: captions generate via the
   intervention path (B1/B-DEC-1), `image_id` lookup hits real annotations,
   CHAIR_i/s and the two length/coverage columns populate, no `image_id`
   KeyErrors.
3. **Annotation-coverage sanity.** For the 8 CHAIR smoke samples, log mentioned vs
   gt object sets for 2–3 captions so Romanus can eyeball that the synonym map
   actually fires (e.g. "bus" matched, not silently dropped). A scorer that finds
   zero hallucinations on everything usually means the synonym map didn't load.
4. **Baseline-only first.** Run `no_intervention` on both benchmarks, all four
   models (LLaVA-1.5, Qwen-VL-Chat, Qwen2-VL, Qwen2.5-VL), at the project's
   standard per-model resolution (Policy A) and **fixed CHAIR `max_new_tokens`**,
   before touching any intervention. These are the reference points; record in
   `RESEARCH_LOG.md` with the judge name (MMHal) and the fixed caption length
   (CHAIR). Only after baselines land do we sweep interventions.

Device policy: these are **accuracy/scoring** runs (generation + post-hoc scoring),
so CPU offload / multi-GPU is allowed (per the device policy). No activation
extraction here, so the single-device constraint does not apply.

---

## Open questions / decisions for Romanus

1. **0a image-content description** — if `load_mmhal_bench` does not surface the
   human-written image-content description, decide: (a) extend the loader to pull
   it from the MMHal source `response_template.json`, or (b) accept image+question
   only judging as a documented deviation (breaks strict VTI comparability). Prefer
   (a) if the field exists in the source.
2. **B-DEC-1 caption routing** — confirm CHAIR should steer the caption by passing
   the caption prompt as `question` through the intervention (not
   `generate_caption`, which bypasses steering). This is required for the
   intervention to affect CHAIR at all; flag if the repo intends otherwise.
3. **Fixed CHAIR `max_new_tokens`** — pick the value now and freeze it (proposed:
   the CHAIR-source default, likely 64). Everything downstream depends on this being
   constant across baseline and interventions.
4. **Judge default for real runs** — `mock` is the safe default for dev. For the
   first scored MMHal runs, which provider? (Affects only comparability, not code,
   since the interface is judge-agnostic — but `judge_name` will be stamped into
   every summary, so decide before the baselines you intend to cite.)
5. **CHAIR synonym map provenance** — if the repo doesn't already vendor the CHAIR
   synonym list, confirm adding the canonical Rohrbach et al. release file rather
   than a hand-rolled subset.
