# Experiment plan — Attention knockout: at which layers is the leading clause read?

Status: draft for implementation (Cursor). Analyst: Claude. 2026-07-20.
Diagnostic experiment name: `leading_clause_attention_knockout`
(new directory under `diagnostic_experiments/`, plain-English naming convention,
local-only logging — no W&B, per the diagnostic-root convention).

---

## Hypothesis / exploratory question

When a misleading leading clause flips a model's yes/no answer on a
discriminative hallucination item, the flip is mediated by attention edges from
downstream positions (the question tokens and the final prompt token) to the
leading-clause token span, concentrated in a contiguous band of decoder layers.

Blocking those edges at the responsible layers should restore the
neutral-prompt answer on flipped items. The output of this experiment is a
per-layer-window "flip recovery" profile for each model, which will serve as
the layer-localization evidence for follow-up intervention experiments
(designed separately; not part of this plan).

This is a localization experiment, not a mitigation method. No steering is
applied anywhere in this plan.

## Models

| Model | HF id | Wrapper | Decoder layers |
|---|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | `LLaVAWrapper` via `create_wrapper` | `wrapper.num_layers` (32) |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | `Qwen2VLWrapper` via `create_wrapper`, `max_pixels=1003520` | `wrapper.num_layers` (28) |

Single unsharded GPU per run (hooks + custom masks are incompatible with
sharding). lambdab2, pick a free A6000 via `CUDA_VISIBLE_DEVICES`. Greedy
decoding is repo-wide but this experiment scores the first token only — no
generation calls needed.

## Data and prompt conditions

Items: the AMBER-100 pin (`data/amber/pinned_amber_disc_100.json`, 20 per
stratum × 5 strata) plus the POPE-30 pin
(`data/pope/pinned_pope_existence_yes_30.json`, 10 gold=yes per POPE split) —
130 items per model, loaded via `load_benchmark("amber", task="discriminative",
subset_ids=...)` and `load_benchmark("pope", ...)` with the pin files.

Prompt conditions, from the existing augmentation outputs
(`augment/outputs/augmented_amber100.jsonl`, `augmented_pope30.jsonl`, built
from `templates/leading_clauses_v1.json`):

- **neutral** — the bare benchmark question.
- **misleading leading** — the {tentative, assertive} clause whose direction
  (toward_yes / toward_no) points *against* the gold label for that item.
  (For a gold=yes item this is the toward_no clause, and vice versa.)
  Truth-congruent clauses are deliberately out of scope here (follow-up plan).
- **filler** — NEW. Length-matched semantically inert prefix; see below.

### Filler control (new augmentation file)

New template file `templates/filler_clauses_v1.json` with **two** filler
paraphrases (e.g. "I'm looking at this picture and I have a question about
what it shows." and one distinct paraphrase). Requirements:

1. No reference to any answer, object, attribute, or relation; no yes/no
   valence.
2. Token length within ±2 tokens of the leading-clause templates under **each**
   model's tokenizer. Exact cross-tokenizer matching is not required; record
   the actual token counts per template per model in the augmentation meta
   file. Two paraphrases exist so filler-specific semantics can be checked for
   (results should be stable across the two).

Extend `augment/build_augmented_jsonl.py` to emit the filler conditions into
the same augmented JSONLs (new condition names `filler_a`, `filler_b`).

### Tokenization invariant (assert in code, per model)

For every item and condition: the token IDs of the shared question span in the
prefixed prompt must be exactly the token IDs of the neutral question at the
same relative offsets, and the prefixed prompt must equal
`[template prefix tokens] + [clause tokens] + [shared question tokens] +
[template suffix tokens]`. If a clause/question boundary merges tokens,
adjust the separator in the template until the assertion passes, and record
the fix. The clause span indices derived from this assertion are the knockout
targets. Hard-fail the run on violation — do not silently continue.

## Stage 0 — offline flip-set computation (zero GPU)

From the existing baseline dumps
(`diagnostic_experiments/perception_diag/{model}/dumps/amber100_baseline/` and
`.../pope30_existence_yes_baseline/`, manifests contain responses, parses, and
first-token yes/no scores):

1. Verify first-token decidedness: for every item × condition, the parsed
   yes/no answer in the manifest must agree with the argmax of the stored
   first-token yes/no scores. Report the agreement rate; investigate before
   proceeding if below ~99%.
2. Per model, per misleading condition (tentative, assertive): the **flip
   set** = items answered correctly under neutral and incorrectly under that
   misleading condition. Primary analysis set = union across the two
   misleading conditions, with per-condition breakdown retained.
3. Report flip-set sizes per model × benchmark × stratum. These counts are a
   headline sycophancy statistic on their own and go in the results table
   regardless of what knockout shows.
4. Also draw a **stable set** (same size as the flip set, matched by stratum
   where possible): items whose answer does not change under the misleading
   clause. Used as a negative control — knockout should not flip these.

If the union flip set for a model is smaller than ~15 items, stop and report
before running knockout; the localization would be underpowered and we should
discuss enlarging the item pool first.

## Knockout method

Attention knockout in the Geva et al. (2023, "Dissecting Recall of Factual
Associations") sense, as applied to VLMs by Neo et al. (2024): for chosen
(query position, key position) pairs at chosen layers, set the pre-softmax
attention score to −inf so those queries cannot read those keys; remaining
attention renormalizes.

- **Blocked keys:** the leading-clause token span (from the tokenization
  invariant above). The template prefix, image tokens, and question tokens are
  never blocked.
- **Query scopes (two, run separately):**
  - *block last-token reading* — only the final prompt position's queries are
    blocked from the clause span. Measures the direct read into the decision
    position.
  - *block all downstream reading* — all positions after the clause span
    (question span + final token) are blocked from it. Measures total clause
    influence including relay through question-token positions.
- **Layer windows (coarse, Neo-et-al.-style):** overlapping windows of **10**
  consecutive decoder layers, stride **5**, with the final window
  right-aligned to include the top layer. LLaVA-1.5 (32 layers, 0-indexed):
  0–9, 5–14, 10–19, 15–24, 20–29, 22–31. Qwen2.5-VL (28 layers): 0–9, 5–14,
  10–19, 15–24, 18–27. Six / five windows per model respectively. This is a
  band-level localization by design; finer windows are a follow-up run
  restricted to whichever band responds, not part of this sweep. Additionally
  one **all-layers** run per scope (see sanity checks).

Implementation: new module (suggest `src/attention_knockout.py` or an
extension of `src/mediation.py` — Cursor's call) that builds a per-item 4D
additive attention mask and threads it through `forward_with_logits`. sdpa
accepts custom 4D masks; flash-attn is not installed in this environment
(intentionally), so no kernel obstacle. **Verification requirement:** for one
item per model, run once in eager mode with `output_attentions=True` and
assert the blocked edges carry exactly zero post-softmax weight. Include this
as a unit test.

Scoring: first-token yes/no probability and logit difference via
`ScoringTarget.yes_no(wrapper)` + `forward_with_logits` /
`compute_yes_prob` from `src/mediation.py`. Prefill-only; no generation.

## Cells

| Cell (plain-English directory name) | Model | Query scope | Items |
|---|---|---|---|
| `llava_block_last_token_reading` | LLaVA-1.5-7B | last token only | flip set + stable set |
| `llava_block_all_downstream_reading` | LLaVA-1.5-7B | all downstream | flip set + stable set |
| `qwen25_block_last_token_reading` | Qwen2.5-VL-7B | last token only | flip set + stable set |
| `qwen25_block_all_downstream_reading` | Qwen2.5-VL-7B | all downstream | flip set + stable set |

Within each cell: the misleading-condition prompt(s) that produced the flip,
one forward pass per layer window (plus the all-layers pass). Filler and
neutral conditions are run **without** knockout (see metrics) plus one
filler-span knockout pass at the peak window (specificity check).

Rough cost per model: ≤ 2 conditions × ~(flip+stable ≈ 60–120 items) ×
(~6 windows + 1 all-layers) × 2 scopes ≈ 1.5–3.5k prefill-only forwards — a
few hours on one A6000. If cost must still be cut, drop the
*block last-token reading* scope first (the all-downstream scope contains the
sanity anchor).

## Metrics (plain English; computed per layer window, per cell)

- **Flip recovery rate** — fraction of flip-set items whose first-token answer
  under knockout equals their neutral answer. Primary metric.
- **Logit-difference recovery** — per item,
  (score_knockout − score_leading) / (score_neutral − score_leading), where
  score = logit(yes) − logit(no). FCCT-style recovery rate. Denominator
  floor: exclude items with |score_neutral − score_leading| < ε (ε chosen at
  analysis time from the score distribution; report the exclusion count).
  Report the distribution (median + IQR), not just the mean.
- **Stable-set false-flip rate** — fraction of stable-set items whose answer
  changes under knockout. Expected near zero; nonzero values bound the
  method's collateral damage.
- **Attention-dilution reference** — answer agreement and mean
  logit-difference shift of filler vs. neutral (no knockout), per filler
  paraphrase. Quantifies how much of the leading effect is "any prefix at
  all." Reported once per model, not per window.
- **Filler-span knockout specificity** — flip recovery when knocking out the
  filler span instead of the clause span, at the peak window only. Expected
  null.

## Sanity checks (run before the window sweep)

1. Tokenization invariant assertion passes for all items × conditions × models.
2. First-token decidedness ≥ ~99% on baseline manifests (stage 0).
3. **Full-depth anchor:** all-layers, all-downstream knockout of the clause
   span should approximately reproduce neutral behavior. Report the agreement
   rate with the neutral answer on the flip set. If this is not high (≥ ~80%),
   the residual difference is positional/dilution rather than clause content —
   report and stop for discussion before interpreting window curves.
4. Eager-mode zero-attention assertion (unit test above).

## Artifacts

Under `diagnostic_results_dir("leading_clause_attention_knockout", model_short)`
via `src.paths`; plots under the matching plots dir. Self-describing names:

- `flip_set_summary_by_model_and_stratum.csv` (stage 0)
- `first_token_answer_agreement_baseline_dumps.csv` (stage 0 check)
- `flip_recovery_rate_by_layer_window.csv` / `.png` (one line per query scope)
- `logit_difference_recovery_by_layer_window.csv` / `.png`
- `stable_set_false_flip_rate_by_layer_window.csv`
- `filler_vs_neutral_behavior_summary.csv`
- `full_depth_knockout_sanity_summary.json`
- `run_metadata.json` (model ids, pins, condition names, window size/stride,
  mask implementation commit, per-template token counts)

Append a factual run record to `RESEARCH_LOG.md` (commands, paths, headline
flip-set sizes and peak-window recovery numbers).

## What would confirm / falsify

- **Localized mediation:** flip recovery concentrated in a contiguous window
  band (recovery high inside, low outside), stable-set false flips near zero,
  filler-span knockout null → the clause is read at an identifiable layer
  band; report the band per model.
- **Distributed mediation:** recovery only under all-layers knockout, flat
  window curves → clause reading is distributed across depth (informative
  either way; report the flat curves).
- **Anchor failure (sanity 3):** full-depth knockout does not restore neutral
  behavior → prefix presence (position/dilution), not clause content, drives
  much of the flip; the filler metrics quantify how much. Stop and report for
  discussion before further interpretation.

## New code required (explicit)

1. `templates/filler_clauses_v1.json` + filler conditions in
   `augment/build_augmented_jsonl.py` (NEW conditions `filler_a`, `filler_b`).
2. Attention-knockout mask builder + forward path (NEW; sdpa 4D mask), with
   the eager-mode verification unit test.
3. Clause-span index derivation + tokenization-invariant assertion utility
   (NEW; write as a reusable standalone utility — future experiments on these
   augmented prompts will need it).
4. Stage-0 offline flip-set script over existing perception_diag manifests
   (NEW; zero GPU).

Nothing in `evaluation/interventions/` is touched by this plan.

## Open questions (answer before or during implementation)

1. Window scheme is fixed (width 10, stride 5, per the tables above); do not
   refine it in this run. If exactly one band responds, a finer-window pass
   inside that band is a small follow-up run to propose separately.
2. ε for the recovery-rate denominator floor: set from the observed
   |score_neutral − score_leading| distribution in stage 0 (propose the 10th
   percentile; confirm with analyst before finalizing plots).
3. Qwen2.5-VL M-RoPE: confirm the text-token mask indexing is unaffected by
   the multimodal position scheme (spot-check the eager attention assertion on
   a Qwen item specifically).
4. Are both filler paraphrases stable in behavior? If they diverge materially,
   flag before interpreting the dilution reference.
5. If the flip set is dominated by one stratum (e.g. existence×no), per-stratum
   knockout curves may be underpowered — acceptable for this exploratory pass,
   but note it in the results.
