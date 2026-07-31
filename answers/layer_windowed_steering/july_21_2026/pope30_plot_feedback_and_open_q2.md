# POPE-30 plot feedback + open question 2 — 2026-07-21

## Plot updates

Nine files under `diagnostic_experiments/perception_diag/windowed_steering_summary/plots/` — one per (metric × method):

| Metric | rotation @ mlp | rotation @ layer | additive @ mlp |
|--------|----------------|------------------|----------------|
| parsed yes-rate | `…_rotation_mlp.png` | `…_rotation_layer.png` | `…_additive_mlp.png` |
| mean P(yes) raw | same pattern | same | same |
| flips yes→no | same pattern | same | same |

For **rotation @ layer**, β=0.9 is not plotted (caption notes the omission).

## Why rotation @ layer β=0.9 was dropped (degeneracy investigation)

Two separate mechanisms:

1. **`parse_outcome`** (`_normalize_yes_no` on the full response) only looks for a yes/no answer. Degenerate text that *starts* with “Yes, …” still counts as parseable yes — e.g. `rotation_layer_0.9_layers_5_14`: **60/60** `degeneracy_flag=True` but **60/60** still `parsed_outcome` ∈ {yes, no}. The yes-rate bar would look “fine” while the generation is looping nonsense after the first sentence.

2. **`degeneracy_flag`** is a weak word-level heuristic (`steered_capture.py`): needs ≥12 whitespace words, then either a 4-gram repeating ≥4 times or uniqueness &lt; 0.35 with ≥24 words. It **misses**:
   - short garbage (`"15"`, `"1000\n\n400"`) — too few words
   - character-level loops (`AAAAAALAALAL…`) — few “words”, high uniqueness by space-split
   - On `layers_0_9` / `layers_all` at β=0.9 those are already `unparseable` (so yes-rate hides them), but the flag itself is False on most of them (45/60 and 42/60).

Not every β=0.9 rotation_layer window is collapsed (mid/late windows often look like normal captions). Omitting the whole β series avoids mixing collapsed and non-collapsed cells in one visual.

## Open question 2 — what it was, and what it is *not*

**“Conditions”** here means the two **prompt conditions** used in the steered POPE-30 dump:

- `neutral` — bare existence question  
- `assertive_toward_no` — leading clause that pushes toward “no”

It does **not** mean intervention methods or layer windows.

### Why the plan asked to flag baseline parse coverage

Plot 3 (flips) reports an **integer count** of items that were baseline-yes and steered-no. The panel subtitle prints the denominator: e.g. `baseline: 27/30 parsed yes under …`.

If that denominator differed a lot by condition — say 28/30 under neutral but only 12/30 under the leading clause — then a bar of “10 flips” would mean very different things in the two rows (10/28 vs 10/12 of the baseline-yes pool). The plan asked for a chat flag so you would not misread row-to-row flip heights as comparable without checking denominators.

**What we found:** denominators match — **27/30** parsed yes under both conditions, and it is the **same three items** that are baseline-no in both (`pope_{adversarial,popular,random}_00016`). So flip counts are on the same 27-item pool in both rows; no denominator caveat.

### Does that mean the model answered identically despite the lead?

**Parsed yes/no:** yes — for all 30 items, `parsed_outcome` under neutral equals `parsed_outcome` under assertive_toward_no (same 27 yes / 3 no, same IDs).

**Full response text:** no — only **6/30** items have identical response strings. Under the leading clause the model usually still says yes, but often with a shorter or reworded continuation.

**First-token score:** also not identical — mean unconditional `score_p_yes_raw` is **0.91** (neutral) vs **0.79** (assertive_toward_no). So the leading clause does shift probability mass even when the eventual parsed answer stays “yes.”
