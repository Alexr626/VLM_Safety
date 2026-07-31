# Attention blocking, eager verify, and filler–neutral agreement (2026-07-20)

Detailed write-up for Gate 3 context and the supporting checks Alex asked about.

---

## 1. How attention blocking is implemented

Location: `src/attention_knockout.py`, driven by
`diagnostic_experiments/leading_clause_attention_knockout/run_knockout.py`.

### What is blocked

For each item under a misleading (or filler) prefixed prompt we locate two
spans in the **decoder sequence** (after LLaVA’s already-expanded image
tokens):

- **Key span (blocked sources):** the leading-clause tokens — the `clause_len`
  positions immediately before the shared question span.
- **Query span (who is forbidden to read those keys):** either
  - `block_last_token_reading` — only the final prompt position, or
  - `block_all_downstream_reading` — every position from the end of the clause
    through the last prompt token (question + template suffix).

Image tokens, chat-template prefix tokens, and the question tokens themselves
are never put in the key (blocked) set.

### How the block is applied

At each selected decoder layer we register a **forward pre-hook** on that
layer’s self-attention module (`get_dispatch(wrapper).get_attn`). The hook:

1. Reads the current hidden-state sequence length.
2. Builds a 4D **additive** attention mask of shape `(batch, 1, Q, K)` that is
   `0` everywhere except `dtype.min` (effectively −∞) at every blocked
   `(query, key)` pair.
3. Merges that mask with whatever attention mask HuggingFace already passed
   (causal / padding), by addition.
4. Writes the merged mask back into the attention call’s kwargs (or positional
   `attention_mask` argument).

The model then runs SDPA (this env has no flash-attn). Softmax / SDPA
renormalizes over the remaining allowed keys, so blocked edges get zero
weight and other keys absorb the mass.

Layer windows are coarse bands of 10 consecutive layers, stride 5, with a
final right-aligned window (LLaVA 32 layers → six windows; Qwen 28 → five),
plus one **all-layers** pass.

Scoring is prefill-only: `forward_with_knockout` → last-position logits →
yes/no token sets via `ScoringTarget.yes_no` (same path as
`forward_with_logits` in `src/mediation.py`). No generation.

### What Gate 3 was asking

If you block **all** layers and **all** downstream queries from reading the
clause, and the flip was mostly “the model read the clause’s yes/no content,”
answers on flipped items should move back toward the **neutral** (no-prefix)
first-token answer. We measured that agreement at **51.4%** on LLaVA (37 flip
evaluations), below the plan’s ~80% bar — so the mask is doing *something*,
but full-depth clause knockout does not restore neutral behavior on most
flips.

---

## 2. Eager-mode zero-attention assertion — what was verified

Function: `verify_knockout_zero_attention` in `src/attention_knockout.py`.

SDPA does not materialize attention weights, so for a **one-item check** we:

1. Temporarily set the model’s attention implementation toward **eager**
   (`config._attn_implementation = "eager"`, and the model attribute when
   present). Transformers may still log that SDPA doesn’t support
   `output_attentions` and fall back to eager for that forward — that is
   expected.
2. Run one VL prefill with the knockout hooks active and
   `output_attentions=True`.
3. For each hooked layer, read the attention tensor `(heads, Q, K)` and take
   the maximum weight on every blocked `(q, k)` pair.
4. Assert that maximum is ≤ `atol` (default `1e-6`).

**LLaVA result (before the window sweep):** item `amber_disc_00326`, condition
`assertive_toward_no`, layers 15–24, all-downstream query scope:

```text
ok=True, max_blocked_attention=0.0, checked_layers=[15..24], family=llava
```

**What that means.** On that forward, every blocked edge had **exactly zero**
post-softmax weight. So the mask is reaching the attention computation and
zeroing the intended edges; Gate 3’s low recovery is **not** explained by
“the knockout silently failed to apply.” It means: even with those edges
truly cut at all layers, the first-token answer often still does not match
neutral — residual effects (prefix presence / position / dilution / other
pathways) remain.

It does **not** by itself prove the clause span indices are semantically
perfect for every item; it proves that whatever positions we marked as blocked
received zero attention under the verify hooks. Span derivation is separate
(`src/prompt_spans.py`: question tokens as suffix of `tok(prefix+question)`,
clause = that many tokens before the question match in the multimodal
`input_ids`).

---

## 3. How filler–neutral agreeability was calculated

Artifact: `filler_vs_neutral_behavior_summary.csv` per cell  
Code: `analyze_knockout.py` (“Filler vs neutral (no knockout)” block).

**Inputs.** For every item in the flip set ∪ stable set (LLaVA: 54 items), the
knockout run stored **no-knockout** first-token scores for:

- `neutral` — bare benchmark question  
- `filler_b` — length-matched inert prefix from
  `templates/filler_clauses_v1.json`

**Answer agreement with neutral** (per filler paraphrase):

$$
\frac{1}{N} \sum_{i} \mathbf{1}\!\left[
  \operatorname{pred}(\mathrm{filler}_{i}) = \operatorname{pred}(\mathrm{neutral}_{i})
\right]
$$

where `pred` is `scores.first_token_pred` (`"yes"` if summed yes-variant
probability ≥ summed no-variant probability, else `"no"`).

**Mean yes-probability shift** (same items):

$$
\frac{1}{N} \sum_{i} \Bigl(
  p_{\mathrm{yes}}(\mathrm{filler}_{i}) - p_{\mathrm{yes}}(\mathrm{neutral}_{i})
\Bigr)
$$

with $p_{\mathrm{yes}}$ = `scores.p_yes_raw` (summed first-token probability
mass on the yes token-id set).

**LLaVA numbers (all-downstream cell, n=54; regenerated 2026-07-20 on
probability scale):**

| Filler | Answer agreement with neutral | Mean Δ $p_{\mathrm{yes}}$ |
|--------|-------------------------------|---------------------------|
| filler_b | 0.630 | +0.084 |

So on this item pool, an inert prefix already changes the first-token answer
on roughly **37–43%** of items relative to neutral. That is the plan’s
“attention-dilution / any-prefix” reference, reported once per model (same
no-knockout scores appear in both LLaVA cells).

---

## Related: overrides Alex gave 2026-07-20

- Proceed with **Qwen** knockout despite flip union 11 and decidedness ~98%.
- Continuous reporting metrics use **yes-token probability** (`p_yes_raw`), not
  logits; ε is recomputed on the probability scale (10th pct of `|Δ p_yes|`).
- Gate 3 remains concerning; this note documents mechanism/verify/filler so
  the analyst can interpret without assuming the knockout failed to apply.
