# POPE `yes_ratio`, the confusion-matrix metrics, and the decode-only / skip-pos0 steering probes

Date: 2026-06-19
Context: clarifying the metrics behind the VTI reproduction and the VTI
rotation-strength experiment. Code reference: `evaluation/classifiers/metrics.py`
(`score_pope_records`, `_normalize_yes_no`, `_prf1`).

---

## 1. What `yes_ratio` is — and where the "actual vs predicted" dimension lives

### POPE setup
POPE asks one yes/no question per sample: *"Is there a [object] in the image?"*
- Ground truth **yes** = the object really is present.
- Ground truth **no** = the object is absent.
- The benchmark is **balanced**: ~50% of samples are yes, ~50% are no.
- Positive class = **yes** (POPE/VTI convention).

A model answer is parsed to `yes` / `no` / `None` (unparseable) by
`_normalize_yes_no` (leading-token priority, then word-boundaried fallback).

### `yes_ratio` definition
```
yes_ratio = (tp + fp) / total = (# samples the model PREDICTED "yes") / (total parsed-gt samples)
```
It is the **fraction of the time the model answered "yes"**, full stop. It is a
property of the model's *answer distribution*, **not** of correctness:

- `yes_ratio ≈ 0.5` → unbiased on balanced POPE.
- `yes_ratio > 0.5` → **yes-bias**: the model over-claims objects are present.
- `yes_ratio < 0.5` → **no-bias**: the model is over-skeptical, under-claims.

So `yes_ratio` does **not** measure actual-vs-predicted correctness. Two models
with identical accuracy can have very different `yes_ratio`, and a model can have
`yes_ratio = 0.5` while being completely wrong (e.g. always wrong but balanced).

**Why we still care:** on POPE, a **false positive (fp) = a hallucination**
(model says "yes, the object is there" when it is not). Yes-bias therefore
correlates with hallucination propensity, which is exactly the failure mode VTI
targets. `yes_ratio` is a cheap, direction-carrying proxy for "is this model
hallucinating more or less?" — that is why POPE/VTI papers report it alongside
accuracy/F1.

### The actual-vs-predicted dimension IS measured (just not always surfaced)
Your instinct is right that `yes_ratio` alone ignores the confusion matrix — but
we **do** compute the full confusion matrix. `score_pope_records` accumulates
`tp / fp / fn / tn` (positive = yes) and returns:

| field | formula | what it penalizes |
|-------|---------|-------------------|
| `accuracy_overall` | (tp+tn)/total | overall correctness |
| `precision_overall` | tp/(tp+fp) | **hallucinations** (fp): saying "yes" when absent |
| `recall_overall` | tp/(tp+fn) | **misses** (fn): saying "no" when present (over-skepticism) |
| `f1_overall` | harmonic mean of P,R | balance of the two |
| `yes_ratio` | (tp+fp)/total | answer bias (not correctness) |

These are written to every `metric_summary.json` in the reproduction harness. So
the actual-vs-predicted dimension exists end-to-end — precision and recall are
the metrics that capture it.

### The real gap is in the rotation-strength experiment
The gap you sensed is specific to the **VTI rotation-strength** experiment
(`evaluation/vti_rotation_strength/rotation_strength.py`). Its per-beta summary
(`_score`) surfaces only `accuracy`, `yes_ratio`, `f1`, `n_unparsed`,
`mean_len` — it drops `precision`/`recall` and the raw `tp/fp/fn/tn`.

And the **decision-flip counts** (`c->w`, `w->c`) measure *changes in
correctness*, but **not the direction in confusion-matrix terms**. A
correct→wrong flip could be either:
- `tp → fn` : a true "yes" that became "no" (steering suppressed a correct yes), or
- `tn → fp` : a true "no" that became "yes" (steering induced a hallucination).

Right now we can't tell those apart from the logged numbers. That distinction is
exactly what tells you *whether rotation reduces hallucinations or just shifts
the yes/no operating point*. **Recommended next step:** add
`precision`/`recall` and a yes→no vs no→yes (equivalently Δtp/Δfp/Δfn/Δtn)
decomposition of the flips to the rotation-strength output. (Not yet done; flagged
for implementation.)

---

## 2. `decode_only`, `skip_pos0`, and what "inert" means

Two forward-pass phases matter when steering during generation:

- **Prefill**: one big forward pass over the *entire* prompt + image, run once
  before any token is generated. This is where the image+question get
  contextualized and where the yes/no decision is effectively formed.
- **Decode**: the incremental steps that emit the answer one token at a time
  (seq length 1 each). For POPE answers ("Yes."/"No.") there are only a handful.

The hook context `vti_hook_ctx` has two debug knobs:

- **`steer_prefill=False` → "decode_only"**: do **not** steer the prefill; steer
  only the decode steps.
- **`skip_first_token=True` → "skip_pos0"**: steer the prefill, but leave
  **sequence position 0** (the first/BOS token) unsteered.

Both probes were measured **at the strongest beta (0.6)** in the sweep; the
per-beta rows in the table used the production defaults (`steer_prefill=True`,
`skip_first_token=False`, i.e. steer everything everywhere).

### "Inert" = produced no measurable change vs the no-hook baseline
When I called `decode_only` **inert**, I meant: at beta=0.6, for **all four
models**, decode-only steering gave `accuracy = baseline`, `yes_ratio =
baseline`, and `flips = 0`. It changed nothing — as if the hooks were not there.

Interpretation: **essentially the entire steering effect comes from the prefill
pass, not from decode-time nudging.** By the time decoding begins, the yes/no
decision is already encoded in the prefill representations; rotating the few
decode-step activations does not move it (and there are very few decode steps for
a 1–2 token answer anyway).

By contrast, **`skip_pos0` at beta=0.6 rescued LLaVA from collapse** (acc
0.55→0.87, mean_len 3.6→36.3, no empty outputs) while still flipping 22
decisions. So within the prefill, steering **position 0 specifically** is what
caused the degenerate "emit EOS immediately / empty output" collapse at high
beta.

Together the two probes **localize the effect**:
1. all the steering action is in prefill (decode-only does nothing), and
2. within prefill, position-0 steering is the thing that breaks the model at high
   strength (skipping it averts the collapse).

### Why the attention sink matters (re-explanation)
In transformer LLMs the **first token (often BOS) becomes an "attention sink"**:
across many heads and many layers, a large share of attention weight is dumped
onto position 0 regardless of content. It functions as a default "park here"
target — when a head has nothing informative to attend to, it attends to the
sink. Two consequences:

1. **Every later position reads from position 0.** So if you corrupt position-0's
   residual at every layer (which is what rotating it during prefill does), you
   are perturbing a representation that the *entire* sequence attends to, and the
   damage propagates globally — the model destabilizes and collapses to immediate
   EOS.
2. The sink role is about **attention routing, not necessarily norm.** On LLaVA
   the position-0 norm was actually modest (~8.3 in the single-sample diagnostic),
   yet leaving it unsteered still prevents the collapse — implicating its **sink
   role** (everyone attends to it) rather than its magnitude.

That is why `skip_pos0` is the targeted fix: it preserves the global attention
sink while still steering all the content-bearing positions.

---

## 3. Qwen2.5 results, read through `yes_ratio`

Now that `yes_ratio` is defined, Qwen2.5's behavior is interpretable.

**Baseline (no hooks):** Qwen2.5-VL has `yes_ratio = 0.37` (vs ~0.43–0.49 for
the other three models) and long outputs (mean_len 154 vs 50–87). So at baseline
it is **no-biased / over-skeptical** — it answers "yes" only 37% of the time on a
50/50 benchmark, i.e. it under-claims objects (recall-limited, fn-heavy).

**Under layer-site rotation:** its `yes_ratio` **rises toward 0.5** (0.465 at
beta=0.5) and accuracy rises **above** baseline (0.925 @ β=0.5 vs 0.87). It is
the only model where wrong→correct flips dominate (`w->c > c->w`) at every beta.

**Contrast with LLaVA:** LLaVA's baseline `yes_ratio` is already ~0.49
(balanced). The *same* rotation pushes its `yes_ratio` **down** to 0.05 at
beta=0.6 (extreme no-bias / suppression), and accuracy **crashes** (0.89→0.55)
with correct→wrong flips dominating.

So the same construction moves `yes_ratio` in **opposite directions** across
models, and rotation only "helps" the model (Qwen2.5) that happened to start
biased in the direction that benefits from the shift on a balanced benchmark.

### The crucial caveat (for the analysis agent)
Qwen2.5's apparent gain is **confounded with bias-correction**: because it
started at 0.37, *any* nudge toward 0.5 mechanically raises accuracy on a
balanced 50/50 set — independent of whether hallucinations actually decreased.

To disentangle "genuine hallucination reduction" from "yes/no rebalancing", look
at **precision vs recall separately** (this is the direct payoff of §1's
recommendation):
- **Genuine hallucination reduction** → **precision** goes up (fewer fp) without
  simply trading away recall.
- **Pure rebalancing** of a no-biased model → mostly **recall** goes up (it adds
  "yes" answers, converting fn→tp), with precision flat or down.

We cannot make that call yet because the rotation-strength experiment does not
surface precision/recall or the confusion-matrix flip decomposition — which is
why §1 recommends adding them.

---

## TL;DR
- `yes_ratio` = fraction of "yes" predictions = answer-distribution / bias metric;
  on balanced POPE, 0.5 = unbiased, and yes-bias ≈ hallucination tendency (fp).
  It does **not** measure correctness.
- Correctness *is* measured: accuracy/precision/recall/F1 from the tp/fp/fn/tn
  confusion matrix, written to every `metric_summary.json`. The rotation-strength
  experiment just doesn't surface precision/recall or decompose its flips by
  confusion-matrix direction — recommend adding that.
- "Inert" = the probe changed nothing vs the no-hook baseline (same acc/yes_ratio,
  0 flips). `decode_only` was inert → the steering effect lives in **prefill**.
  `skip_pos0` rescued LLaVA from collapse → **position-0 (attention-sink) prefill
  rotation** is what breaks the model at high beta.
- Attention sink = first token that most heads/layers dump attention onto; because
  everything reads from it, corrupting it propagates globally → collapse. Skipping
  it preserves stability.
- Qwen2.5 starts no-biased (`yes_ratio` 0.37); rotation pushes it toward 0.5 and
  raises accuracy (net w→c), unlike the others. But this is confounded with
  bias-correction; precision vs recall is needed to tell rebalancing from real
  hallucination reduction.
