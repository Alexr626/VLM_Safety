# Gate results explained — attention knockout (2026-07-20)

Plain-language walkthrough of each “gate” from the plan and what happened.

## Gate A — Flip-set size (≥ ~15 items)

**What it checks.** Before spending GPU time on knockout, Stage 0 builds the
**flip set**: items the model gets **right** on the bare (neutral) question and
**wrong** once a misleading leading clause is prepended. The plan says: if that
set has fewer than ~15 items for a model, localization is underpowered — stop
and discuss enlarging the pool.

**What we saw.**

| Model | Flip-set size | Pass? |
|-------|---------------|-------|
| LLaVA-1.5-7B | 27 | Yes |
| Qwen2.5-VL-7B | 11 | No |

So LLaVA was allowed to proceed into knockout. Qwen was **not** — we never ran
Qwen knockout cells. That is a sample-size stop, not a claim about Qwen’s
mechanism.

(For context: LLaVA’s 27 flips are almost all from AMBER-100; POPE-30 added 0.
Qwen had 8 on AMBER-100 + 3 on POPE-30.)

---

## Gate B — First-token decidedness (≥ ~99%)

**What it checks.** Knockout scores the **first generated token** (yes vs no
logits), not the full string. Stage 0 checks that, on the existing baseline
dumps, the **parsed** yes/no from the generated response matches that
first-token argmax. If they often disagree, first-token scores are a shaky
proxy for “the model’s answer.”

**What we saw.**

| Model | Dump | Agreement | Pass? |
|-------|------|-----------|-------|
| LLaVA | AMBER-100, POPE-30 | 100% | Yes |
| Qwen | AMBER-100 | 98.5% (457/464 parseable) | Soft fail |
| Qwen | POPE-30 | 98.0% (147/150) | Soft fail |

Qwen’s misses were few: 7 AMBER cases where generation text parsed opposite to
the first-token score, and 3 POPE cases where `p_yes == p_no` (tie) so the
first-token rule picked “yes” while the string parsed as “no.” The plan said
“investigate before proceeding if below ~99%.” We flagged it; it did **not**
by itself block LLaVA. It is extra reason to be careful before trusting Qwen
first-token metrics if/when the flip-set gate is fixed.

---

## Gate C — Full-depth anchor (≥ ~80% agreement with neutral)

**What it checks (this is the confusing one).**  
Imagine knocking out the leading-clause tokens at **every** decoder layer, for
**all** positions after the clause (the “all-downstream” scope). If the flip
were mostly caused by the model *reading the clause content*, cutting that
read everywhere should push the answer back toward the **neutral** (no-clause)
answer on flipped items.

The plan’s threshold: on the flip set, that full-depth knockout should match
the neutral answer at least ~80% of the time. If it does **not**, a lot of the
flip is probably about “there is a prefix at all” (position / length /
dilution), not about the clause’s yes/no wording — and then per-window recovery
curves are hard to read as “which layers read the clause.”

**What we saw (LLaVA, both scopes).**

- Flip evaluations: **37** (27 flip items × the misleading conditions that
  actually flipped them; some items flip under both tentative and assertive).
- Agreement with neutral under **all-layers** clause knockout: **51.4%**
- Pass (≥ 80%)? **No**

So mechanically the mask works (eager check: blocked edges had attention mass
0), but **blocking the clause at all layers does not restore neutral behavior
on most flipped items**. That is why the plan says: report and stop for
discussion before treating the layer-window curves as layer-localization
evidence.

Related numbers that sit next to this gate (not separate plan gates, but why
the failure is unsurprising):

- **Filler vs neutral** (no knockout): only ~57–63% answer agreement — an inert
  length-matched prefix already moves many answers.
- **Filler-span knockout** at the peak window recovered ~72%, similar to clause
  knockout there — not the “near null” specificity the plan hoped for.

---

## Gate D — ε for logit-difference recovery (confirm with analyst)

**What it is.** Not a pass/fail stop. For the continuous “how much of the
logit gap did knockout close?” metric, items with a tiny neutral−leading gap
are excluded (noisy denominators). Stage 0 proposed ε = **1.0546875** = 10th
percentile of \|margin_neutral − margin_leading\| on flip rows. Analysis used
that value; the plan asked the analyst to confirm before calling plots final.

---

## Quick map: what ran vs what stopped

```
Stage 0
  ├─ LLaVA flip set 27  → OK → ran knockout + analysis
  ├─ Qwen flip set 11   → STOP (Gate A) — no Qwen knockout
  └─ Qwen decidedness ~98% → FLAG (Gate B) — investigate if enlarging Qwen pool

LLaVA knockout
  ├─ Eager zero-attention verify → OK
  ├─ Full-depth anchor 51% → STOP interpreting windows (Gate C)
  └─ ε = 1.05 → provisional (Gate D)
```
