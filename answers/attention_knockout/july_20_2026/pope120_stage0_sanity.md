# POPE-120 assertive protocol — Stage 0 sanity (2026-07-20)

## Protocol
- Pin: `data/pope/pinned_pope_existence_yes_120.json` (40 gold=yes × random/popular/adversarial)
- Conditions: `neutral`, `assertive_toward_no`, `filler_b` only
- `filler_b` = `Answer the question about the image. ` (8 tokens = assertive on LLaVA + Qwen)

## Prefix length
`assertive_toward_no` = `filler_b` = **8 tokens** on both models.

## Stage 0 results

| Model | first-token↔parsed | flip_union | Gate (≥15) |
|-------|--------------------|------------|------------|
| LLaVA-1.5-7B | 0.9917 | **6** | FAIL |
| Qwen2.5-VL-7B | 1.0000 | **24** | PASS |

## Filler vs neutral (no knockout, n=120)
| Model | answer agree with neutral | mean Δp(yes) filler−neutral |
|-------|---------------------------|-----------------------------|
| LLaVA | 0.975 | −0.080 |
| Qwen | 1.000 | +0.002 |

Length-matched `filler_b` tracks neutral closely (especially Qwen). LLaVA barely flips under assertive_toward_no on this set (6/120).

## Artifacts
`diagnostic_experiments/leading_clause_attention_knockout/results/stage0_pope120/`

Full knockout not started — LLaVA flip gate failed.
