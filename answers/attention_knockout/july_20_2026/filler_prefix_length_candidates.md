# Filler prefix length vs assertive_toward_no (2026-07-20)

Target prefix (`assertive_toward_no`): `"The correct answer is clearly no. "`
Token count: **8** on LLaVA-1.5-7B and Qwen2.5-VL-7B.

Current `filler_b`: 15 tokens on both (Δ+7) — matched to tentative, not assertive.

## Exact matches (8 tokens on both models)

| Prefix | Notes |
|--------|--------|
| `Answer this question about this picture. ` | User suggestion; **needs trailing space** (without space → 7) |
| `What is the answer to this question?` | User suggestion; **no trailing space** (with space → 9) |
| `Answer the question about the image. ` | |
| `I am asking about this picture. ` | |
| `Look at the picture and answer. ` | |
| `Look at this picture and reply. ` | |
| `Please look at this picture carefully. ` | |
| `Regard this picture and respond. ` | |

Artifact: `diagnostic_experiments/leading_clause_attention_knockout/results/stage0_pope120/filler_prefix_length_candidates.json`

POPE-120 baseline dumps were stopped before continuing; wait for filler choice before re-running.
