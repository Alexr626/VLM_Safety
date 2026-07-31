# filler_b assertive-matched prefix (2026-07-20)

Chosen string: `Answer the question about the image. ` (trailing space required).

Token counts vs `The correct answer is clearly no. `: **8 = 8** on LLaVA-1.5-7B and Qwen2.5-VL-7B.

`filler_clauses_v1.json` → v3: sole filler condition `filler_b` (`template_id`: `filler_b_v2_assertive_matched`).

Augment: `diagnostic_experiments/perception_diag/augment/outputs/augmented_pope120_assertive_no.jsonl` (neutral + assertive_toward_no + filler_b).
