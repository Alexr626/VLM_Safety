# One POPE example: neutral vs leading toward no (Qwen) — 2026-07-21

**Path:** `diagnostic_experiments/perception_diag/windowed_steering_summary/qwen_pope_one_example_neutral_vs_leading_toward_no.html`

Single item `pope_adversarial_00000` (gold=yes). Side-by-side:

- Neutral: “Is there a snowboard in the image?” → **yes** (P(yes)=0.87)
- Leading toward no: “The correct answer is clearly no. …” → **no** (P(yes)=0.18)

Qwen2.5-VL no-intervention baseline.
