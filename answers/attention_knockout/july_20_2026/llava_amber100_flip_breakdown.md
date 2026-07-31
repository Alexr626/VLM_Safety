# Why LLaVA flipped often on AMBER-100 (Stage 0 breakdown)

**Headline:** All **27 / 27** LLaVA flip-union items are from **AMBER-100**. **0 / 30** from POPE-30 (existence×yes). That alone explains why POPE-only pins yielded few LLaVA flips.

## Flip rate by qtype × gold (AMBER-100, 20 items per cell except no existence×yes)

| Cell | Flips / items | Flip rate | Mean neutral p(yes) flip | Mean neutral p(yes) non-flip |
|------|---------------|-----------|--------------------------|------------------------------|
| **relation × yes** | **11 / 20** | **55%** | 0.57 | 0.50 |
| **relation × no** | **8 / 20** | **40%** | 0.33 | 0.28 |
| attribute × no | 3 / 20 | 15% | 0.36 | 0.38 |
| existence × no | 3 / 20 | 15% | 0.25 | 0.14 |
| attribute × yes | 2 / 20 | 10% | 0.57 | 0.78 |

**19 / 27** flips are **relation** (70%). Attribute + existence contribute only 8.

## Misleading condition counts (item can appear in multiple)
- tentative_toward_yes: 14  
- assertive_toward_no: 13  
- tentative_toward_no: 6  
- assertive_toward_yes: 4  

## Neutral p(yes) (flip vs non-flip)
- Gold=yes: flip mean **0.57** vs non-flip **0.69** (flips slightly less confident yes)
- Gold=no: flip mean **0.32** vs non-flip **0.27**

## Artifacts
`diagnostic_experiments/leading_clause_attention_knockout/llava-1.5-7b-hf/results/stage0/amber100_flip_breakdown/`

Plots (also copied under `…/results/plots/`):
- `llava_amber100_flip_count_by_qtype_and_gold.png`
- `llava_amber100_flip_rate_by_qtype_and_gold.png`
- `llava_amber100_neutral_yes_probability_flip_vs_nonflip_by_gold.png`
- `llava_amber100_flip_yes_probability_neutral_vs_misleading.png`
- `llava_amber100_flip_items_by_misleading_condition.png`
- `llava_stage0_flip_union_amber100_vs_pope30.png`
- `llava_amber100_mean_neutral_yes_probability_flip_vs_nonflip_by_cell.png`
