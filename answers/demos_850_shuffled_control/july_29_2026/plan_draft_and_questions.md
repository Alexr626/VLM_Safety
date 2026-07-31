# demos_850 shuffled-control plan — clarifying questions

Date: 2026-07-29

## Draft plan

`implementation_plans/7-29-26/demos_850_shuffled_control_per_partition_block_plan_2026-07-29.md`

## What the 7-22 plans require for a valid control

From the sanity plan (construction):

- Same ids / order / captions as the deployed cell; images deranged (0 fixed points).
- Separate act-cache namespace (cache key omits image).
- Gating: fixed points 0, captions unchanged N/N, id-order mismatches 0, shape OK, forwards prove no wrong-cache reuse.
- No cosine / behavioral runs in the construction plan.

From the geometric plan (follow-up): CPU cosine + magnitude vs deployed, only after gating PASS.

## Defaults assumed in the draft

- Dimension **`all` only** (8 cells), not all five dimensions.
- Construction + sanity **only**; geometric comparison as a later plan.
- One derangement **per block size** over that block’s ids, seed **1234**.
- New namespace `shuffled_control_demos850/` so demos_v2 `shuffled_control/all_nd200` stays intact.
- Deployed targets: `demos850_ba05bd96_all_nd{N}_s42_r2_partition` (ids verified equal to partition blocks).

## Questions for Alex

1. Controls for dimension `all` only, or all five dimensions?
2. Include geometric comparison in this plan, or stage it after sanity like 7-22?
3. Confirm per-block derangements with seed 1234 (vs one global 850-id derangement)?
