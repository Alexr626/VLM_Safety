# Split demos_v2 qualitative HTML galleries

**Date:** 2026-07-14

## Layout

```
evaluation/results/2026-07-13/_samples/{model}/{benchmark}/
  {additive_layer|additive_mlp|uniform_rotation_layer|uniform_rotation_mlp}/
    {all|existence|attribute|counting|relation}/
      {50|100|200|500}/review.html
```

Example path requested:
`…/llava-1.5-7b-hf/amber/additive_layer/counting/100/review.html`

## Fixes / features

- **320** galleries written (2 models × 2 benches × 4 interventions × 5 dims × 4 nd).
- Includes `uniform_rotation_layer` and `uniform_rotation_mlp` (old per-dim mega-HTML regex only matched `additive_*`).
- CHAIR pages annotate each response with per-caption `CHAIR_s` / `CHAIR_i` (plus hallucinated object list) in the yellow note next to the response.
- Each page columns: baseline + β∈{0.5, 0.2, 0.9} for that intervention.

## Regenerator

```bash
python helper_scripts/render_demosv2_qual_split_review.py --run_date 2026-07-13
```
