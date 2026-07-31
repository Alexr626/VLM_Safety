# AMBER-100 LLaVA response galleries + summary cleanup

**Date:** 2026-07-22

## Galleries

1. **Relation gold=yes, leading toward no (baseline)**  
   `…/windowed_steering_summary/amber100_relation_gold_yes_leading_toward_no_baseline_response_gallery.html`  
   n=20; baseline parsed outcomes in header (no=18, yes=2).

2. **Attribute + relation gold=no, additive @ mlp layers 0–9**  
   `…/windowed_steering_summary/amber100_attribute_relation_gold_no_additive_mlp_layers_0_9_response_gallery.html`  
   Both neutral and assertive_toward_yes; β∈{0.2,0.5,0.9}; red border = flip vs baseline.

```
python diagnostic_experiments/perception_diag/build_amber100_llava_response_galleries.py
```

## Cleanup

Removed stale POPE HTML/JSON under `windowed_steering_summary/` that referenced deleted dump tags or the pre-split `pope30_mlp_2x2_*` summaries. Kept current `pope30_{yes,no}_mlp_2x2_*.json`, consolidated JSON, and `plots/`.
