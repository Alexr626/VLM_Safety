# S1 LLaVA AMBER-25 HTML review galleries

Generated with `helper_scripts/render_perception_s1_review.py` (same sticky-image gallery layout as demos_v2 qual reviews).

**Start here:**  
`evaluation/results/2026-07-16/_samples/llava-1.5-7b-hf/amber/llava-1.5-7b-hf_amber_perception_s1_index.html`

| Page | Contents |
|------|----------|
| `…_perception_s1_B{0–12}_review.html` | One experimental cell; 25 items × 5 leading-clause conditions |
| `…_perception_s1_compare_neutral_review.html` | All cells side-by-side, **neutral** prompt only |
| `…_perception_s1_index.html` | Links to all of the above |

Each response panel notes `parsed=… · mass=… · p_yes=…`. Led conditions include the full leading-clause prompt above the model response.

**Regenerate:**
```bash
python helper_scripts/render_perception_s1_review.py \
  --dump_dir diagnostic_experiments/perception_diag/llava-1.5-7b-hf/dumps/s1_full_cell_smoke \
  --run_date 2026-07-16
```
