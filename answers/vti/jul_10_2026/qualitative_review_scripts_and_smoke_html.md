# Qualitative review scripts and July 2 smoke HTML galleries

## Original textual-experiment galleries (2026-06-22)

**Scripts:**
- `helper_scripts/review_responses.py` — single-id or ad-hoc multi-id gallery CLI
- `helper_scripts/sample_responses.py` — batch extractor for a full diagnostic `run_date`
- `helper_scripts/run_sample_responses.sh` — wrapper (`RUN_DATE=2026-06-22`)

**Shared library:** `helper_scripts/review_lib.py` (HTML builder, image embedding, response parsing)

**June 22 HTML outputs (textual Exp1+Exp2 on pinned subsets):**
- `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/amber/llava-1.5-7b-hf_amber_review.html`
- `evaluation/results/2026-06-22/_samples/llava-1.5-7b-hf/chair/llava-1.5-7b-hf_chair_review.html`
- (also Qwen2.5 variants under `_samples/qwen2.5-vl-7b-instruct/`)

Sample IDs come from `*_response_samples.json` in the same directories (AMBER-25 stratified + CHAIR-5).

## July 2 smoke galleries (textual + visual cells)

**New script:** `helper_scripts/render_smoke_review.py`  
**Wrapper:** `helper_scripts/run_render_smoke_review.sh`

```bash
RUN_DATE=2026-07-02 bash helper_scripts/run_render_smoke_review.sh
```

**Outputs:**
- `evaluation/results/2026-07-02/_samples/llava-1.5-7b-hf/amber/llava-1.5-7b-hf_amber_smoke_review.html`
- `evaluation/results/2026-07-02/_samples/llava-1.5-7b-hf/chair/llava-1.5-7b-hf_chair_smoke_review.html`

Each page shows all **16 intervention cells** per sample (baseline + 3 textual + 12 visual), using the same pinned ids as the June bundle.

## Textual vs visual in July 2 run

The July 2 smoke included **both** arms (Stage 0 textual gate control + Stages 1/1b visual grid). The new `*_smoke_review.html` files show all cells side-by-side for qualitative comparison.
