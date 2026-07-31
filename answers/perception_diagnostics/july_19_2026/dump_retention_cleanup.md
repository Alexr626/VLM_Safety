# Perception diag cleanup — dumps retained only

**Date:** 2026-07-19

Kept dumps + creation scripts; removed Package A/C analyses, experiment plans, RunAI perception helpers, derived plots/JSON, S0 dump, and HTML galleries.

**Kept:**
- `…/llava-1.5-7b-hf/dumps/s1_full_cell_smoke/`
- `…/qwen2.5-vl-7b-instruct/dumps/s3b_qwen25_amber25/`
- `run_dump.py`, `capture/`, `augment/` (+ amber25 JSONL), `templates/leading_clauses_v1.json`, `run_scripts/run_s1_*.sh`, `run_scripts/run_s3b_*.sh`

**Docs:** `IMPLEMENTATION.md` perception § and `RESEARCH_LOG.md` (from 2026-07-16 perception block) rewritten to dump-creation facts only.
