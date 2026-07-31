# knockout_records.jsonl now includes full prompts (2026-07-20)

Each entry in all four cell `knockout_records.jsonl` files now has:

- `prompt` — full text scored for that `(item_id, condition_id)`
- `question_neutral` — the bare yes/no question

Joined offline from the perception_diag augment JSONLs (no GPU re-run). `run_knockout.py` writes these fields on future runs.

Paths:

- `diagnostic_experiments/leading_clause_attention_knockout/llava-1.5-7b-hf/results/llava_block_last_token_reading/knockout_records.jsonl`
- `…/llava_block_all_downstream_reading/knockout_records.jsonl`
- `…/qwen2.5-vl-7b-instruct/results/qwen25_block_last_token_reading/knockout_records.jsonl`
- `…/qwen25_block_all_downstream_reading/knockout_records.jsonl`
