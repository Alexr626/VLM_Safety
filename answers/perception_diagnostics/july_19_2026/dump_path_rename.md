# Perception dump path rename (plain English)

**Date:** 2026-07-19

| Former | New |
|--------|-----|
| `dumps/s1_full_cell_smoke/` | `dumps/amber25_all_steering_settings/` |
| `dumps/s3b_qwen25_amber25/` | `dumps/amber25_steering_settings/` |
| `B0`…`B12` | `baseline`, `rotation_mlp_0.2`, `rotation_mlp_0.5`, … `additive_layer_0.9` |
| `run_s1_full_cell_smoke.sh` | `run_llava_amber25_dump.sh` |
| `run_s3b_qwen25_amber25.sh` | `run_qwen25_amber25_dump.sh` |

`cell_id` in manifests and `run_tag` in `run_metadata.json` updated to match. `IMPLEMENTATION.md` / `RESEARCH_LOG.md` updated.
