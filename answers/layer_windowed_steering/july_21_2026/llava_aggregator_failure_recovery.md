# LLaVA JSON decode failure — investigation & recovery

## Cause

POPE-30 dump **completed successfully** (63/63 cells). The Stage A shell then ran the aggregator, which crashed on:

`data/amber/dumps/llava-1.5-7b-hf/amber100_baseline/baseline/manifest.jsonl`

That file is **pretty-printed** (multi-line objects). `load_manifest` read it line-by-line as JSONL → `JSONDecodeError`. Because `run_llava_windowed_steering_stage_a.sh` uses `set -e`, the crash **skipped A2 (AMBER-100)** entirely.

## Artifact loss

| Artifact | Status |
|----------|--------|
| LLaVA POPE-30 steered dumps (63 cells, acts/norms/manifests) | **Intact — nothing lost** |
| LLaVA AMBER-100 steered dumps | **Never started** — nothing written, nothing lost |
| Aggregator consolidated JSON | Was missing; **rebuilt** after fixing parser |

## Fix

- Aggregator now accepts pretty-printed concatenated JSON (same pattern as knockout `iter_json_records`).
- Stage scripts no longer abort dumps if the aggregator fails.

## Inventory

```json
{
  "date": "2026-07-21T12:38:29",
  "failure": {
    "where": "aggregator after A1 (not during dump)",
    "cause": "LLaVA amber100_baseline/baseline/manifest.jsonl is pretty-printed multi-line JSON; load_manifest assumed single-line JSONL",
    "effect": "stage_a.sh set -e aborted before A2 AMBER-100 dump started"
  },
  "recovered": {
    "pope30_windowed_steering": {
      "path": "data/pope/dumps/llava-1.5-7b-hf/pope30_windowed_steering",
      "n_cells_complete": 63,
      "n_cells_expected": 63,
      "n_manifest_records": 3780,
      "lost": false
    },
    "amber100_windowed_steering": {
      "path": "data/amber/dumps/llava-1.5-7b-hf/amber100_windowed_steering",
      "status": "never_started",
      "lost": false,
      "note": "No AMBER steered dumps were written; nothing to recover or lose."
    },
    "consolidated_json": "/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/diagnostic_experiments/perception_diag/windowed_steering_summary/windowed_steering_consolidated_results.json"
  }
}
```
