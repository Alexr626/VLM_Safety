# Existence grammar polish + full-300 nohup

## Code changes

1. **`deterministic_existence_insert`** now treats `includes` / `features` / `shows` / `contains` / `has` like `with`/`including`, so the bedroom case becomes:
   - `includes a chair, a bed, a person, and a tv…`
   - not `a chair, includes a bed…`
2. **Stage 3** always runs a **text-only Haiku grammar polish** after the deterministic draft (`STAGE3_GRAMMAR_SYSTEM`), then re-validates. Full LLM existence insert remains fallback only.
3. Driver: `data_scripts/vti_demos_v2/run_full300.sh` — wipes stage1–5, keeps stage0 (300), no `--limit`, assembles with `--n-final 300` (keep every stage-4 pass).

## nohup (full pool, maximize finals)

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
mkdir -p data/vti/v2/logs

nohup bash data_scripts/vti_demos_v2/run_full300.sh \
  > data/vti/v2/logs/full300_nohup.out 2>&1 &

echo $!
tail -f data/vti/v2/logs/full300_nohup.out
```

The script also tees a stamped log under `data/vti/v2/logs/full300_YYYYMMDD_HHMMSS.log`.

**Yield knobs already set for max finals:** no stage limits; `--n-final 300`; fresh stage1–5 so old rejects are retried with the new existence path.

Expect heavy Anthropic spend (stage1/2/4 vision on up to 300; stage3 ≈1 cheap text call per pass). Resume-by-id still applies if you re-run a single stage after a crash without wiping.
