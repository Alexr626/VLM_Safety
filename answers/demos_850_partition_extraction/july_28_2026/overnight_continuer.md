# Overnight continuer — go to bed checklist

Date: 2026-07-28 ~22:24

## What was set up

Not a Cursor agent (IDE disconnect would stop it). A **nohup bash continuer** on lambdab2:

- Script: `helper_scripts/run_demos850_continue_after_topup.sh`
- Waiting on live top-up PID **416721**
- Continuer bash PID **585996** (child 586017 from process-subst tee)

## What it will do after stages 1–4 finish

1. Check `demos_v2.jsonl` still `9a44f4af…` + stage jsonl append-only vs snapshot
2. If admissible new stage-4 passes &lt; 295 → one mining round-2 (300 candidates) then stages 1–4 again; if still short → **stop** (no block-size change)
3. `assemble_demos_850.py --write-partition`
4. Extract LLaVA then Qwen on a free GPU (`--check_cache_fidelity`, Qwen `--max_pixels 1003520`)
5. Verify both models
6. Append a factual entry to `RESEARCH_LOG.md`

## Morning checks

```bash
cat ~/dev/vlm_hallucination_mitigation_summer_2026/logs/demos850_overnight_status.json
tail -50 ~/dev/vlm_hallucination_mitigation_summer_2026/logs/demos850_overnight_continue_*.log
pgrep -af 'run_demos850_continue|run_demos850_topup|stage[1-4]|extract_demos850'
```

Desired terminal state: `"state": "complete"`.  
Stop states: `stopped_shortfall`, `failed`, or a stuck `waiting` / mid-stage name.
