# LLaVA smoke OK; Qwen smoke + 3-GPU pack

Date: 2026-07-31

LLaVA smoke (`steer-llava-smoke`) reached `SMOKE_OK` (layout verify, H100,
baseline + steered POPE limit=5). Early `run_bash_lf.py` IndexError was a
missing-argv invoke; usage check added; verify no longer nests run_bash_lf.

Plan: 3 concurrent GPUs after Qwen smoke —
LLaVA CHAIR→POPE; Qwen CHAIR; Qwen POPE. Pack at
`/tmp/runai_steering_2026-07-31/` including `qwen_meandiff_directions.tar.gz`.
