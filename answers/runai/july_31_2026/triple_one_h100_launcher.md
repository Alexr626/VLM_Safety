# Single-job triple launcher on one H100

Date: 2026-07-31

Replaced the three separate `--gpu-devices-request 1` submits with
`run_steering_triple_one_h100.sh`: one RunAI job, one GPU, three concurrent
processes (LLaVA CHAIR→POPE; Qwen CHAIR; Qwen POPE). Load stagger default 45s.
Markers: `TRIPLE_OK` / `TRIPLE_FAIL`. Pack: `/tmp/runai_steering_2026-07-31/`.
