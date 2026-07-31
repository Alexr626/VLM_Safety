# Qwen colocated with LLaVA on GPU 0

Alex clarified Stage B should share GPU 0 with Stage A (not wait for a second free A6000).

- Plan amended: parallel colocation on one A6000.
- Qwen Stage B launched (`run_qwen_windowed_steering_stage_b.sh`).
- At launch: GPU 0 ~30.6 GiB used / ~18 GiB free (LLaVA + Qwen residents).
