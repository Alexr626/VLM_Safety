# demos850 shuffled-control implementation launched

Date: 2026-07-29

Module + runner implemented. Derangements written (seed 1234, four blocks). LLaVA and Qwen extracting concurrently on GPU 0.

Logs:
- `logs/demos850_shuffled_control_llava_2026-07-29.log`
- `logs/demos850_shuffled_control_qwen25_2026-07-29.log`

Monitor:
```bash
pgrep -af extract_demos850_shuffled_control
nvidia-smi --query-gpu=index,memory.used --format=csv
tail -f logs/demos850_shuffled_control_{llava,qwen25}_2026-07-29.log
```
