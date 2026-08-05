# Triple log: looks healthy; “VRAM allocated” is per-process

Date: 2026-07-31

Orchestrator: VERIFY_SYNC_OK, remap applied (incl. chair combined), HF prefetch,
three workers started (pids 1495 / 2183 / 3238) with LOAD_STAGGER.

The `VRAM allocated : 13.2 GB` / `15.4 GB` lines come from each worker’s own
PyTorch process (`torch.cuda.memory_allocated()` for that process only), not
device-wide occupancy. Concurrent models show up as interleaved log lines and
in `nvidia-smi` **Memory-Usage** (whole GPU), not in those per-process prints.
