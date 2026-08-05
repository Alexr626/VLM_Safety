# Triple grid progress + ETA (log SCP ~2026-07-31 12:40 ET)

Job start: 15:12 UTC (11:12 ET). Elapsed at review: ~1.5h.

## Position
- LLaVA: still on CHAIR, layers=all, around nd=500 (POPE not started) — ~6/37 chair
- Qwen CHAIR: layers=all, nd=500, β≈0.9 in flight — ~6–7/37
- Qwen POPE: layers=all, nd=50, β≈0.9 mid-splits — ~3–4/37

## ETA (contended H100 rates)
Wall ≈ max(workers) ≈ **27–31 hours** remaining.
Bottlenecks: Qwen CHAIR (~52 min/cell) and LLaVA’s CHAIR→POPE chain.
