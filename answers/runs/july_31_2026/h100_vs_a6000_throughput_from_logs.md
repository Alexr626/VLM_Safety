# H100 (triple, contended) vs A6000 (lambdab2) throughput

Date: 2026-07-31

## Apples-to-apples: LLaVA CHAIR, 500 samples, chair_max_new_tokens=512

| Machine | Sharing | Rate (samples/s) | ~min / 500-cell |
|---------|---------|------------------|-----------------|
| lambdab2 A6000 | solo on GPU 0 | **~0.19–0.22** (log steady ~0.20) | **~42–43** |
| RunAI H100 | 3-way share (LLaVA+2×Qwen) | **~0.31–0.37** | **~23–27** |

Per-job LLaVA CHAIR on contended H100 is still ~**1.6–1.9×** the A6000 solo rate.

## Other rates (not directly comparable across benches)

- H100 Qwen CHAIR (contended): ~**0.15–0.17** samp/s
- H100 Qwen POPE N=200 (contended): ~**0.37–0.41** samp/s
- A6000 Qwen AMBER N=450 (solo GPU, recent): ~**13.5 min/cell** ≈ **0.56** samp/s (short answers — not CHAIR)

## Caveat

H100 numbers are under **full triple utilization**. Solo H100 would be faster per job;
this log does not isolate a long solo-H100 stretch after all three are loaded.
