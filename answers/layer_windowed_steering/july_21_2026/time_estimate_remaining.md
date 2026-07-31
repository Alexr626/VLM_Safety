# Time estimate — windowed steering remaining (2026-07-21 ~11:05)

Snapshot from LLaVA log under GPU-0 colocation with Qwen.

## LLaVA (rate used for projection)

- Done: **23 / 63** POPE-30 cells (~mid cell 24)
- Recent colocated pace: **~68–74 s per POPE cell** (~2.3 s/item; AMBER cell ≈ ×3.33 → ~3.8 min)

| Remaining | Estimate |
|-----------|----------|
| POPE-30 (40 cells) | **~45 min** |
| AMBER-100 (63 cells) | **~4.0 h** |
| **LLaVA total remaining** | **~4.75 h** (ETA ~15:50 local if rate holds) |

## Context (not the asked projection base)

Qwen is ~10 s/item under share → ~20 h remaining alone; wall-clock to finish **both** on GPU 0 ≈ Qwen (~20 h), not LLaVA.
