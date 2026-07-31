# demos_v2 qual grid — run status (2026-07-14 morning)

**Checked:** 2026-07-14 ~08:38 ET

## Summary

| Model | Status | Cells (chair / amber) | Galleries |
|-------|--------|------------------------|-----------|
| LLaVA-1.5-7B | **finished** | 241 / 241 (full grid) | all 5 dims × 2 benches |
| Qwen2.5-VL-7B | **in progress** (~last slice) | 205 / 203 of 241 | all, existence, attribute, counting (no relation yet) |

Qwen process still alive on GPU 0 (`relation` / `nd=50` / `β=0.9`). No OOM / fatal markers in either log.

## Smoke gate (both passed)

- Hash `9a44f4afde0324b5` present in config
- Result dir carries `__dall__nd500`
- Additive MLP β=0.5 response **differs** from baseline (LLaVA 428→448 chars; Qwen 592→568)

## Direction caches

Both models: 20 slugs (5 dims × 4 nd) under `experiment_artifacts/vti/{model}/textual_v2/`.

**PC1 explained-variance (nd=500):**

| dim | LLaVA | Qwen |
|-----|-------|------|
| all | 0.195 | 0.646 |
| existence | 0.226 | 0.492 |
| attribute | 0.156 | 0.783 |
| counting | 0.232 | 0.680 |
| relation | 0.220 | 0.563 |

## Morning deliverable paths (ready now)

`evaluation/results/2026-07-13/_samples/{model}/{bench}/*_demosv2_all_review.html` (and other finished dims).

Logs: `logs/demosv2_qual_{llava,qwen}_2026-07-13.log`
