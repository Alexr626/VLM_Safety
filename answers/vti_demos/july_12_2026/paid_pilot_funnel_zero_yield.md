# Paid pilot funnel (5 candidates) — 0 final demos

## Funnel

| Stage | Pass | Reject | Notes |
|-------|------|--------|-------|
| 0 | 5 | — | dry-run candidate pool |
| 1 | 4 | 1 | `000000002466` counting_check fail (plants not distinguishable) |
| 2 | 3 | 1 | `000000001145` structural: counting category `person` missing from caption |
| 3 | 1 | 2 | both rejects: existence LLM produced **2** edit regions (insert + replace) |
| 4 | 0 | 1 | `000000003915`: attribute “silver” fridge disputed; count “≥2 potted plants” unsure |
| 5 | 0 | — | nothing to assemble |

## Takeaways

1. Pipeline plumbing works with real APIs; yield on this tiny kitchen-heavy pool was 0/5.
2. Stage-3 existence edits are the main mechanical failure mode (model rewords beyond a pure insert).
3. Stage-4 correctly filtered a borderline attribute/count claim (plan: verifier is a filter, manual review is authority).

## Next

Mine a larger top-up (or remine 300) and run stages 1–4 again; optionally harden stage-3 existence to deterministic string insert before/alongside the LLM.
