# July 2, 2026 VTI smoke test — what was run?

## Short answers

- **Sample sizes:** AMBER-25 + CHAIR-5 (pinned from the 2026-06-22 LLaVA diagnostic sample draw).
- **July 2 results were from BOTH textual and visual steering**, not one or the other:
  - **Stage 0:** textual arm (gate positive control)
  - **Stages 1 + 1b:** visual arm (main smoke objective)
- **Can the driver run textual-only or visual-only?** Yes, by which `--stage` you invoke (or by editing the cell lists). The harness is intervention-agnostic; stages are split by design.
- **Stage 2** discrepancy toggles were **not** run on July 2.

Full regenerated report: `evaluation/results/2026-07-02/_diagnostics/vti_visual_smoke_summary.md`
