# AMBER-100 LLaVA merged plots — clarifying questions

**Date:** 2026-07-22

## Proposed layout (as understood)

LLaVA-only merged PNGs (no Qwen). One figure per visual reasoning type (`attribute` or `relation`):

| | additive @ mlp | rotation @ mlp |
|---|---|---|
| **gold=yes** | neutral + assertive (2 panes) | neutral + assertive (2 panes) |
| **gold=no** | neutral + assertive (2 panes) | neutral + assertive (2 panes) |

That is **four panes on top** (yes × 2 methods × 2 conditions) and **four on bottom** (no × same), matching the spoken “four on top / four on bottom.”

## Open questions for Alex

1. Metrics: mean first-token prob + flips only (like POPE merged), or also accuracy?
2. Confirm nested conditions (neutral + assertive) inside each gold×method cell.
3. Mean-P figure: top uses P(yes), bottom P(no) on one PNG — OK?
4. Output dir: `…/plots/merged_plots/amber-100-runs/`?
5. Methods: additive @ mlp + rotation @ mlp only (omit rotation @ layer)?
