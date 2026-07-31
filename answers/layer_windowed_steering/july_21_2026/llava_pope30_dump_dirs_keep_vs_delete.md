# LLaVA POPE-30 dump dirs: keep vs delete

Location: `data/pope/dumps/llava-1.5-7b-hf/`

## Keep (new unique-pin yes/no runs)

| Directory | What it is |
|---|---|
| `pope30_yes_windowed_steering` | POPE-30-yes (rebuilt pin + in-grid baseline), 64 cells |
| `pope30_no_windowed_steering` | POPE-30-no (control), 64 cells |

## Safe to delete for POPE-30 cleanup (legacy all-gold-yes / triplicated pin)

| Directory | What it is |
|---|---|
| `pope30_windowed_steering` | Older windowed grid on triplicated POPE-30 pin (~63 cells, no in-grid baseline) |
| `pope30_existence_yes_baseline` | Separate baseline for that older yes-30 dump |

## Unrelated to POPE-30 (leave unless you intend a broader cleanup)

| Directory | What it is |
|---|---|
| `pope_yes_120_baseline` | 120-item yes baseline |
| `pope_no_120_baseline` | 120-item no baseline |
