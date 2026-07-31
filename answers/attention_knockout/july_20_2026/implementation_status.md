# Attention knockout leading-clause plan — implementation status (2026-07-20)

## Done

- Filler templates + augment rebuild (`filler_b`)
- `src/prompt_spans.py`, `src/attention_knockout.py`, unit tests
- Stage 0 flip/stable sets
- LLaVA cells: `llava_block_all_downstream_reading`, `llava_block_last_token_reading` + analysis artifacts
- `IMPLEMENTATION.md` / `RESEARCH_LOG.md` updated

## Gates / stop points (plan)

1. **Qwen knockout not run** — flip union **11 < 15**. Enlarge item pool before Qwen cells.
2. **Full-depth sanity failed on LLaVA** — all-layers clause knockout agrees with neutral on **51.4%** of flip evaluations (threshold ~80%). Plan says stop for discussion before interpreting window curves.
3. **ε** — proposed **1.0546875** (10th pct \|Δlogit-margin\|); confirm with analyst before treating recovery plots as final.
4. Qwen first-token↔parsed agreement **98.0–98.5%** (below ~99%): 7 AMBER generation-parse mismatches + 3 POPE `p_yes==p_no` ties.

## LLaVA headline numbers (factual)

- Flip union 27; stable set 27; eager verify max blocked attention = 0
- Full-depth agreement 0.5135 (both scopes)
- Peak window recovery (all-downstream): layers_10–19 = 0.7568
- Filler vs neutral agreement ~0.57–0.63
- Filler-span knockout at layers_10–19 recovery 0.7222 (not near null)

Route interpretation / next design to the analyst chat.
