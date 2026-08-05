# Qualitative response HTML — answered sampling / presentation choices

## Settled

- **N = 50** per HTML (independent draw per file).
- **Stratify by gold only when the source plot is a gold-yes / gold-no accuracy view**; for overall-accuracy plots, uniform random over the items counted in that plot.
- **No special “disagreement” filter** — random from the plot’s counted examples is enough.
- **Presentation:** image left; question + gold; labeled responses per config with parsed yes/no, correct/incorrect vs gold, and full response text.
- **One HTML file per plot path.**

## Still open (for Alex)

1. Qwen early plot was listed twice — keep one, or replace the duplicate with another path (e.g. late / all)?
2. Confirm config set = **every series/point that plot shows** (e.g. full nd×β grid for `accuracy_vs_beta_by_steering_vector_sample_size`, full window×β for nd500 heat-style plots, baseline+3 windows for gold-label nd500 β0.9).
3. Output dir default unless overridden: `{llava,qwen}_amber_results/qualitative/<plot_stem>.html`.
