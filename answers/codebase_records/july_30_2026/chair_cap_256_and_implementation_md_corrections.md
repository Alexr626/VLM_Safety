# CHAIR cap → 256, and corrections owed to `IMPLEMENTATION.md`

Written 2026-07-30. Handoff for Cursor. Every edit below was attempted and **refused by the
permission settings** — `Edit(evaluation/**)`, `Write(evaluation/**)`, `Edit(IMPLEMENTATION.md)`
and `Write(IMPLEMENTATION.md)` are all in the deny list in `.claude/settings.json`. Nothing in
this file has been applied. It is a specification of edits, not a record of them.

Two of the changes originate in Alex's instruction of 2026-07-30; the rest are facts the planner
verified while expanding `designs/07_30_26/steering_vector_visual_reasoning_validation.md`.

---

## 1. CHAIR generation cap is 256 from now on

Alex's decision, 2026-07-30: **CHAIR runs are always done at `max_new_tokens = 256`.** The
reason is one he already held — 64 truncated captions before either model finished, which biases
`chair_s` and `chair_i` in both directions. The 64 default was correct when the Step 0 probe
froze it and is no longer. Any script that still says otherwise gets updated.

The result that motivated this is not an `IMPLEMENTATION.md` entry; the frozen default is,
because it is a code fact.

### Code edits

| File | Line | Now | Change to |
|---|---|---|---|
| `evaluation/run_eval.py` | 50 | `p.add_argument("--chair_max_new_tokens", type=int, default=64,` | `default=256` |
| `evaluation/run_eval.py` | 51 | help text `"... (default 64). "` | `"... (default 256). "`, and append: `"Raised from 64 to 256 on 2026-07-30: 64 truncated captions before the models finished, biasing chair_s/chair_i."` |
| `evaluation/chair_amber_diagnostics/run_scripts/run_exp1_repro_grid.sh` | 41 | `CHAIR_CAP="${CHAIR_CAP:-64}"` | `CHAIR_CAP="${CHAIR_CAP:-256}"` |
| `evaluation/run_scripts/run_demosv2_qual_grid.sh` | 39 | `CHAIR_CAP="${CHAIR_CAP:-512}"` | `CHAIR_CAP="${CHAIR_CAP:-256}"` |

Already consistent with 256, no edit needed — recording them so they are not "fixed" in the
wrong direction:

- `evaluation/runners/eval_runner.py:189` — `chair_max_new_tokens: int = 256` already.
- `evaluation/chair_amber_diagnostics/make_diagnostic_summary.py:37` — `CHAIR_CAP = 256` already.
  This constant was previously **stale in the other direction**: it read 256 while every run it
  summarised had been generated at 64 and the CLI default was 64. After the edits above it is
  correct rather than accidentally right.
- `diagnostic_experiments/vti_visual_smoke/run_smoke.py:38` — `CHAIR_CAP = 256` already.

### `IMPLEMENTATION.md` edits

| Line | Now | Change to |
|---|---|---|
| 1269 | `--chair_max_new_tokens 64` in the canonical baseline invocation block | `--chair_max_new_tokens 256` |
| 1283 | ``- `--chair_max_new_tokens` (default **64**) freezes the CHAIR caption-generation`` | `(default **256**)`, and add a sentence: "Raised 64 → 256 on 2026-07-30 (`run_eval.py:50`); 64 truncated captions before generation finished. 256 is the standing default for every CHAIR run." |
| 1513 | "sharing one pinned subset draw + the frozen CHAIR cap (64) + the verbatim VTI prompt" | "(64 at the time; the standing default is 256 from 2026-07-30)" |
| 1515 | step0 bullet, "Result frozen the cap at **64** (see RESEARCH_LOG 2026-06-22)." | keep the historical sentence, append: "**Superseded 2026-07-30** — the standing cap is 256." |
| 1517 | Exp1 repro grid description, `--chair_max_new_tokens 64` | `--chair_max_new_tokens 256` (matching the script default after the edit above) |
| 577 | Qual-grid frozen facts, ``--chair_max_new_tokens 512`` "(deliberate vs 6/22 cap 64)" | leave the recorded run facts alone; append "(the script default is now 256)" |

Leave alone — these are records of what a past run actually used, and rewriting them would
falsify the record: line 892 (`max_new_tokens=64` for the pinned CHAIR-5 subset), and every
`RESEARCH_LOG.md` entry naming 64 or 512.

`readme.md` lines 247, 259 and 414 also document the default as 64 and should follow.

---

## 2. Facts the planner verified that `IMPLEMENTATION.md` does not carry

All three are code/disk facts, so they belong in `IMPLEMENTATION.md`. Cursor should re-verify
each before writing it, since none was established by Cursor.

**`data/pope/pinned_eval_ids.json` is not consumable by `--subset_ids_file`.** It has no
top-level `pope` key and is not a flat id list, and `load_pope` has no `subset_ids` parameter.
`--limit 200` per split was verified to reproduce the three pinned id lists byte-for-byte. The
file is referenced at line 1808 as the ordering source for the POPE-30-yes pin; the pins section
should say plainly how it is and is not consumable, so nobody reaches for `--subset_ids_file`
with it again.

**The demos_850 activation cache is complete for both models.** All 850 ids × `{value, all}` ×
both models are present under `experiment_artifacts/vti/{llava-1.5-7b-hf,qwen2.5-vl-7b-instruct}/textual_v2/_act_cache/`.
Consequence worth stating where the cache is described: extracting a mean-difference direction
from this cache is **zero forward passes**, CPU-only, and needs no model load.

**Layer counts, read from the cached stacks:** LLaVA-1.5-7B 33 rows × 4096; Qwen2.5-VL-7B-Instruct
29 rows × 3584.

---

## 3. Deliberately not written to `IMPLEMENTATION.md`

The Qwen2.5-VL AMBER-450 baseline figures the planner measured (`n_unparsed` 84/450, accuracy
0.642, neg/pos item accuracy 0.826/0.351) are a **run result**, not a code fact. Under the
`CLAUDE.md` split added 2026-07-30 they belong in `RESEARCH_LOG.md` if anywhere.

Alex has stated the cause: that run hit out-of-memory errors, which produced the unanswered
items, and `--max_pixels 1003520` being enforced going forward prevents recurrence. He has
instructed that no check be run to investigate further. No such check is planned, and the
sanity check that would have re-measured it has been removed from the implementation plan.
