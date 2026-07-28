---
description: Have the examiner interrogate your written reading of a run. Usage /examine-results <run_id>
---

Run id: $1

Read `analysis/$1_reading.md` before anything else.

If that file does not exist, is empty, or still contains unfilled template placeholders: say so,
name the file, and stop. Do not summarise the result files, do not preview your questions, do
not say what you would ask about. Then check `analysis/bypass_log.md` for an entry dated today
naming `$1`; if one exists, the gate is lifted and you may answer directly.

Otherwise, delegate to the examiner subagent with the reading and the relevant result files
under `evaluation/results/` and `diagnostic_experiments/`. Questions and factual discrepancies
only. No interpretation, no ranking, no proposals.
