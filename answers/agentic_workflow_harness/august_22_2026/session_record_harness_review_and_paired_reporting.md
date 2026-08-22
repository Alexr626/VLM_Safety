# Session record — harness review, template tables, and paired-comparison reporting

**Date:** 2026-08-22
**Question this answers:** what was established, changed, and left open during the first Claude Code
session on the personal-workstation device after the Nokia handover — covering harness state, the
reading/design/extraction templates, and the discovery that the repo's reporting convention does not
cover paired comparisons.

Facts and decisions only. No reading of any run appears here; the result files under
`evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/` were deliberately not opened,
so that `analysis/08_17_26/pope_0619_vs_0730_matched.md` remains Alex's first characterisation of
them. Only that directory's `README.md`, which describes the frame rather than the outcome, was read.

---

## 1. Context

Alex has left Nokia and continues this work personally, with his former manager as advisor. The work
is on public benchmarks only, so nothing about the confidentiality rule in `CLAUDE.md` changes.

The session opened on the procedural question: how to write up the matched LLaVA POPE comparison
built on 2026-08-06, given that its two arms come from a 2026-06-19 run that predates the agentic
workflow entirely and a 2026-07-30 run made under a different hypothesis, so no single design spec
covers the contrast.

---

## 2. Harness state on this device

`bash .claude/hooks/verify_harness.sh` → **passed=14 failed=0**. The pre-write gate is live.

Four stale or mis-scoped items, none of which break the gate:

1. **`/home/alex/CLAUDE.md` is a different project's instruction file.** It documents the VLM Safety /
   ShiftDC repo — different models, directories, and method — and is loaded into every session under
   `/home/alex`, headed "These instructions OVERRIDE any default behavior." The largest contamination
   risk in the current setup. Unresolved.
2. `WORKFLOW_MAP.md:25` still names the repo root as
   `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/`.
3. `.claude/settings.json` allow-list entries at lines 52, 56, 58 carry hardcoded `/home/romanus/...`
   and `/tmp/claude-1287/...` paths from the Nokia machine. Inert, but the allowlist was tuned for a
   path that no longer exists, so expect more permission prompts here than previously.
4. The `Write(analysis|designs|extractions)` denies remain listed as **unverified** in the
   `WORKFLOW_MAP.md` enforcement inventory. `verify_harness.sh` cannot test them — its
   `write under analysis/ exit=0` case exercises the hook, not the permission layer.

Permission facts that shaped the session: the main session may `Edit` but not `Write` under
`designs/`, `extractions/`, `analysis/`; `ABSTRACT.md` is denied for both `Write` and `Edit`, so it
is entirely Alex's to type.

**Relevant to the move to the MacBook:** `helper_scripts/export_agentic_workflow_harness.sh` copies
the whole harness into `exports/agentic_workflow_harness_<date>/`, with `--flat` and `--zip` modes.
Its static file list is maintained by hand. Items 1 and 3 above are machine-specific and will need
re-checking on any new device.

---

## 3. Procedural conclusions for the 06-19 vs 07-30 write-up

Established by reading `WORKFLOW.md`, `WORKFLOW_MAP.md`, and `.claude/commands/examine-results.md`:

- **The reading gate depends only on the reading file existing**, not on a design behind it
  (`examine-results.md:22-27`). The absence of a design does not block `/examine-results`.
- **A design spec should not be backfilled.** A prediction table written after the plots have been
  seen is the failure the harness exists to prevent. The template's front sections instead have an
  honest filling: cite `designs/07_30_26/steering_vector_visual_reasoning_validation.md` as covering
  the 07-30 arm only, copy its prediction table verbatim for the part it covers, and state plainly
  that the cross-run contrast had no standing prediction.
- **No bypass-log entry is required.** The bypass log covers taking an agent's answer in place of
  one's own work; writing a reading without a design does not do that.
- **Order:** reading → `/examine-results` → `ABSTRACT.md`. Editing the abstract first turns its
  sentence into the reading Alex then writes toward.
- The next question — why the author-derived direction moves POPE accuracy and the demos_850 one
  does not — is an **experiment**, so `designs/<date>/<name>.md`, not `extractions/`. Being CPU-only
  numpy over existing caches does not make a contrast an extraction (`WORKFLOW.md:90-96`). Any new
  directions it needs go under *Primitives this design requires* in that same spec; one spec, one plan.

### Factual correction to the stated framing

Alex described the contrast as 70 author demos versus ~500 demos_850 demos — a demo-count and
demo-source difference. `evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched/README.md:12-16`
records **three** things varying at once:

| Arm | Demos | nd | Direction recipe |
|---|---|---|---|
| 06-19 | author `demos.jsonl` | 70 | PC1 + mean |
| 07-30 | `demos_850` partition | 500 | raw mean-difference |

The same README records a third block on disk — `demos_850` at nd500 with PC1 + mean
(`*_r2_partition`) — marked "not an eval cell here."

### Open hole found

`analysis/08_05_26/steering_vector_visual_reasoning_validation.md` is an **unedited copy of the
template**, every angle-bracket placeholder intact. The 07-30 run has a design, a plan, and a
research-log entry, but no reading. `/examine-results` would refuse on it at `examine-results.md:24`.

---

## 4. Template changes made this session

Stamps bumped to `2026-08-17` in all three (note: the stamp date is wrong — see §7).

Convention introduced: tables are pre-drawn so numbers are typed rather than pipes; `...` marks an
unfilled cell; spare rows and columns are to be deleted. **The hook cannot see a leftover `...`** the
way it sees an angle-bracket placeholder, and each template now says so in its header.

An initial pass added ten tables and was cut back on Alex's instruction to only those holding numbers
not printed elsewhere. Final state: seven tables.

**`templates/design_template.md` — 4 tables.**
- Prediction table: baseline column and a third explanation column added, units in the header.
- Cells: original six columns retained; a `Cell` label column, which is the documented naming
  exception in `CLAUDE.md`.
- Method-validation sweep: a factor/levels/provenance table plus a single stated arm-and-run-count
  line, since the section's own prose requires both.
- Sample size: benchmark / n / baseline rate / smallest separable difference.
- Reverted to their original bullets: item sets, lineage, and *What I predict* — the last duplicated
  the prediction table directly above it.

**`templates/reading_template.md` — 2 tables.**
- One results table replacing four: `Cell / arm | n | Metric | Baseline | Value | Delta | Interval on
  the delta`, with a paired-versus-difference-of-aggregates declaration above it. **This table is now
  known to be wrong — see §6.**
- The design's prediction table pasted forward, plus a new explicit slot for arms that had no
  standing prediction.
- Cut: predicted-vs-observed, and the competing-explanations table, both of which re-quoted values
  printed above them.

**`templates/extraction_template.md` — 1 table.**
- Only the set-relations table under field 2, the field `WORKFLOW_MAP.md:85` calls the load-bearing
  one. No column for output paths, cache layout, or forward-pass counts, since *Not your job* assigns
  those to the plan.

Verified after editing: all tables column-consistent, no malformed rows; placeholder counts 27 / 20 /
12 for design / reading / extraction, all far above the hook's `MAX_PLACEHOLDERS = 2`, so an
untouched copy is still rejected. `verify_harness.sh` does not read `templates/`.

---

## 5. Tutor session — paired inference

Routed via `/tutor`. Agent id `a097e31c24e0a4fa0`, live at session end. It did not open the analysis
file or any results path; the 600 items and two-arm framing were carried as stated premises into a
synthetic analogue.

Output: `answers/concepts/august_17_2026/comparing_two_intervention_arms_on_the_same_items.md`
(misfiled date — see §7). Eleven sections. Load-bearing results:

- Two arms scored on the same items is a paired binary design, so McNemar-family is correct and a
  two-sample proportion test is wrong. **The pairing instinct was confirmed; the procedure was not.**
- **McNemar cannot be built from TP/FP/TN/FN.** Those are each arm's marginals. McNemar needs the
  joint — the 2×2 of arm A correct/wrong against arm B correct/wrong, joined on item id (§4.1). The
  marginals fix the accuracy difference exactly and its standard error not at all. §3.3 works a case
  where identical confusion matrices admit McNemar $p$ from $4\times10^{-8}$ to $0.21$. Free bound
  from marginals alone: the largest possible $z$ is $\sqrt{n|\Delta|}$.
- **Stratification is not a replacement for pairing.** Both axes are live: pairing across arms (shared
  items, covariance subtracts), stratification across gold labels (disjoint items, variances add).
  Conflating them was the error. §4 works both at once; §4.4 gives the interaction contrast.
- **Neither arm being a baseline is fine** — McNemar is symmetric (§5.1). What breaks is differencing
  two deltas each computed against a shared baseline: the point estimate is *exactly* right, since the
  baseline cancels, but the variances cannot be added because the two deltas share baseline noise
  (§5.2). §5.3 puts the naive route at $1.28\times$ too large in its worked case, growing to
  $3.08\times$ when the arms are nearly identical — conservative, but wrong, and worst exactly where
  two similar interventions are being separated. The direct A-vs-B table gives the right SE with no
  covariance bookkeeping.
- **Two standard errors, one variance formula** (§6.2): $\operatorname{Var}(\hat\Delta) = (\pi_d -
  \Delta^2)/n$, evaluated at $\Delta = 0$ for the score test (McNemar's denominator, $\text{SE}_0 =
  \sqrt m / n$) and at $\hat\Delta$ for the Wald interval ($\text{SE}_W = \frac1n\sqrt{m -
  (c-b)^2/n}$). $\text{SE}_W \le \text{SE}_0$ always. Do not write "the CI excludes zero, so
  $p<0.05$ by McNemar" — that uses one denominator to claim something about the other.
- **Minimum honest record of a paired comparison** (§6.3): $n$, $a$, $b$, $c$, $d$, $m = b+c$,
  $\hat\Delta$, SE and which one, CI, $p$ and which test. $m$ in particular, because the paired MDE
  is $(z_{\alpha/2}+z_\beta)\sqrt{\pi_d/n}$ and without $\hat\pi_d$ the interval cannot be compared
  against the pre-run "smallest difference separable from noise" commitment.
- **What the interval does not cover** (§6.4): randomness over items only. It says nothing about
  direction-estimation variance — two arms whose directions came from different demo draws differ
  partly for that reason, and the file states that resampling the demo set is the only route and is a
  different experiment. Nor about run-to-run variance, nor whether the size is worth asserting.
- A precondition Alex explicitly decided not to pursue: if the 600 items include several questions per
  image they are not independent and every SE is too small. **His call, recorded as declined.** The
  error runs toward calling a difference real rather than missing one; §8.1 quantifies it.

`learning/review_queue.md` was updated by the tutor: the stratified entry stays open with a
second-discussion note, and clustered items is a new entry with the source field left for Alex.

---

## 6. The reporting-convention defect

Alex identified that the reading template's results table asks for an interval on a single accuracy
metric while the interval must be computed from the paired contingency counts, which the table has
nowhere to put. **Confirmed against §6.3.** Three distinct defects in the table as written:

1. The counts instruction pointed at the wrong counts — accuracy's own numerator and denominator,
   which do not reconstruct the interval beside them.
2. The row was the wrong unit of observation. $b$ and $c$ are properties of a *pair* of arms; a row
   shaped as arm-vs-baseline hardcodes the routing-through-baseline framing §5.2 warns about.
3. $m$ is load-bearing for the column's own justification, and the table did not carry it.

Proposed replacement, **not yet applied**: split by unit of observation. Per-arm table keeps the
marginals (`Arm | n | Metric | Value`); a second table carries one row per comparison
(`Comparison (A vs B) | n | a | b | c | d | m | Δ̂ | SE (which) | 95% CI | p (which test)`).

### The general defect behind it

The harness has one concept — "the counts underneath a metric" — doing two jobs and defined for only
one. **Attribution** asks which input moved; its counts are the metric's own constituents.
**Inference** asks whether the result is separable from noise; its counts are whatever the reported
uncertainty was computed from. For an unpaired proportion these coincide, which is why it never
surfaced. For a paired comparison they do not overlap at all — $a, b, c, d$ appear nowhere in
accuracy's formula.

### Sites, scoped to reporting

1. `CLAUDE.md:169-170` — "report a metric together with the counts it is built from." The origin;
   everything else restates it. Generalisation: the metric's constituents *and* the counts its
   reported uncertainty was computed from. Keeping the two named separately preserves checkability,
   which broadening to "whatever counts matter" would lose.
2. `.claude/commands/examine-results.md:71-73` (check 6, Precision), mirrored at
   `.claude/agents/examiner.md:105-106` — already asks whether comparisons are paired and whether the
   pairing survived into storage. Two additions: an interval on a comparison is not an interval on a
   number, so "what interval sits on each number" mis-frames the paired case; and it should ask which
   SE and which test produced what is reported.
3. `templates/reading_template.md` — the table, per above.

Check 4 (Composite attribution) is correct for the job it names and should be left alone; it is simply
not the only counts requirement.

### A claim made and then withdrawn

It was asserted that `.claude/agents/planner.md:162-164` and `.claude/commands/plan.md:174` also
needed changing, on the grounds that "lineage in terms of measured quantities" would cause the paired
counts never to be written to disk. **This was wrong.** Per-item records are stored for every arm:
`responses.json` files carry `id`, `ground_truth`, and `response`, and
`evaluation/classifiers/metrics.py:38` derives correctness from those at scoring time rather than
requiring it to have been stored. 1905 such files exist under `evaluation/results/`. So $a, b, c, d$
for any pair of arms is a join on `id` between two existing files — no rerun, no plan change, nothing
lost. The design template's Lineage section and the planner need no change for this.

**Consequence worth carrying forward:** the counts needed for the current reading are already
computable from what is on disk.

---

## 7. Outstanding, for the next session

**Blocking the current write-up**
- `analysis/08_17_26/pope_0619_vs_0730_matched.md` is in progress and was copied from the pre-table
  version of the reading template.
- The §11 comprehension check in the tutor file is unanswered. Question 1: state in one sentence what
  the join on item id supplies that the two confusion matrices do not. The tutor considers the session
  unfinished until it is answered.
- Agresti 8.6(d), assigned 08/05, still outstanding — the interaction-across-independent-strata problem.

**Decisions pending, offered and not taken**
- Apply the §6 reading-template table change.
- Draft the `CLAUDE.md:169-170` and `examine-results.md` check-6 rewordings.

**Errors found, unfixed**
- `designs/07_30_26/steering_vector_visual_reasoning_validation.md:297` cites
  `templates/design_template.md:127-129`. This session's edits moved that text; it is now at
  **155-157**. General hazard: line-number citations into a template break on any template edit;
  section-name citations do not.
- `templates/design_template.md:193` states that the binomial SE at the baseline rate is enough to
  state resolution. That is the unpaired formula. §6.3 puts the paired MDE at $n = 600$ anywhere from
  2.56 to 7.24 points depending on discordance — a factor of nearly three at fixed $n$, driven by the
  intervention pair rather than the benchmark. A design-time error, distinct from the reporting one.
- **Date slips.** Today is 2026-08-22. The tutor wrote to `answers/concepts/august_17_2026/` and the
  three template stamps were set to `2026-08-17`; all four should read `august_22_2026` / `2026-08-22`.
  `analysis/08_17_26/` is Alex's own and may be deliberate.
- Doc drift from the baseline column added to the prediction table: `WORKFLOW.md:111` still describes
  it as "one row per condition, one column per competing explanation," and
  `.claude/commands/examine-design.md:47-50` tells the examiner to read each column against the
  explanation it belongs to, which a baseline column has none of. The practice predates this session —
  `designs/07_30_26/...` already used a baseline column with a note explaining it.

**Harness items, machine-specific**
- `/home/alex/CLAUDE.md` contamination (§2, item 1).
- Stale `/home/romanus` paths in `settings.json` and `WORKFLOW_MAP.md:25`.
