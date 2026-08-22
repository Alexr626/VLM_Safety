<!-- Last updated: 2026-07-30 — harness template; delete this line in your copy -->

# Reading — evaluation/results/2026-08-06/_analysis_pope_0619_vs_0730_matched

Written by Alex, after the numbers are in and before the examiner sees anything. Copy to
`analysis/<run_id>.md`, or `analysis/<date>/<run_id>.md`.

Write this from the result files directly. Do not ask an agent what the files show before you
have written this; a factual summary from an agent arrives with an implied reading attached.

---

## Design spec this run came from

None - this comes from a comparison between one run, 06/19, that occurred before the harnesses that define the research agentic workflow of this repository existed, and another, 07/30, that was not meant to compare the efficacy of the steering vectors that were used in each run, respectively.

## What I predicted

As there was no design spec written for the comparison of these runs, I made no prediction in how their results would compare.

## What the numbers are

Facts only. Cells, values, intervals. No reading yet.

One row per cell. Units in the header. Two column rules rather than extra tables:

- **If the metric is a composite, add numerator and denominator columns here.** `CLAUDE.md`
  requires the counts alongside the ratio, because a ratio can hold steady while both of its
  inputs move, and once the denominator is discarded the attribution cannot be recovered by
  reanalysis.
- **Delete the interval columns only if nothing in the pipeline produced an interval** — and
  then say so under Precision rather than leaving it silent.

Deltas here are paired per item / a difference of two aggregates: <answer>

| Cell / arm | n | Metric | Baseline | Value | Delta | Interval on the delta |
|Beta = 0.2, 06-19 run | 600 | accuracy | 84.67 | 83.67 | -1.00 |---|
|Beta = 0.5, 06-19 run | 600  | accuracy | 84.67 | 85.50 | 0.83 |---|
|Beta = 0.9, 06-19 run | 600  | accuracy | 84.67 | 88.00 | 3.33 |---|

| Cell / arm | n | Metric | Baseline | Value | Delta | Interval on the delta |
|Beta = 0.2, 07-30 run | 600-averaged  | accuracy | 84.67 | 84.67 | 0 |---|
|Beta = 0.5, 07-30 run | 600-averaged  | accuracy | 84.67 | 83.33 | -1.33 |---|
|Beta = 0.9, 07-30 run | 600-averaged  | accuracy | 84.67 | 83.17 | -1.50 |---|


Files these came out of, so the numbers are traceable without rerunning anything:

answers/runs/august_6_2026/pope_0619_vs_0730_experimental_config.md


## Where my prediction was wrong

No prediction was made due to the lack of a design spec. 

## My reading



<reading>

## Competing explanations I considered and why I rejected them

At minimum, the alternatives from the design spec's prediction table. For each, the specific
cell or number that rules it out.

- <explanation — what rules it out>

## How much of the original question this answers

<answer — be specific about the part it does not answer>

## What I am not able to conclude from this

<answer>

## Open questions this raises

Questions, not experiments. Deciding what to run next comes after the examination, not before.

- <question>

## Evidence tier

One of: qualitative on a handful of items / aggregate over many items, single seed /
controlled, multi-seed and multi-model with baselines.

<tier>

## Confidence

How much you would bet on your reading surviving a doubled n and a second seed, and why.

<answer>
