# CHAIR caption-length check — what “failure” means (and what it is not)

Date: 2026-07-30  
Source: `implementation_plans/7-30-26/steering_vector_visual_reasoning_validation_plan_2026-07-30.md` (Sanity checks; Open questions §1)  
Context: Clarification after the implementer mentioned a “CHAIR check failure” contingency before any check had been run.

## Short version

Nothing has failed. There is no failed CHAIR run on disk from this session. The implementer was talking about a **hypothetical** outcome of the **pre-grid sanity check** (20 images × caps 256 and 512), not about skipping CHAIR in the evaluation order.

Alex’s required execution order stays: **AMBER → CHAIR → POPE**. CHAIR is supposed to run immediately after AMBER. That was never in dispute in the plan’s sequence table.

## What the check is

Before either overnight grid process starts, the plan runs `step0_chair_token_cap.py` once per model on the fixed 20-image draw (`--num_images 20 --seed 1234`) at `--caps 256 512`.

It asks: under the frozen CHAIR prompt, do captions **naturally stop before 256 new tokens**, or do they hit the length cap?

**Pass (plan wording):** zero captions hit the cap at 256 on both models. Then 256 and 512 should produce identical text, so CHAIR cannot move just because of the cap.

**What the plan calls invalidating / “failure”:** any caption that hits the 256-token cap on either model. Then the fixed-256 design for CHAIR rows is not safe, and the plan says to report counts and how much `chair_s` / `chair_i` / `avg_objects_mentioned` differ between 256 and 512, and stop the CHAIR design as written.

This check is separate from the grid. It is not “CHAIR failed after AMBER.” It is a ~20-minute probe that must finish **before** the two model drivers launch.

## What the open question was

The plan’s Open question 1 is only about policy **if that probe fails**:

- continue AMBER and POPE while the cap question is resolved (CHAIR phase blocked), or  
- halt the whole overnight launch.

The plan text also says a probe failure “blocks only the CHAIR phase; the AMBER and POPE phases are untouched.” That is **not** the same as changing the happy-path order. Happy path is still AMBER, then CHAIR, then POPE.

## What the implementer wrongly implied

In the previous turn, the implementer sketched an overnight contingency that, on probe failure, would launch with `BENCHMARKS="amber pope"` (skip CHAIR). That was an unauthorized policy choice on an unresolved open question, and it conflicts with the user’s stated requirement that CHAIR run right after AMBER in the main sequence.

Correct behavior going forward:

1. Happy path (probe passes): launch both models with full `amber chair pope` order — AMBER block, then CHAIR block, then POPE block.  
2. Probe failure: **do not silently drop CHAIR**. Stop and surface the probe numbers; Alex decides halt vs AMBER+POPE-only.

## Relation to execution order

| Stage | When | Role of CHAIR |
|---|---|---|
| Mean-difference extraction | CPU, before GPU | none |
| Caption-length probe | GPU, before grid | only the 20-image 256/512 check |
| Grid (if probe passes) | overnight | full CHAIR-500 cells **right after** that model’s AMBER cells, before POPE |

So: want CHAIR right after AMBER → that is already the plan when the probe passes. The only undecided case is what to do if the probe says 256 truncates.
