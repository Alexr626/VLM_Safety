---
description: Expand an approved design spec into a repo-grounded implementation plan. Usage /plan <exp_id>
---

Spec id: $1

Read `designs/$1_design.md` or `extractions/$1_extraction.md`, whichever exists. If neither
exists, say so and stop — do not draft one. If both exist, ask which this plan is for.

Delegate to the planner subagent. The spec is a whitelist: no cell, condition, metric,
model, item set, or seed enters the plan unless it is in the spec. If the design is inadequate,
return questions and write nothing.

The plan file must carry exactly one of `design_spec: designs/$1_design.md` or
`extraction_spec: extractions/$1_extraction.md` on its second line. A pre-write hook enforces
this, including that only one is present.
