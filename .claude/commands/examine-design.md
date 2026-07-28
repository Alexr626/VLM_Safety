---
description: Have the examiner interrogate an experiment design before it is planned. Usage /examine-design <exp_id>
---

Experiment id: $1

Read `designs/$1_design.md` before anything else. If it does not exist or is an unfilled
template, say so and stop.

Delegate to the examiner subagent. Focus on the prediction table: whether it is complete, and
whether any two explanation columns are identical across every populated row. Where they are,
ask which cell separates them — do not name the cell yourself. Also check that the primary
measurement is a primitive, that every item set is declared, and that the stated n can separate
the predicted effect from zero.
