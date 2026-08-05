# Why “1 GPU free” was said — deserved vs allocated

Date: 2026-07-31

RunAI project JSON for `nlm-mh` exposed two different numbers:

- **deserved** (under `resources`): project fair-share / guaranteed quota
  (observed as 4).
- **allocated** (under `status.nodePoolQuotaStatuses`): GPUs currently counted
  against the project (observed as 3 at the earlier check).

“~1 free” was **deserved − allocated** (4 − 3), not “only 1 GPU exists.”

That is **not** the same as counting `runai workload list --status Running`
rows: allocated can include jobs from other users in the project, or allocations
not obvious from a short list, so the Running column alone understates use.

Pending after submit means the scheduler did not place the pod yet — causes
include project allocation at cap, node-pool contention, or other users’ jobs,
not necessarily “only one GPU in the whole cluster.”
