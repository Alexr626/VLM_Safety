# RunAI nlm-mh GPU allocation status (2026-07-15)

**Verdict:** Yes — `nlm-mh` project GPU quota is fully allocated (100%).

## Evidence from `runai` CLI

`runai project list`:

| Project | GPU Quota | Allocated GPUs | Allocation Ratio |
|---------|-----------|----------------|------------------|
| nlm-mh  | 4.00      | 4.00           | 100.00%          |

Running workloads (`runai workload list -p nlm-mh --status Running`):

| Workload     | Type      | GPU Alloc. |
|--------------|-----------|------------|
| hal44        | Training  | 2.00       |
| llama3-70b8  | Inference | 2.00       |

Sum = 4.00 = full project quota. No Pending jobs.

## Implication

A new 1-GPU H100 training submit for this project will not start until at least one of those running jobs finishes or is stopped (quota free), regardless of whether the H100 pool itself has spare nodes.
