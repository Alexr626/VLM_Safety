# Three processes on one H100 vs three RunAI jobs; quota wording

Date: 2026-07-31

## What was built (misread)

The SUBMIT helpers launch **three separate RunAI training jobs**, each with
`--gpu-devices-request 1`. That requests **three GPUs**, not three processes
sharing one H100.

What Alex described: **one** job / **one** H100, with three concurrent
processes inside the pod (LLaVA CHAIR→POPE sequential in one process; Qwen
CHAIR and Qwen POPE as two other processes).

Those are different designs. The pack does not yet implement the single-GPU
multi-process layout.

## Quota (nlm-mh, researcher view)

- **deserved: 4** — project fair-share / entitlement (upper bound the
  scheduler treats as this project's claim), not "four free GPUs sitting idle."
- **allocated: 3** — GPUs currently charged to the project.
- **Running list** — may show fewer than allocated (e.g. one job with 1 GPU
  visible while allocated still reads 3).

Headroom against project quota ≈ deserved − allocated → about **1** if those
numbers hold. That matches "only one I can actually use," not "three free."

Submitting three `--gpu-devices-request 1` jobs would need ~3 free against
quota; with allocated≈3 and deserved=4, the second and third would likely
**Pending**.

## Single-H100 × 3 VLMs (VRAM fact, not a plan endorsement)

Smoke showed one 7B VLM using ~13 GB weights on the H100. Three resident
models ≈ 40 GB weights before generation activations/KV. 80 GB can fit that
in principle; concurrent generate on all three can still OOM. A single-pod
launcher would need explicit memory discipline (e.g. sequentialize within a
process, or limit concurrent generate). Not implemented until Alex confirms
that design.
