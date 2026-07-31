# Does “Initializing” mean RunAI keeps retrying the NFS mount?

**Date:** 2026-06-25

## Short answer

**Partially yes, but not forever.**

`Initializing` means the workload has **not** reached `Running`, `Completed`, or `Failed` yet. The platform **will retry** starting the pod/container after mount failures, but only up to a **backoff limit**. It is **not** a guarantee the mount will eventually succeed.

---

## Two levels of retry

### 1. Kubelet retries (same pod)

On `probe-nfs-0-0`, Events show many `Failed` mount errors between 16:51:59 and 16:53:19 — the **same pod** being retried with backoff (seconds apart). This is container **create** failing repeatedly before the pod is marked dead.

### 2. RunaiJob controller (new pods)

For `setup-vlm4`, the job created:

- `setup-vlm4-0-0` → **Error** (mount or start failure)
- `setup-vlm4-0-1` → **Succeeded** (same node `gpu-node005`)

So the **job** can spawn `probe-nfs-0-1`, `probe-nfs-0-2`, … up to the configured backoff limit (HANDOFF saw 7 attempts on an earlier `setup-vlm`).

When the limit is hit, phase becomes **Failed** with message like *“RunaiJob has reached the specified backoff limit”*.

---

## What `Initializing` means for `probe-nfs`

| Observation | Meaning |
|-------------|---------|
| Phase `Initializing` | Job still in retry / waiting loop |
| Only `probe-nfs-0-0`, `CreateContainerError` | First pod failed to mount; job may create `0-1` next or may be stuck |
| GPU Allocated 1.00 | Quota held while retrying |
| No logs | Container never started — nothing to stream |

**It does not mean:** “scheduler is queueing for a GPU.” GPU was already assigned.

**It might mean:** job controller will try another pod soon, **or** it is wedged between retries, **or** it will eventually flip to `Failed`.

---

## Will it retry until mount succeeds?

| Outcome | Possible? |
|---------|-----------|
| Next pod attempt succeeds (like `setup-vlm4-0-1`) | Yes — intermittent flake |
| Same node, same mount error every attempt | Yes — may exhaust backoff and **fail** |
| Retry on different node with working mount | Possible if scheduler reschedules |
| Infinite retries until success | **No** — backoff limit applies |

Mount failure is **not** something your bash command can fix by waiting; only a new pod attempt (possibly on another node) or admin fix on the node/mount agent helps.

---

## What you should do

1. **Do not wait indefinitely** on `probe-nfs` — NFS is already validated by `setup-vlm4`.
2. Poll: `runai training describe probe-nfs -p nlm-mh` — watch for new pods (`probe-nfs-0-1`) or phase → `Failed`.
3. If stuck **Initializing** for a long time while holding a GPU, ask Chun-Nam to delete the workload or free quota.
4. For real work: submit **`smoke-pope`** with a **new job name** when quota allows — treat `probe-nfs` as disposable.
