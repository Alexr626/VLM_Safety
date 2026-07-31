# Why `probe-nfs` failed and whether it will block future runs

**Date:** 2026-06-25

## TL;DR

- **Not** a GPU quota problem (1 GPU was allocated).
- **Not** your bash command, tarball, or NFS data — the container **never started**.
- Failure is at **NFS volume mount** inside Kubernetes: `mount callback failed ... no users found`.
- **Same submit pattern worked** for `setup-vlm4`, `probe-mm`, `probe2` — so this is **intermittent / node-level infra**, not a permanent misconfiguration on your side.
- **May or may not recur** on the next submit; if it does repeatedly on the same node, escalate to Chun-Nam / cluster admin.

---

## What actually failed

Timeline from `runai training describe probe-nfs`:

1. Scheduler **assigned** pod to `gpu-node004` on `h100-pool` ✓
2. GPU **allocated** (1.00) ✓
3. Image **pulled** ✓
4. **CreateContainerError** when kubelet tries to mount NFS into the container ✗

Your command (`ls`, `micromamba --version`) **never ran**.

The error:

```
failed to create containerd container: mount callback failed on .../containerd-mount...: no users found
```

RunAI mounts NFS with:

```
--nfs path=/volume1/airl-datalake,server=gpustorage-1.cloud.bell-labs.com,mountpath=/home/datalake,readwrite
```

At mount time the platform must map **your RunAI/K8s identity** to a user the NFS server accepts. **"no users found"** means that mapping step failed on the **node** — before any of your files are touched.

---

## Relation to HANDOFF.md errors

HANDOFF documents the **same error string** when:

| Misconfig | Result |
|-----------|--------|
| `--gpu-devices-request 0` | `no users found` mount error |
| `--node-pools l40s-pool` (with this image + NFS) | same |

Your `probe-nfs` used **correct** flags: `1` GPU, `h100-pool`. So this is **not** the classic misconfiguration case — it is the same **error class** at the infrastructure layer, triggered differently (likely node/mount-agent glitch).

---

## Evidence your setup still works

| Job | Node | Outcome |
|-----|------|---------|
| `setup-vlm4` | `gpu-node005` | Pod `0-0` **Error**, pod `0-1` **Succeeded** → full setup |
| `probe-mm` | (completed) | micromamba + NFS listing OK |
| `probe-nfs` | `gpu-node004` | Stuck on mount errors |

`setup-vlm4` even **failed once on the same node** then succeeded on retry — mount/NFS bring-up can be flaky.

NFS **content** is fine (setup wrote env, weights, data). This failure is **pod bootstrap**, not corrupt storage.

---

## Will it persist on future runs?

| Scenario | Likelihood | What to do |
|----------|------------|------------|
| **Transient** — next job lands on another node or retries | Common | Submit with **new job name**; often works |
| **Node-specific** — `gpu-node004` mount agent broken | Possible | Retries on that node keep failing; tell admin |
| **Quota stuck** — failed job holds 1 GPU while retrying | Happening now | Delete/stop `probe-nfs` to free slot |
| **Your config broken** | **Unlikely** | `setup-vlm4` proved otherwise |

**For real experiments:** use the same NFS + image + `h100-pool` + 1 GPU template that worked for `setup-vlm4`. Expect occasional mount flakes; retry is normal.

**You do not need `probe-nfs`** — NFS was already validated. Proceed to `smoke-pope` when quota allows.

---

## What to do now

1. **Stop caring about `probe-nfs` logs** — it may never produce stdout if the container cannot mount.
2. **Free GPU quota** (optional): ask admin or use RunAI delete/stop if available for stuck `probe-nfs`.
3. **Check quota:** `runai project list` (was 2/2 GPUs allocated).
4. **Submit smoke test** with new name when a GPU is free — same flags as successful jobs.
5. **If 2–3 consecutive jobs fail with same mount error on describe** → send Chun-Nam the `describe` Events block (node name + `no users found`).

---

## How to tell mount failure vs your code failing

| | Mount failure (`probe-nfs`) | Your script failing |
|--|----------------------------|---------------------|
| Pod phase | `CreateContainerError` | Often `Error` after `Started` |
| Logs | `pod is not ready` / empty | Your `echo` / Python tracebacks |
| Events | `mount callback failed` | No mount errors; app errors in logs |
| NFS data | Untouched by this job | Results may be partial on NFS |
