# How to check output of the `probe-nfs` RunAI workload

**Date:** 2026-06-25

## Quick answer

From **lambdab2**:

```bash
runai project set nlm-mh
runai workload list -p nlm-mh
runai training logs probe-nfs -p nlm-mh
```

Logs are the workload **output** for a short probe job like this — there is no separate results file on NFS for `probe-nfs`.

---

## Step-by-step

### 1. Confirm you are logged in

```bash
runai whoami
```

If expired: `runai login`

### 2. Check job status

```bash
runai workload list -p nlm-mh
```

Find `probe-nfs` in the table:

| Status | What it means |
|--------|----------------|
| **Initializing** | Waiting for GPU / pulling image / starting pod — **logs not available yet** |
| **Running** | Pod is up; logs may stream |
| **Completed** | Finished successfully — full logs available |
| **Failed** | Pod errored — check logs and `describe` |

### 3. Stream or fetch logs (main output)

```bash
runai training logs probe-nfs -p nlm-mh
```

Tail only the end:

```bash
runai training logs probe-nfs -p nlm-mh 2>&1 | tail -30
```

**Success looks like:** directory listing under `/home/datalake/romanus/`, `micromamba --version` (e.g. `2.8.1`), no tracebacks.

### 4. If logs say “pod is not ready”

```
Error: failed to print logs. workload is not ready to stream: pod is not ready
```

The job is still **Initializing** or the container has not started. Wait 1–5+ minutes (longer if the cluster is busy), re-check status, then retry logs:

```bash
runai workload list -p nlm-mh | grep probe-nfs
runai training logs probe-nfs -p nlm-mh
```

### 5. More detail (optional)

```bash
runai training describe probe-nfs -p nlm-mh
```

Useful fields:
- **Phase** — `Completed` vs `Failed`
- **Command** — what actually ran in the pod
- **Pods** table — per-attempt phase (`Error` vs completed)

### 6. Follow logs while running (optional)

Some CLI versions support following live output; if not, poll:

```bash
watch -n 30 'runai workload list -p nlm-mh | grep probe-nfs'
```

Then run `runai training logs probe-nfs -p nlm-mh` once status is **Completed**.

---

## What `probe-nfs` does not produce

- No `metric_summary.json` on NFS
- No files written to `/airl-datalake/romanus/` by the probe itself

It only **reads** NFS and prints to stdout → captured in RunAI logs.

---

## As of 2026-06-25

`probe-nfs` was still **Initializing** (GPU contention). Retry `runai training logs probe-nfs -p nlm-mh` after status shows **Completed**.
