# Fix for failed `runai training logs`

## Typo
You ran:
```bash
runai training logs pd-s2-stage-p nlm-mh --follow
```
That treats the workload name as `pd-s2-stage-p`. Use a **space** before `-p`:

```bash
runai training logs pd-s2-stage -p nlm-mh --follow
```

## Timeout
`context deadline exceeded` on `https://runai.cloud.bell-labs.com/cluster-api/status` is a control-plane hiccup. Retry the corrected command; `workload list` already shows `pd-s2-stage` Running, so the job itself is fine.

Optional: bump timeout, then retry:
```bash
runai config set --status-timeout-duration 2m
runai training logs pd-s2-stage -p nlm-mh --follow
```

## Progress without logs
NFS artifacts appearing under `/airl-datalake/romanus/vlm_hallucination/data/amber/images/` (WinSCP) also indicate S2 is advancing.
