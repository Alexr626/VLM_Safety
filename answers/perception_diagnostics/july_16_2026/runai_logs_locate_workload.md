# `could not locate desired workload` on logs (pd-s2-stage still Running)

## Still a typo in one attempt
`pd-s2-stage-p` is wrong. Correct name: `pd-s2-stage`.

## Why logs fail while list works
RunAI’s **list** API can see the job; **log streaming** often needs a fresher OIDC token / cluster connection. Your kubectl also shows `No valid id-token` — same root cause.

## Fix (in order)

```bash
runai login
runai project set nlm-mh

# get current pod name
runai training standard describe pd-s2-stage -p nlm-mh | grep -E 'Name|Phase|Pod'

# try logs with explicit pod (from describe, e.g. pd-s2-stage-97cwd)
runai training standard logs pd-s2-stage -p nlm-mh --pod=pd-s2-stage-97cwd --follow

# or without --pod after login
runai training standard logs pd-s2-stage -p nlm-mh --follow
```

If kubectl works after login:
```bash
kubectl logs -n runai-nlm-mh pd-s2-stage-97cwd -f
```
(replace pod name from describe; namespace is usually `runai-nlm-mh`)

## Monitor without logs
Job is still useful even if logs are broken — watch NFS in WinSCP:
- `vlm_hallucination/data/amber/images/` growing
- later `experiment_artifacts/vti/.../textual_v2/`
- workload phase → Completed / Failed via `runai workload list -p nlm-mh`
