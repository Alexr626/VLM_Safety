# Viewing RunAI job stdout while the container runs

Yes — logs are available from outside the container via the RunAI CLI (Kubernetes streams container stdout/stderr).

```bash
# follow live
runai training logs pd-s2-stage -p nlm-mh --follow

# or without follow
runai training logs pd-s2-stage -p nlm-mh
```

If that alias fails, try:
```bash
runai logs pd-s2-stage -p nlm-mh
```

Workload status (phase / node):
```bash
runai workload list -p nlm-mh
runai training standard describe pd-s2-stage
```

Persistent artifacts (AMBER images, direction caches, etc.) land on NFS under `/airl-datalake/romanus/vlm_hallucination/` and stay after the pod exits — that is separate from the log stream.
