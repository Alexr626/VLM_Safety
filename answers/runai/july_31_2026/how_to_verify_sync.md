# How to verify a Completed sync job on RunAI

Date: 2026-07-31

`Phase: Completed` only means the pod exited 0. For `sync-steering-llava`,
`SYNC_OK` in the logs means the outer command reached `echo SYNC_OK`. In the
submit that was used, `update_repo_from_tarball.sh` was invoked via
`bash …/run_bash_lf.py` (wrong — needs `python3`), so those checks did not run;
`|| true` also swallowed the failure.

## Quick checks

```bash
runai training describe sync-steering-llava -p nlm-mh   # Phase / Pod Succeeded
runai training logs sync-steering-llava -p nlm-mh | tail -80
```

## Strong check: submit a verify probe

Mount NFS and `test` the files the smoke/full jobs need (scripts, 4 meandiff
dirs with `directions.npz`, optional CHAIR cells, micromamba env). Look for
`VERIFY_SYNC_OK` in the logs.

Also usable from WinSCP: browse `/airl-datalake/romanus/vlm_hallucination/` for
the same paths.

## Result (2026-07-31)

First verify probe failed on bash quoting. Resubmit `verify-sync2` **Completed** with:

- required scripts + `SKIP_CONDA_ACTIVATE` present
- `meandiff_dirs=4` with all `directions.npz`
- `chair_summaries=3`
- micromamba + `envs/vlm_hal` present
- `VERIFY_SYNC_OK`
