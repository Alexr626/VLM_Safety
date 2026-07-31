# `git archive` failed from `/tmp`

`fatal: not a git repository` means you ran the command in `/tmp`, not the repo.

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o /tmp/vti_repo.tar.gz HEAD
tar -tzf /tmp/vti_repo.tar.gz | grep -E 'run_perception_s3_smoke|perception_diag/run_dump'
```

`git archive` only includes **committed** files at `HEAD`. Commit `perception_diag` + the S3 helper first if those greps miss.
