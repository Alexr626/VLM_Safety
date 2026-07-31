# Tarball location for WinSCP

`git archive` to `/tmp` may not be visible from WinSCP / another view of the host. Use:

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o ~/dev/vti_repo.tar.gz HEAD
```

File: `/home/romanus/dev/vti_repo.tar.gz`
