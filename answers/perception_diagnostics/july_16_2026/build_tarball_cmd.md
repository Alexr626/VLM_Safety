# Build vti_repo.tar.gz outside /tmp

```bash
cd ~/dev/vlm_hallucination_mitigation_summer_2026
git archive --format=tar.gz -o ~/dev/vti_repo.tar.gz HEAD
ls -lh ~/dev/vti_repo.tar.gz
tar -tzf ~/dev/vti_repo.tar.gz | grep -E 'run_perception_s2_stage|demos_v2.jsonl|run_dump'
```

File: `/home/romanus/dev/vti_repo.tar.gz`
