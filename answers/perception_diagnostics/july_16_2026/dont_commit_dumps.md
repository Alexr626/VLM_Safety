# Avoid committing dump acts/norms for NFS staging

**Problem:** `diagnostic_experiments/perception_diag/` has ~4088 files; ~4000 are `acts/*.npy` + `norms/*.npy` under `llava-…/dumps/` (~604MB). Those are run products, not code.

## Do this

1. **Gitignore the dumps** (and `__pycache__`), then commit only source + small artifacts:
   - `checks/`, `capture/`, `analyze/`, `hd_qc/`, `run_scripts/`, `templates/`
   - `run_dump.py`, `augment/build_augmented_jsonl.py`
   - optional small outputs: `artifact_checks/*.json|html|md|csv`, `augment/outputs/*.jsonl`
   - **not** `**/dumps/**/acts/`, `**/dumps/**/norms/`, full dump trees

2. **Suggested `.gitignore` lines:**
```gitignore
diagnostic_experiments/perception_diag/**/dumps/
diagnostic_experiments/perception_diag/**/__pycache__/
diagnostic_experiments/**/__pycache__/
```

3. **NFS still needs (via WinSCP, outside git):**
   - AMBER images (`data/amber/images/`)
   - LLaVA demos_v2 direction cache (`experiment_artifacts/.../textual_v2/demosv2_9a44f4af_all_nd200_...`)
   - After commit: `git archive` code tarball as usual
   - S3 **re-generates** dumps on the cluster — no need to upload lambdab2 acts/norms

4. If you want a local backup of dumps, keep them on lambdab2 `/data` or a separate NFS folder; just don’t put them in git.
