# Artifact archive and restore

Large gitignored trees are **not** on GitHub/GitLab. Pack them to Google Drive, then unpack into a clone.

**Do not archive:** HuggingFace model weights (`HF_HOME`), `.env`, `wandb/` run DBs, nested unused vendor clones.

Drive folder URL (fill after upload):

```
<PASTE_GOOGLE_DRIVE_FOLDER_URL_HERE>
```

## GitHub full-tree push (Phase 2 note)

Pushing `new_research_workflow` → `git@github.com:Alexr626/VLM_Safety.git` branch `VLM_hallucination_mitigation` from lambdab2 failed with `Permission denied (publickey)` — the host key `romanus@lambdab2` is not authorized on that GitHub account.

**Offline alternative already built on this machine:**

```
exports/new_research_workflow_for_github.bundle   (~436M; gitignored under exports/)
```

On a machine that can push to `Alexr626/VLM_Safety`:

```bash
git clone git@github.com:Alexr626/VLM_Safety.git
cd VLM_Safety
git fetch /path/to/new_research_workflow_for_github.bundle new_research_workflow:new_research_workflow
git checkout VLM_hallucination_mitigation
git merge --ff-only new_research_workflow   # or reset --hard if you intend to replace tip
git push origin VLM_hallucination_mitigation
```

Or authorize the lambdab2 ed25519 public key on GitHub, then from the internship clone:

```bash
git push -u github new_research_workflow:VLM_hallucination_mitigation
```

---

## Pack A — restore-critical (recommended first)

Enough to reopen the POPE 06-19 vs 07-30 thread and continue demos850 steering without re-fitting every direction from scratch.

**Built on lambdab2 2026-08-07** under `/tmp/vlm_artifact_packs/` (upload these to Drive):

| Tarball | Contents | Size on disk |
|--------------------------|----------|-----------|
| `packA_directions_no_act_cache.tar.gz` | `experiment_artifacts/vti/` excluding `_act_cache/` | **276M** |
| `packA_eval_results_key_dates.tar.gz` | `evaluation/results/{2026-06-19,2026-07-30,2026-08-05,2026-08-06}/` | **35M** |
| `packA_generated_data_vti_and_pins.tar.gz` | `data/vti/` + pinned amber/chair/pope JSON | **798M** |
| `packA_act_caches_textual_v2.tar.gz` (optional) | `textual_v2/_act_cache/` for llava + qwen2.5 + qwen2 | **2.9G** |

Also offline GitHub mirror bundle (gitignored): `exports/new_research_workflow_for_github.bundle` (~436M).

### Build Pack A (from repo root)

```bash
OUT=/tmp/vlm_artifact_packs
mkdir -p "$OUT"

# Directions without act caches
tar --exclude='*/_act_cache' --exclude='*/_act_cache/*' \
  -czf "$OUT/packA_directions_no_act_cache.tar.gz" experiment_artifacts/vti

# Key result dates
tar -czf "$OUT/packA_eval_results_key_dates.tar.gz" \
  evaluation/results/2026-06-19 \
  evaluation/results/2026-07-30 \
  evaluation/results/2026-08-05 \
  evaluation/results/2026-08-06

# Generated VTI data (demos / partitions / staging)
tar -czf "$OUT/packA_generated_data_vti_and_pins.tar.gz" data/vti \
  data/amber/pinned_amber_disc_1500.json \
  data/amber/pinned_amber_disc_450.json \
  data/amber/pinned_amber_disc_100.json \
  data/chair/pinned_chair_500.json \
  data/pope/pinned_pope_existence_yes_30.json \
  data/pope/pinned_pope_existence_no_30.json \
  data/pope/pinned_pope_existence_yes_120.json \
  data/pope/pinned_pope_existence_no_120.json \
  2>/dev/null || true

# Optional act caches
tar -czf "$OUT/packA_act_caches_textual_v2.tar.gz" \
  experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/_act_cache \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/_act_cache \
  experiment_artifacts/vti/qwen2-vl-7b-instruct/textual_v2/_act_cache \
  2>/dev/null || true

ls -lh "$OUT"
```

---

## Pack B — optional full `data/`

Skip deciding COCO vs pins. Entire `data/` tree.

**Built on lambdab2 2026-08-07:** `/tmp/vlm_artifact_packs/packB_full_data.tar.gz` (**48G**). Durable copy path (if synced): `/home/romanus/vlm_artifact_packs_2026-08-07/`.

Alternatively re-download benchmarks with `bash data_scripts/download_all_benchmarks.sh` and only restore Pack A.

```bash
OUT=/tmp/vlm_artifact_packs
mkdir -p "$OUT"
tar -czf "$OUT/packB_full_data.tar.gz" data
ls -lh "$OUT/packB_full_data.tar.gz"
```


---

## Restore into a fresh clone

```bash
cd /path/to/cloned/repo
# After uploading tarballs to Drive and downloading them locally:
tar -xzf packA_directions_no_act_cache.tar.gz
tar -xzf packA_eval_results_key_dates.tar.gz
tar -xzf packA_generated_data_vti_and_pins.tar.gz
# optional:
# tar -xzf packA_act_caches_textual_v2.tar.gz
# tar -xzf packB_full_data.tar.gz

export HF_HOME=/path/to/huggingface   # models download separately
conda env create -f environment.yml
conda activate vlm_hallucination_mitigation
```

Re-run the matched POPE analysis (no generation):

```bash
python evaluation/pope_0619_vs_0730_matched/build_comparison.py
```

---

## Checklist after restore

- [ ] `data/vti/demos_850.jsonl` and `demos_850_partition_s42.json` present
- [ ] Meandiff + r2 nd500 dirs under `experiment_artifacts/vti/{llava,qwen2.5}/textual_v2/`
- [ ] `evaluation/results/2026-06-19` and `2026-07-30` LLaVA POPE cells present
- [ ] `.env` created locally if needed — never from the archive
