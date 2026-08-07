# Artifact archive and restore

Large gitignored trees are **not** on GitHub/GitLab. Pack them to Google Drive, then unpack into a clone.

**Do not archive:** HuggingFace model weights (`HF_HOME`), `.env`, `wandb/` run DBs, nested unused vendor clones.

Drive folder URL (fill after upload):

```
<PASTE_GOOGLE_DRIVE_FOLDER_URL_HERE>
```

**Pack locations on lambdab2 (identical):**

- `/tmp/vlm_artifact_packs/`
- `/home/romanus/vlm_artifact_packs_2026-08-07/`

---

## Upload order (Google Drive)

Upload in this order so a partial upload still covers the POPE thread first:

| Priority | Tarball | Size | Notes |
|----------|---------|------|-------|
| 1 | `packA_directions_no_act_cache.tar.gz` | 276M | All VTI dirs including shuffled / perlayer / visual; no act caches |
| 2 | `packA_eval_results_key_dates.tar.gz` | 35M | 2026-06-19, 07-30, 08-05, 08-06 |
| 3 | `packA_generated_data_vti_and_pins.tar.gz` | 798M | `data/vti/` + pins |
| 4 | `packA_act_caches_textual_v2.tar.gz` | 2.9G | Re-extract meandiff/PCA without forwards |
| 5 | `packC_shuffled_act_caches.tar.gz` | 1.1G | Shuffled-control act caches (legacy + demos850) |
| 6 | `packD_eval_results_remaining.tar.gz` | 1.8G | Other result dates (incl. ~2.4G source `2026-07-13`) |
| 7 | `packE_amber_pope_dumps.tar.gz` | 7.7G | Steered dumps only (skip if uploading Pack B) |
| 8 | `packB_full_data.tar.gz` | 48G | Full `data/` (COCO + dumps + everything) |

Also optional: `exports/new_research_workflow_for_github.bundle` (~436M) for GitHub mirror if SSH push is unavailable.

After upload, paste the Drive folder URL above and commit this file on `new_research_workflow` / `manager_handoff`.

---

## GitHub full-tree push (Phase 2 note)

Pushing `new_research_workflow` → `git@github.com:Alexr626/VLM_Safety.git` branch `VLM_hallucination_mitigation` from lambdab2 failed with `Permission denied (publickey)` — the host key `romanus@lambdab2` is not authorized on that GitHub account.

Authorize this public key on GitHub (Settings → SSH and GPG keys):

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFUg5yCYE3FmgNM+DpuWGSV30U5gVSWat+zyzjNncsKV romanus@lambdab2
```

Then:

```bash
ssh -T git@github.com
cd /home/romanus/dev/vlm_hallucination_mitigation_summer_2026
git fetch github
git push -u github new_research_workflow:VLM_hallucination_mitigation
```

**Offline alternative already built:**

```
exports/new_research_workflow_for_github.bundle   (~436M; gitignored under exports/)
```

---

## Pack A — restore-critical (recommended first)

Enough to reopen the POPE 06-19 vs 07-30 thread and continue demos850 steering without re-fitting every direction from scratch.

| Tarball | Contents | Size |
|---------|----------|------|
| `packA_directions_no_act_cache.tar.gz` | `experiment_artifacts/vti/` excluding `_act_cache/` (includes textual_v2, meandiff, perlayer, shuffled dirs, visual) | **276M** |
| `packA_eval_results_key_dates.tar.gz` | `evaluation/results/{2026-06-19,2026-07-30,2026-08-05,2026-08-06}/` | **35M** |
| `packA_generated_data_vti_and_pins.tar.gz` | `data/vti/` + pinned amber/chair/pope JSON | **798M** |
| `packA_act_caches_textual_v2.tar.gz` | `textual_v2/_act_cache/` for llava + qwen2.5 + qwen2 | **2.9G** |

### Rebuild Pack A (from repo root)

```bash
OUT=/tmp/vlm_artifact_packs
mkdir -p "$OUT"

tar --exclude='*/_act_cache' --exclude='*/_act_cache/*' \
  -czf "$OUT/packA_directions_no_act_cache.tar.gz" experiment_artifacts/vti

tar -czf "$OUT/packA_eval_results_key_dates.tar.gz" \
  evaluation/results/2026-06-19 \
  evaluation/results/2026-07-30 \
  evaluation/results/2026-08-05 \
  evaluation/results/2026-08-06

tar -czf "$OUT/packA_generated_data_vti_and_pins.tar.gz" data/vti \
  data/amber/pinned_amber_disc_1500.json \
  data/amber/pinned_amber_disc_450.json \
  data/amber/pinned_amber_disc_100.json \
  data/chair/pinned_chair_500.json \
  data/pope/pinned_pope_existence_yes_30.json \
  data/pope/pinned_pope_existence_no_30.json \
  data/pope/pinned_pope_existence_yes_120.json \
  data/pope/pinned_pope_existence_no_120.json

tar -czf "$OUT/packA_act_caches_textual_v2.tar.gz" \
  experiment_artifacts/vti/llava-1.5-7b-hf/textual_v2/_act_cache \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/textual_v2/_act_cache \
  experiment_artifacts/vti/qwen2-vl-7b-instruct/textual_v2/_act_cache
```

---

## Pack B — optional full `data/`

Entire `data/` tree (**48G** tarball). Skip Pack E if you upload this.

```bash
OUT=/tmp/vlm_artifact_packs
tar -czf "$OUT/packB_full_data.tar.gz" data
```

Alternatively re-download benchmarks with `bash data_scripts/download_all_benchmarks.sh` and restore Pack A (+ C/D as needed).

---

## Pack C — shuffled-control act caches

Re-extract shuffled-image control directions without GPU forwards.

| Tarball | Contents | Size |
|---------|----------|------|
| `packC_shuffled_act_caches.tar.gz` | `shuffled_control/_act_cache/` and `shuffled_control_demos850/_act_cache/` for llava + qwen2.5 | **1.1G** |

```bash
OUT=/tmp/vlm_artifact_packs
tar -czf "$OUT/packC_shuffled_act_caches.tar.gz" \
  experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control/_act_cache \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control/_act_cache \
  experiment_artifacts/vti/llava-1.5-7b-hf/shuffled_control_demos850/_act_cache \
  experiment_artifacts/vti/qwen2.5-vl-7b-instruct/shuffled_control_demos850/_act_cache
```

---

## Pack D — remaining evaluation results

Everything under `evaluation/results/` **except** the four Pack A key dates (avoids duplicating 06-19 / 07-30 / 08-05 / 08-06).

| Tarball | Contents | Size |
|---------|----------|------|
| `packD_eval_results_remaining.tar.gz` | Other result trees (notably `2026-07-13`, plus 06-18, 06-22, 07-02, smokes, `_logs`, …) | **1.8G** |

```bash
OUT=/tmp/vlm_artifact_packs
tar -czf "$OUT/packD_eval_results_remaining.tar.gz" \
  --exclude='evaluation/results/2026-06-19' \
  --exclude='evaluation/results/2026-07-30' \
  --exclude='evaluation/results/2026-08-05' \
  --exclude='evaluation/results/2026-08-06' \
  evaluation/results
```

---

## Pack E — AMBER / POPE steered dumps only

Useful if you skip Pack B but want dump trees without full COCO.

| Tarball | Contents | Size |
|---------|----------|------|
| `packE_amber_pope_dumps.tar.gz` | `data/amber/dumps/`, `data/pope/dumps/` | **7.7G** |

```bash
OUT=/tmp/vlm_artifact_packs
tar -czf "$OUT/packE_amber_pope_dumps.tar.gz" data/amber/dumps data/pope/dumps
```

---

## Restore into a fresh clone

```bash
cd /path/to/cloned/repo
# After downloading tarballs from Drive:
tar -xzf packA_directions_no_act_cache.tar.gz
tar -xzf packA_eval_results_key_dates.tar.gz
tar -xzf packA_generated_data_vti_and_pins.tar.gz
# recommended for low-compute re-extraction:
tar -xzf packA_act_caches_textual_v2.tar.gz
tar -xzf packC_shuffled_act_caches.tar.gz
# historical eval cells:
tar -xzf packD_eval_results_remaining.tar.gz
# dumps without full COCO, OR full data:
# tar -xzf packE_amber_pope_dumps.tar.gz
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
- [ ] Optional: shuffled + textual_v2 act caches restored if you expect to re-extract
- [ ] `.env` created locally if needed — never from the archive
