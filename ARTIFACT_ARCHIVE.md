# Artifact archive and restore

Large gitignored trees are **not** on GitHub/GitLab. Pack them to Google Drive, then unpack into a clone.

**Do not archive:** HuggingFace model weights (`HF_HOME`), `.env`, `wandb/` run DBs, nested unused vendor clones.

Drive folder URL (fill after upload):

```
<PASTE_GOOGLE_DRIVE_FOLDER_URL_HERE>
```

---

## Pack A — restore-critical (recommended first)

Enough to reopen the POPE 06-19 vs 07-30 thread and continue demos850 steering without re-fitting every direction from scratch.

| Tarball (suggested name) | Contents | Approx size (2026-08-07) |
|--------------------------|----------|---------------------------|
| `packA_directions_no_act_cache.tar.gz` | `experiment_artifacts/vti/` **excluding** any `_act_cache/` directories | ~0.3–0.5G |
| `packA_eval_results_key_dates.tar.gz` | `evaluation/results/{2026-06-19,2026-07-30,2026-08-05,2026-08-06}/` | ~0.2G |
| `packA_generated_data_vti_and_pins.tar.gz` | `data/vti/` demos/partitions/json (+ `v2/` staging if present); already-tracked pins are in git but included for convenience | varies (exclude huge dumps if desired) |

Optional add-on if you need to re-extract meandiff/PCA without forwards:

| Tarball | Contents | Approx size |
|---------|----------|-------------|
| `packA_act_caches_textual_v2.tar.gz` | `experiment_artifacts/vti/*/textual_v2/_act_cache/` | ~3G |

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

Skip deciding COCO vs pins. Entire `data/` tree (~51G on lambdab2 as of 2026-08-07, dominated by `data/coco/` ~39G).

```bash
OUT=/tmp/vlm_artifact_packs
mkdir -p "$OUT"
tar -czf "$OUT/packB_full_data.tar.gz" data
ls -lh "$OUT/packB_full_data.tar.gz"
```

Alternatively re-download benchmarks with `bash data_scripts/download_all_benchmarks.sh` and only restore Pack A.

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
