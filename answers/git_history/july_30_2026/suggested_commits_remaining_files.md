# Suggested commits for remaining uncommitted work

Date: 2026-07-30. After harness commits `5eff7fe` / `1e02a71`.

## Excluded (large / regenerated artifacts)

| Path | Approx size | Why |
|---|---|---|
| `experiment_artifacts/vti/*/shuffled_control_demos850/` | ~567M + ~359M | Direction/cache trees (mostly `.npz`) |
| `experiment_artifacts/vti/*/shuffled_control_demos850_perlayer/` | ~8.4M + ~6.5M | Per-layer direction `.npz` trees |
| `diagnostic_experiments/perlayer_pca_control/verification/global_fit_directions_sha256_manifest_{pre,post}_run.json` | ~2M each | Byte-identity manifests; regenerable from directions on disk |

Also local-only by `.gitignore`: `implementation_plans/` (including `7-30-26/*.md`). Not suggested unless you intentionally force-add.

## Suggested commits (5)

### 1. demos_850 pool + partition pins

Pin the 850-row demo set and the tooling that builds/resumes it.

- `.gitignore` (remaining `!data/vti/demos_850*` allowlist only)
- `src/paths.py`
- `data/vti/demos_850.jsonl` (~2.0M; same class as tracked `demos_v2.jsonl`)
- `data/vti/demos_850_partition_s42.json`
- `data_scripts/vti_demos_v2/assemble_demos_850.py`
- `data_scripts/vti_demos_v2/stage0_mine_candidates.py` (`--summary-out`)
- `helper_scripts/run_demos850_topup_stages1to4.sh`
- `helper_scripts/run_demos850_continue_after_topup.sh`

Draft message:

```
Add demos_850 pool, seed-42 partition pin, and top-up assembly helpers.
```

### 2. demos_850 / per-layer PCA extraction code

Library + run/verify entry points (no direction bytes).

- `evaluation/interventions/vti/directions_v2.py` (`fit_locus` slug)
- `evaluation/interventions/vti/directions_partition.py`
- `evaluation/interventions/vti/shuffled_control_partition.py`
- `evaluation/interventions/vti/perlayer_pca.py`
- `evaluation/run_scripts/extract_demos850_partition_directions.py`
- `evaluation/run_scripts/extract_demos850_shuffled_control_directions.py`
- `evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py`
- `helper_scripts/verify_demos850_partition_extraction.py`
- `helper_scripts/verify_perlayer_pca_control_extraction.py`

Draft message:

```
Add demos_850 partition, shuffled-control, and per-layer PCA direction extractors.
```

### 3. Small shuffled-control derangement maps

Same pattern as the already-tracked nd200 derangement file; these are tiny JSON maps, not direction tensors.

- `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd50_s1234.json`
- `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd100_s1234.json`
- `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd200_s1234.json`
- `experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd500_s1234.json`

Draft message:

```
Track demos_850 shuffled-image derangement maps for nd50/100/200/500.
```

### 4. Extraction specs (replace renamed file)

- add `extractions/steering_vector_diff_sample_size_07_28_26_extraction.md`
- add `extractions/shuffle_control_vector_diff_sample_size_07_29_26_extraction.md`
- delete `extractions/steering_vector_diff_sample_sizes_07_28_26_extraction.md` (old plural name)

Draft message:

```
Replace sample-size extraction specs; drop plural-named 07_28 duplicate path.
```

### 5. Per-layer PCA control diagnostic record + filled designs

Code, tables, plots, summary, and light verification notes for the geometric comparison — analogous to `e925d1b`, without the large direction trees / 2M sha256 manifests.

**Designs**

- `designs/07_30_26/steering_vector_visual_reasoning_validation.md` (filled)
- `designs/perlayer_pca_control_07_29_26.md` (filled)

**Diagnostic package** (exclude the two `*_sha256_manifest_*.json` files)

- `diagnostic_experiments/perlayer_pca_control/scripts/`
- `diagnostic_experiments/perlayer_pca_control/summaries/`
- `diagnostic_experiments/perlayer_pca_control/tables/`
- `diagnostic_experiments/perlayer_pca_control/plots/`
- `diagnostic_experiments/perlayer_pca_control/verification/activation_cache_untouched_report.md`
- `diagnostic_experiments/perlayer_pca_control/verification/global_fit_directions_byte_identity_report.md`
- `diagnostic_experiments/perlayer_pca_control/verification/perlayer_pca_extraction_manifest_2026-07-29.json`

Draft message:

```
Record per-layer PCA control geometric comparison and related design specs.
```

Optional split: put the two designs in their own commit if you want specs separate from plots/tables.

## Leave out / decide explicitly

- `designs/PCA_visual_reasoning_validation_07_30_26.md` — still an unfilled template (angle-bracket placeholders). Not gate-ready; suggest leaving untracked until filled, or commit only if you want the stub on disk for continuity.
- Full `shuffled_control_demos850*` direction trees — excluded per your request. Note: an earlier commit (`e925d1b`) did track a single nd200 shuffled-control `.npz` pair; this proposal stays stricter for the demos_850 fan-out.
- `implementation_plans/7-30-26/*` — ignored by `.gitignore`; planner outputs stay local unless you force-add.

## Order

1 → 2 → 3 is dependency-friendly (pins before extractors before derangement maps that name those pins). 4 and 5 are independent of that chain and of each other.
