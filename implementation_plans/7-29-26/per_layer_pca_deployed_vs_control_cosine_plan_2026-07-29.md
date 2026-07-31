# Per-layer PCA vs global PCA: deployed-against-control cosine over depth
design_spec: designs/perlayer_pca_control_07_29_26.md

Date: 2026-07-29
Status: ready to implement, gated on the sanity checks below.

---

## Question

Does refitting the VTI steering direction with one PCA per decoder layer — instead of one
global PCA over the flattened `(num_layers+1) x hidden` diff vector — lower the per-layer
cosine between the deployed direction and its shuffled-image control at mid-to-late layers,
on both models and particularly on LLaVA, where the global-fit direction norm is roughly
constant across layers?

The comparison is always **deployed direction against shuffled-image control direction**,
both arms fit under the same scheme, one cosine per decoder layer. It is never per-layer-fit
against global-fit. Each cell yields two curves over depth: one from the new per-layer
directions and one from the existing global-fit directions.

## Design spec reference

`designs/perlayer_pca_control_07_29_26.md`.

The spec's **Primitives this design requires** section lists two primitives — per-layer-PCA
deployed directions and per-layer-PCA control directions, both at nd 50/100/200/500 on both
models — as strict prerequisites of the named measurement. Neither exists on disk. The steps
that produce them are Sections 2 and 3 below, under this design spec. No extraction spec is
written and no primitive outside that list is produced.

## Cells

| Cell | Model (`model_short`) | Decoder layers | Item set | Sample sizes | Intervention |
|---|---|---|---|---|---|
| 1 | `qwen2.5-vl-7b-instruct` | 28 (indices 0–27), hidden 3584 | demos850 `all` | 50 / 100 / 200 / 500 | none — geometric only |
| 2 | `llava-1.5-7b-hf` | 32 (indices 0–31), hidden 4096 | demos850 `all` | 50 / 100 / 200 / 500 | none — geometric only |

Each cell is evaluated under two fit schemes (per-layer, global) and two arms (deployed,
shuffled-image control). 2 models x 4 sample sizes x 2 arms x 2 schemes = 32 direction
objects, of which 16 already exist (global) and 16 are produced here (per-layer).

No model is loaded. No steering is applied. No benchmark is run.

## Data

All inputs exist and are read-only. Nothing here is regenerated.

**Demo pool and blocks**

- `data/vti/demos_850.jsonl` — 850 rows, content hash `ba05bd960cad0c18`, slug prefix `ba05bd96`.
- `data/vti/demos_850_partition_s42.json` — disjoint blocks of 50 / 100 / 200 / 500, seed 42,
  `selection_policy = "disjoint_partition"`. The four blocks are disjoint and their union is
  all 850 ids, so the four sample sizes are four independent draws, not nested prefixes.

**Existing global-fit directions (baseline arm of the comparison, read-only)**

- Deployed: `experiment_artifacts/vti/{model_short}/textual_v2/demos850_ba05bd96_all_nd{N}_s42_r2_partition/`
  for N in {50, 100, 200, 500}. Each holds `directions.npz`, `components.npz`, `metadata.json`.
- Control: `experiment_artifacts/vti/{model_short}/shuffled_control_demos850/all_nd{N}/`
  for the same N. Same three files.

**Cached activation stacks (read-only, zero forward passes)**

- Deployed arm: `experiment_artifacts/vti/{model_short}/textual_v2/_act_cache/` —
  5100 files per model, verified present: 850 ids x 6 variants
  (`vl_v2_value`, `vl_v2_all`, `vl_v2_existence`, `vl_v2_attribute`, `vl_v2_counting`,
  `vl_v2_relation`). This run touches only `vl_v2_value` and `vl_v2_all`.
- Control arm: `experiment_artifacts/vti/{model_short}/shuffled_control_demos850/_act_cache/` —
  1700 files per model, verified present: 850 ids x 2 variants (`vl_v2_value`, `vl_v2_all`).
  Same demo ids and captions, images deranged within block.

Each cached `.npz` holds keys `layer_0 … layer_{num_layers}` — `num_layers+1` rows
(embedding row + every decoder layer), float32, hidden-dim vectors. Verified: LLaVA files
have 33 keys of shape `(4096,)`; Qwen has 29 of shape `(3584,)`.

**Derangement files (read-only)**

`experiment_artifacts/vti/shuffled_control_demos850_image_derangement_nd{N}_s1234.json`
for the four N. Loaded with `load_or_write_block_derangement(..., force=False)`, which
validates and does not rewrite an existing file.

**Item-set note.** No new item set is constructed. The demo ids for each cell are read from
the `ids_used` field of the corresponding existing `metadata.json` — for the deployed arm from
the global-fit partition cell, for the control arm from the global-fit control cell — and
cross-checked against `data/vti/demos_850_partition_s42.json` block `str(N)`. That guarantees
the two schemes are fit on identical demos in identical order.

---

## Sanity checks that must pass first

These gate everything below. Implement them as a single script,
`helper_scripts/verify_perlayer_pca_control_extraction.py`, with `--mode record` (run before
any extraction) and `--mode verify` (run after). Checks marked **gate** halt the run on
failure. Checks marked **report** produce numbers Alex reads before the cosine curves are read;
the script does not decide whether they invalidate the design.

### Check 1 — pre-run hash snapshot of every existing global-fit artifact (gate)

`--mode record` writes SHA-256, byte size, and mtime for every file under all 16 existing
global-fit cell directories (8 deployed, 8 control, across both models) to

```
diagnostic_experiments/perlayer_pca_control/global_fit_directions_sha256_manifest_pre_run.json
```

It also records the file count, and the `(name, size, mtime)` triples, for both `_act_cache`
directories on both models (expected 5100 and 1700).

This snapshot must be taken **before** any code from Section 2 runs. These artifacts are not
git-tracked — `.gitignore` line 145 ignores any path component matching `textual_v2*`, and the
`shuffled_control_demos850/` tree is untracked — so git cannot serve as the byte-identity
baseline. The manifest is the only record.

Invalidates the design if: the manifest cannot be written, or any listed directory is missing.

### Check 2 — the per-layer fit differs from the global fit only in fit locus (gate)

The global scheme's PCA mean, reshaped to `(num_layers+1, hidden)`, is exactly the per-layer
mean of the diffs at each layer — the flattening is layer-major, so `mean` over demos commutes
with the reshape. A per-layer refit that changes anything other than the fit locus will break
that identity.

For each of the 16 cells, compare `components.npz["pca_mean_flat"]` from the existing global
cell, reshaped to `(num_layers+1, hidden)`, against the per-layer fit's `mean_[:, 0, :]`.
Verified reachable: `pca_mean_flat` has shape `(135168,) = 33 x 4096` for LLaVA and
`(103936,) = 29 x 3584` for Qwen.

Expect max absolute difference 0.0. A dry-run on random tensors of both models' shapes gives
exactly 0.0. Gate at `max_abs_diff <= 1e-6 * max(abs(global_mean))`; record the observed value.

Also assert, in the same check, that the per-layer reconstruction is literally
`components_[l, 0, :] + mean_[l, 0, :]` per row, and that dropping row 0 yields
`(num_layers, hidden)` matching the existing global cell's `directions.npz` shape.

Invalidates the design if this fails: the two arms would then differ in more than the fit
locus, and the spec's first stated assumption does not hold.

### Check 3 — existing slug strings are unchanged by the new fit-locus field (gate)

Assert, with no `fit_locus` argument supplied:

```python
partition_slug("ba05bd960cad0c18", "all", N, seed=42, rank=2)
  == f"demos850_ba05bd96_all_nd{N}_s42_r2_partition"     # for N in 50,100,200,500
textual_v2_slug("9a44f4afde0324b5", "all", 200, seed=42, rank=2)
  == "demosv2_9a44f4af_all_nd200_s42_r2_prefix"
```

and that `partition_cache_dir(model_short, partition_slug(...))` still resolves to an existing
directory containing `directions.npz` for all 8 existing deployed cells.

Invalidates the design if this fails: the shadowing guard the spec requires is not in place and
the baseline arm is at risk.

### Check 4 — the two activation caches are genuinely different data (gate)

The cache key is `(demo_id, variant)` with no image path in it. That is exactly why the control
arm has its own namespace. Guard against the two arms accidentally reading the same cache: for
5 demo ids per model (the first 5 ids of the nd50 block), load the `vl_v2_all` stack from
`textual_v2/_act_cache` and from `shuffled_control_demos850/_act_cache` and assert the max
absolute difference is strictly greater than 0 at every layer row, and that the per-layer
cosine between them is below 1.0 at some layer.

Invalidates the design if this fails: the control arm is not a control.

### Check 5 — item-set identity between the two schemes (gate)

For each of the 16 cells:

- The ids the per-layer fit reads equal, elementwise and in order, the `ids_used` list in the
  corresponding existing global `metadata.json`.
- For the deployed arm, that list equals `data/vti/demos_850_partition_s42.json` `blocks[str(N)]`
  elementwise and in order.
- `check_partition_integrity(demos_path, partition_obj)` passes: 850 unique ids, blocks disjoint,
  union equal to the demo pool, hash `ba05bd960cad0c18`.
- For the control arm, `load_or_write_block_derangement(ids_used, N, deployed_slug=..., force=False)`
  returns a mapping with 0 fixed points whose key set equals `ids_used`, read from the existing
  `..._nd{N}_s1234.json` without rewriting it.

Invalidates the design if this fails: the two schemes are not compared on identical data.

### Check 6 — output shapes and finiteness (gate)

Per-layer directions are `(28, 3584)` for Qwen and `(32, 4096)` for LLaVA, all entries finite,
no all-zero layer row, per-layer L2 norm strictly positive at every decoder layer.

### Check 7 — no writes to any activation cache, and zero forward passes (gate)

`--mode verify` re-reads the `_act_cache` file counts and `(name, size, mtime)` triples recorded
in Check 1 and asserts they are identical. The extraction code must load stacks with
`ActivationCache.load(...)`, which raises `FileNotFoundError` on a miss, never
`ensure_variant_activation`, which forwards and writes on a miss. The extraction script must not
import `src.model` or construct a wrapper; every per-layer cell's metadata carries
`"forwards_executed": 0` and `"act_cache_read_only": true`.

### Check 8 — PC1 share of direction norm, and direction norm progression over depth (report)

The spec's second stated assumption. For every cell, scheme, and arm, compute per decoder layer:

- `pc1_norm[l] = ||components.npz["pc0"][l+1]||` (row 0 is the embedding row, dropped to align
  with `directions.npz`)
- `direction_norm[l] = ||directions.npz["layer_{l}"]||`
- `pc1_share[l] = pc1_norm[l] / direction_norm[l]`

Report per model and scheme: min, median, and max of `pc1_share` across decoder layers; the
full per-layer values in the CSV; and whether `direction_norm` is monotonically non-decreasing
in `l` (report the number of descending steps and the largest descending step, not a verdict).

The spec's stated expectation is LLaVA's PC1 share rising from about 9% under the global fit to
20–72% under the refit, and Qwen's from about 0.8% to 1.5–11%. Report the observed values
against those; do not gate on them. The spec's stated invalidator is PC1 shares decreasing, and
specifically direction norms not increasing from layer to layer.

### Check 9 — PC1 sign convention diagnostic (report)

Grounding fact, verified by reading `evaluation/interventions/vti/pca.py`: `svd_flip` gathers
the sign from column 0 of `U` for every component (`torch.gather(u, 1, max_abs_cols)` with an
index of trailing dim 1). For component 0 — the only one that enters the direction — this is
the standard convention: the sign of the largest-magnitude entry of `U`'s first column. So the
quirk does not affect the measured quantity. It does affect PC2, which is stored but unused.

What does change between schemes: under the global fit one sign is chosen for the whole
flattened PC1, so PC1's orientation is consistent across layers within a fit. Under the
per-layer fit a sign is chosen independently at every layer, in each arm separately. Where the
PC1 share from Check 8 is large, a per-layer sign disagreement between the deployed and control
fits moves the cosine at that layer.

Report, per layer, arm, cell, and scheme, into the CSV: `cos(PC1_l, mean_l)` and its sign,
alongside `pc1_share[l]`. No threshold, no annotation on the cosine plots, no verdict.

**Also report, per layer and cell, whether the two arms agree.** The quantity that matters is
not either arm's sign on its own — it is whether the deployed and control fits chose the same
orientation at the same layer, because only a disagreement moves the primary measurement.
Emit `sign(cos(PC1_l, mean_l))` for each arm and a boolean `pc1_sign_agrees_between_arms[l]`,
and plot the per-layer agreement pattern (Section on Artifacts). Compute it for both fit
schemes; under the global scheme one sign covers the whole flattened PC1, so its per-layer
pattern is constant by construction and serves as the reference the per-layer pattern is read
against.

Decided by Alex, 2026-07-29, resolving Open question 1: **report only, apply no correction.**
Do not pin the sign convention, do not align either arm to the other, and do not align PC1 to
the mean. Any correction changes what the run measures and is a separate decision, to be taken
after this diagnostic and the cosine curves are both on disk. The implementer applies no sign
handling beyond what `svd_flip` already does.

---

## Code to write

All new unless stated. Two small edits to existing modules, both default-preserving.

### 2.1 Edit — fit-locus field in the slug builders

`evaluation/interventions/vti/directions_partition.py`:

```python
FIT_LOCUS_GLOBAL = "global"
FIT_LOCUS_PERLAYER = "perlayer"

def partition_slug(
    demos_hash: str,
    dimension: str,
    num_demos: int,
    seed: int = 42,
    rank: int = 2,
    fit_locus: str = FIT_LOCUS_GLOBAL,
) -> str:
    base = (
        f"{SLUG_STEM}_{demos_hash[:8]}_{dimension}_nd{num_demos}_"
        f"s{seed}_r{rank}_partition"
    )
    if fit_locus == FIT_LOCUS_GLOBAL:
        return base
    if fit_locus == FIT_LOCUS_PERLAYER:
        return f"{base}_perlayer"
    raise ValueError(f"fit_locus must be one of ('global', 'perlayer'); got {fit_locus!r}")
```

`evaluation/interventions/vti/directions_v2.py::textual_v2_slug` takes the same defaulted
`fit_locus` parameter with the same semantics. The spec names `textual_v2_slug` explicitly in
its shadowing analysis; the builder that actually produces the demos850 slugs under test is
`partition_slug`, so both get the field. Neither changes any string when the argument is
omitted, which is what Check 3 asserts.

`partition_slug` is called with positional/keyword arguments in
`directions_partition.py` (three call sites), `shuffled_control_partition.py`
(`deployed_partition_slug`), and `helper_scripts/verify_demos850_partition_extraction.py`.
None of them pass `fit_locus`, so all continue to resolve to the existing directories.

### 2.2 New module — `evaluation/interventions/vti/perlayer_pca.py`

```python
FIT_LOCUS_PERLAYER = "perlayer"
FIT_LOCUS_DESCRIPTION = {
    "global": "one PCA over the flattened (num_layers+1)*hidden diff vector",
    "perlayer": "one centered PCA per (num_layers+1) row, on that row's hidden-dim diffs",
}

def perlayer_partition_cache_dir(model_short: str, slug: str) -> Path:
    """experiment_artifacts/vti/{model_short}/textual_v2_perlayer/{slug}"""

def perlayer_shuffled_control_dir(model_short: str, num_demos: int) -> Path:
    """experiment_artifacts/vti/{model_short}/shuffled_control_demos850_perlayer/all_nd{N}_perlayer"""

def load_stack_readonly(cache: ActivationCache, demo_id: str, variant: str) -> torch.Tensor:
    """(num_layers+1, hidden) float32. Raises FileNotFoundError on a cache miss.
    Never forwards, never writes."""

def perlayer_pca_fit(
    diffs_3d: torch.Tensor,   # (num_layers+1, n_pairs, hidden), float32
    rank: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[float]]]:
    """(components (L+1, rank, hidden), mean (L+1, hidden),
        direction (L+1, hidden), evr_per_layer (L+1 lists of length rank))"""

def obtain_textual_vti_perlayer_from_stacks(
    h_stacks: Sequence[torch.Tensor],
    value_stacks: Sequence[torch.Tensor],
    rank: int = 2,
) -> Tuple[torch.Tensor, dict]:
    """Same signature and return contract as
    directions_v2.obtain_textual_vti_v2_from_stacks, per-layer fit."""

def compute_perlayer_direction_for_cell(
    model_short: str,
    arm: str,              # "deployed" | "control"
    num_demos: int,
    *,
    demos_path: Optional[Path] = None,
    partition_path: Optional[Path] = None,
    rank: int = 2,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, dict, dict]:
    """Returns (directions (num_layers, hidden), meta, checks)."""
```

**`perlayer_pca_fit` reuses `evaluation/interventions/vti/pca.py::PCA` unmodified.** `PCA.fit`
already takes the 3-D branch: given `(L+1, N, hidden)` it sets `mean_` to `(L+1, 1, hidden)`,
runs a batched `torch.linalg.svd`, and returns `components_` of shape `(L+1, rank, hidden)`.
Verified by direct execution at both models' shapes. This is the batched per-layer fit the
design asks for; the global path in `directions_v2._live_pca_fit` reaches the 2-D branch only
because it flattens the diffs first.

`_live_pca_fit` cannot be reused, because its explained-variance block indexes `S[0]` — batch 0
of a single-batch fit. The per-layer version computes `evr[l][j] = S[l, j]**2 / sum_j S[l, j]**2`,
one ratio vector per layer.

Diff construction mirrors `obtain_textual_vti_v2_from_stacks`: `value − h_value` per pair, per
layer, i.e. `diffs_3d[:, i, :] = value_stacks[i].float() - h_stacks[i].float()`. Same polarity
constant `DIFF_POLARITY = "value_minus_h_value"`, same
`STEER_COMPONENT = 0`, same `PC1 + mean` reconstruction, same `[1:]` decoder slice.

**Output is written with the existing `save_textual_v2_directions`**, so the on-disk schema is
identical to the global cells and the same readers work. To do that, flatten the per-layer
components back to the global layout before the call:

```python
components_flat = torch.stack(
    [components[:, j, :].reshape(-1) for j in range(rank)], dim=0
)                                    # (rank, (L+1)*hidden)
pca_mean_flat = mean.reshape(-1)     # ((L+1)*hidden,)
```

`save_textual_v2_directions` reshapes each row back to `(n_layers_plus, hidden_dim)` under keys
`pc0` / `pc1`, so `components.npz["pc0"]` holds the per-layer PC1 stack — directly comparable to
the global cells' `pc0`, and readable by the existing
`compare_deployed_vs_shuffled_control.py::load_decoder_pc1_norms`.

**Metadata additions** on top of the fields the global cells carry:

- `"fit_locus": "perlayer"`, `"fit_locus_description": FIT_LOCUS_DESCRIPTION["perlayer"]`
- `"explained_variance_ratio_per_layer"`: list of `L+1` lists of length `rank`
- `"pc1_explained_variance_per_layer"`: list of `L+1` floats
- `"explained_variance_ratio"`, `"pc1_explained_variance"`, `"pc2_explained_variance"`: `null` —
  the scalar form is not defined for a per-layer fit, and any reader keyed on the old name
  should get `null` rather than a wrong number
- `"global_fit_source_dir"`: the parallel global cell path, relative to the project root
- `"global_fit_metadata_sha256"`, `"global_fit_directions_sha256"`: hashes of the parallel
  global cell's files at read time, so lineage is recoverable
- `"forwards_executed": 0`, `"act_cache_dir"`, `"act_cache_read_only": true`
- `"arm"`: `"deployed"` or `"shuffled_image_control"`
- control arm only: `"control_for_deployed_slug"`, `"control_type": "image_derangement"`,
  `"derangement_file"`, `"derangement_seed": 1234`, `"derangement_fixed_points": 0`

Retained unchanged in form: `pc1_layer_norms`, `pc2_layer_norms`, `direction_layer_norms`,
`mean_diff_layer_norms_mean_over_demos`, `ids_used`, `n_pairs`, `token_policy`, `diff_polarity`,
`sign_convention`, `selection_policy`, `seed`, `rank`, `steer_component`,
`steer_reconstruction`, `content_hash_sha256_16`, `partition_file`, `block_size`,
`model_short`, `slug`, `date`, `git_commit`.

### 2.3 New run script — `evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py`

CPU-only. Does not import `src.model` and does not construct a wrapper.

```
python evaluation/run_scripts/extract_demos850_perlayer_pca_directions.py \
  --models llava-1.5-7b-hf qwen2.5-vl-7b-instruct \
  --arms deployed control \
  --num_demos 50 100 200 500
```

Arguments: `--models` (model_short strings, default both), `--arms` (default both),
`--num_demos` (default `50 100 200 500`), `--demos_path`, `--partition_path`,
`--force_recompute`, `--out_root` (default `experiment_artifacts/`).

Expected decoder shape per model is read from the parallel global cell's `directions.npz`
rather than hardcoded, then cross-checked against `{"llava-1.5-7b-hf": (32, 4096),
"qwen2.5-vl-7b-instruct": (28, 3584)}`. Both must agree.

Refuses to write if the target directory already contains `directions.npz` unless
`--force_recompute` is passed. Prints, per cell: model, arm, N, n_pairs, output shape,
PC1 explained-variance range across layers, cache hits, forwards executed (must be 0).

Exit code 1 if any gating check in Section 1 fails.

### 2.4 New analysis script

`diagnostic_experiments/perlayer_pca_control/compare_perlayer_vs_global_pca_deployed_vs_control.py`

CPU-only numpy plus matplotlib. Loads the four direction objects per (model, N) — deployed and
control, per-layer and global — with `load_textual_v2_directions`, and their `components.npz`.
Reuses the per-layer cosine helper from
`diagnostic_experiments/perception_diag/control/geometric_comparison/compare_deployed_vs_shuffled_control.py::per_layer_cosine`
by import, so the cosine definition is shared with the existing demos_v2 curve.

```
python diagnostic_experiments/perlayer_pca_control/compare_perlayer_vs_global_pca_deployed_vs_control.py \
  --models llava-1.5-7b-hf qwen2.5-vl-7b-instruct \
  --num_demos 50 100 200 500
```

It re-runs the metadata cross-check the existing script performs: recomputed per-layer L2 norms
must agree with `metadata.json["direction_layer_norms"][1:]` to within 1e-4 max absolute, for
all 32 direction objects.

### 2.5 New verification script

`helper_scripts/verify_perlayer_pca_control_extraction.py`, `--mode record | verify`, as
specified in Section 1.

---

## Metrics

All are primitives read off the direction arrays. No composite is introduced.

1. **Per-layer cosine, deployed against shuffled-image control.** For decoder layer `l`,
   `dot(deployed[l], control[l]) / (||deployed[l]|| * ||control[l]||)`, in float64, with the
   denominator floored at 1e-12. One value per layer, per sample size, per model, per fit
   scheme. This is the primary measurement.
2. **Per-layer L2 norm of each direction.** `||deployed[l]||` and `||control[l]||`, per layer,
   sample size, model, and scheme. Secondary measurement.
3. **Per-layer PC1 norm and PC1 share of direction norm.** `||pc0[l+1]||` and its ratio to the
   direction norm at the same layer, per arm, sample size, model, and scheme. This is the
   quantity the spec's second assumption is stated in. Reported as a check, not a result.
4. **Per-layer PC1 explained-variance ratio.** For the per-layer scheme, `S[l,0]**2 / sum_j S[l,j]**2`
   per layer; for the global scheme, the single scalar `pc1_explained_variance` already in
   metadata. Distinct from metric 3 and reported separately, because the spec's assumption text
   uses the word "share" for metric 3.
5. **Per-layer cosine between PC1 and the fit mean**, per arm, and its sign; and whether the two
   arms' signs agree at each layer. `dot(pc0[l+1], pca_mean[l+1]) / (||pc0[l+1]|| * ||pca_mean[l+1]||)`,
   its sign in {-1, 0, +1}, and the boolean equality of the deployed and control signs. Per
   layer, sample size, model, and scheme. Diagnostic for Check 9, reported as a check, not a
   result. No correction is derived from it in this run.
6. **Layer-band summary of metric 1.** Median and mean of the per-layer cosine within each band
   named in the spec's prediction table, and only those bands: Qwen layers 0–10 and 18–27,
   LLaVA layers 0–10 and 18–31, inclusive. One row per (model, band, sample size, scheme).
   The prediction table gives a single number per band, so a within-band location statistic is
   needed to read against it; both median and mean are reported because the spec does not say
   which. Layers 11–17 are reported per-layer only, since the prediction table names no band
   covering them.

---

## Artifacts

**New directions (16 cells)**

```
experiment_artifacts/vti/{model_short}/textual_v2_perlayer/
    demos850_ba05bd96_all_nd{N}_s42_r2_partition_perlayer/
        directions.npz          # layer_0..layer_{L-1}, num_layers, hidden_dim
        components.npz          # pc0, pc1, pca_mean_flat, n_layers_plus, hidden_dim, rank
        metadata.json

experiment_artifacts/vti/{model_short}/shuffled_control_demos850_perlayer/
    all_nd{N}_perlayer/
        directions.npz
        components.npz
        metadata.json
```

for `model_short` in {`llava-1.5-7b-hf`, `qwen2.5-vl-7b-instruct`} and N in {50, 100, 200, 500}.

Both parent directories are new and neither exists today. Together with the defaulted
`fit_locus` field in Section 2.1, these are the two independent shadowing guards the spec
requires: a code path that reuses the old slug builder produces a string without the `_perlayer`
suffix, and even so it would resolve under `textual_v2/`, not `textual_v2_perlayer/`.

Note for the record: `.gitignore` line 145 (`textual_v2*`) also matches `textual_v2_perlayer`,
so the new deployed cells are git-ignored exactly as the existing ones are. The
`shuffled_control_demos850_perlayer/` tree will be untracked, matching its
`shuffled_control_demos850/` sibling. Neither affects reachability on disk; the sha256 manifests
are the record.

**Verification**

```
diagnostic_experiments/perlayer_pca_control/
    global_fit_directions_sha256_manifest_pre_run.json
    global_fit_directions_sha256_manifest_post_run.json
    global_fit_directions_byte_identity_report.md
    activation_cache_untouched_report.md
    perlayer_pca_extraction_manifest_2026-07-29.json
```

`global_fit_directions_byte_identity_report.md` states, per file, whether the pre-run and
post-run SHA-256 match, and lists every per-layer output path together with the fact that it
did not exist at manifest-record time.

**Analysis outputs** (all under `diagnostic_experiments/perlayer_pca_control/`)

```
cosine_and_norms_deployed_vs_control_by_layer_{model_short}.csv
```
One row per (fit scheme, sample size, decoder layer). Columns:
`fit_scheme`, `num_demos`, `layer`, `cosine_deployed_vs_control`,
`deployed_direction_l2`, `control_direction_l2`,
`deployed_pc1_l2`, `control_pc1_l2`,
`deployed_pc1_share_of_direction_norm`, `control_pc1_share_of_direction_norm`,
`deployed_pc1_explained_variance`, `control_pc1_explained_variance`,
`deployed_cosine_pc1_with_mean`, `control_cosine_pc1_with_mean`,
`deployed_pc1_sign`, `control_pc1_sign`, `pc1_sign_agrees_between_arms`.

The three sign columns are Check 9. `deployed_pc1_sign` and `control_pc1_sign` are
`sign(cos(PC1_l, mean_l))` in that arm, as integers in {-1, 0, +1}; `pc1_sign_agrees_between_arms`
is their equality as a boolean. Emitted for both fit schemes.

```
cosine_by_layer_band_and_sample_size_{model_short}.csv
```
One row per (fit scheme, sample size, layer band). Columns: `fit_scheme`, `num_demos`,
`layer_band` (`"0-10"`, `"18-27"` or `"18-31"`), `median_cosine`, `mean_cosine`,
`min_cosine`, `max_cosine`, `n_layers_in_band`.

```
cosine_deployed_vs_control_by_layer_per_layer_and_global_pca_{model_short}.png
```
2x2 facet figure, one panel per sample size. Two lines per panel: "Per-layer PCA fit" and
"Global PCA fit". X axis "Decoder layer index", Y axis "Cosine similarity, deployed vs
shuffled-image control", fixed to [-1.05, 1.05], horizontal reference line at 0. Panel titles
"50 demo pairs", "100 demo pairs", "200 demo pairs", "500 demo pairs". Figure title
"LLaVA-1.5-7B: deployed direction vs shuffled-image control, per-layer PCA fit and global PCA
fit" (and the Qwen2.5-VL-7B equivalent).

```
direction_l2_norm_by_layer_deployed_and_control_per_layer_and_global_pca_{model_short}.png
```
Same 2x2 facet layout. Four lines per panel: deployed and control, per-layer and global fit.
X axis "Decoder layer index", Y axis "Direction L2 norm". Figure title
"LLaVA-1.5-7B: per-layer direction magnitude, deployed and shuffled-image control, under both
PCA fit schemes".

```
pc1_share_of_direction_norm_by_layer_per_layer_and_global_pca_{model_short}.png
```
2x2 facet layout, four lines per panel (deployed and control, both schemes). Y axis
"PC1 L2 norm divided by direction L2 norm". This is the figure for Check 8.

```
pc1_sign_agreement_by_layer_per_layer_and_global_pca_{model_short}.png
```
2x2 facet layout, one panel per sample size. This is the figure for Check 9. Each panel shows,
against "Decoder layer index" on the x axis, three per-layer series drawn as step or marker
traces on a discrete y axis with ticks at -1, 0, +1: deployed arm sign, control arm sign, both
under the per-layer fit. Layers where `pc1_sign_agrees_between_arms` is false are marked
distinctly — a shaded vertical band or filled marker — because the readable signal is whether
disagreements are isolated at single layers or contiguous across a band, and those two patterns
mean different things for the cosine curve. The global-fit signs are drawn as flat reference
lines in the same panel, one per arm, since the global scheme fixes one sign for the whole
flattened PC1. Figure title "LLaVA-1.5-7B: PC1 orientation per layer, deployed and
shuffled-image control, per-layer PCA fit against global PCA fit" (and the Qwen2.5-VL-7B
equivalent).

No correction is applied and no verdict is stated on this figure. It exists so the sign pattern
is visible alongside the cosine curve rather than buried in a CSV column.

```
perlayer_vs_global_pca_control_summary.md
```
Facts only, no interpretation: input paths and their SHA-256; shapes; the pass/fail table for
Checks 1–7; the reported values for Checks 8 and 9; the full per-layer cosine tables for all 16
(model, sample size, scheme) combinations; the band summary table; and the list of artifacts
written. Written in the style of
`diagnostic_experiments/perception_diag/control/geometric_comparison/geometric_comparison_summary.md`.

**Baseline note.** The global-scheme curves in these outputs are computed here for the first
time on demos850. The only cosine curve already in the repo,
`diagnostic_experiments/perception_diag/control/geometric_comparison/geometric_comparison_summary.md`,
is demos_v2 `all` at nd200 with the deployed slug `demosv2_9a44f4af_all_nd200_s42_r2_prefix` and
control `shuffled_control/all_nd200` — a different item set and a different derangement file.
It is not read, not modified, and not comparable row-for-row. The existing script is imported
for its cosine helper only; its output directory is not written to.

---

## Cost

**Forward passes: zero.** Every `(demo_id, variant)` pair needed by both arms is already cached:
5100 files under `textual_v2/_act_cache` and 1700 under `shuffled_control_demos850/_act_cache`,
per model, all verified present. The extraction loads stacks with `ActivationCache.load`, which
raises on a miss rather than forwarding. No model is loaded, so the device_map-sharding hazard
does not apply and no RunAI job is needed.

**Compute:** CPU only, on lambdab2. No GPU is required; at the time of planning GPU 0 was free
and GPUs 1–3 were at roughly 48 GB of 49 GB used, which does not constrain this run.

**Peak memory:** worst case is LLaVA at nd500. Diffs `(33, 500, 4096)` float32 = 270 MB; the two
loaded stack tensors another 540 MB; `PCA.fit` holds a centered copy (270 MB) plus
`U (33,500,500)` = 33 MB and `Vh (33,500,4096)` = 270 MB; the explained-variance SVD adds a
transient of the same order. Peak under about 2.5 GB, plus LAPACK workspace. The machine
reported 481 GB available. Non-issue.

**Wall clock:** dominated by decompressing 1700 `.npz` files per (model, arm) — 6800 file reads
total. Estimate under 15 minutes for the full 16-cell extraction, plus a couple of minutes for
the analysis script.

**Disk:** existing nd500 cells occupy about 2.1 MB (LLaVA) and 1.7 MB (Qwen) each. Sixteen new
cells is roughly 30 MB, plus about 3 MB of plots, CSVs, and manifests. 742 GB free on the
device holding the repo.

---

## Order of operations

1. Run `helper_scripts/verify_perlayer_pca_control_extraction.py --mode record`. Checks 1, 3, 4,
   5 run here, before any new code writes anything. Stop on any failure.
2. Land the Section 2.1 slug edits; re-run Check 3.
3. Land Section 2.2 and run Check 2 as a standalone assertion on one cell per model — the mean
   identity is the cheapest test that the refit changes only the fit locus, and it costs one
   fit. Stop on failure.
4. Run `extract_demos850_perlayer_pca_directions.py` for all 16 cells. Checks 6 and 7 run inline.
5. Run `--mode verify`. Confirm every pre-run SHA-256 is unchanged and every `_act_cache`
   `(name, size, mtime)` triple is unchanged. **Stop here and report if anything moved** — do
   not compute a cosine against artifacts that may have been overwritten.
6. Run the analysis script. It refuses to start unless
   `global_fit_directions_byte_identity_report.md` exists and records all-match.

---

## What confirms or falsifies

The spec's prediction table, read at each of the four sample sizes:

| Condition | Explanation A predicts | Explanation B predicts |
|---|---|---|
| Qwen, layers 0–10 | cosine ≈ 0.99 | cosine ≈ 0.95 |
| Qwen, layers 18–27 | cosine ≈ 0.6 | cosine ≈ 0.2 |
| LLaVA, layers 0–10 | cosine ≈ 0.99 | cosine ≈ 0.95–0.99 |
| LLaVA, layers 18–31 | cosine ≈ 0.96 | cosine ≈ 0.3–0.5 |

The spec names the LLaVA late band as the cell doing the most work in separating A from B.

The spec's stated abandonment condition: LLaVA showing essentially no decrease in cosine as
layers progress under the per-layer scheme, or a decline matching what the global scheme
produces on the same cell.

Reading these against the numbers is Alex's. The scripts report the per-layer curves, the band
summaries, and the Check 8 and Check 9 diagnostics, and state nothing about what they mean.

---

## Open questions

1. **Per-layer sign independence, at high PC1 share.** Under the global scheme `svd_flip` fixes
   one sign for the whole flattened PC1; under the per-layer scheme it fixes a sign
   independently at each layer, in the deployed and control fits separately. Where the PC1
   share of direction norm is small the sum `PC1 + mean` is dominated by the mean, which has no
   sign ambiguity, and the cosine is stable. The spec's own assumption is that the refit pushes
   LLaVA's PC1 share to 20–72%, which is the regime where a per-layer sign disagreement between
   arms moves the primary measurement. The plan reports `cos(PC1_l, mean_l)` and its sign per
   layer and arm so the effect is visible, and applies no correction. This is the one place where
   the measurement could move for a reason unrelated to the fit locus, and it lands on the cell
   the spec says does the most work.

   **Resolved by Alex, 2026-07-29: report only, correct nothing.** The run proceeds with
   `svd_flip` as-is. Check 9 additionally reports per-layer sign agreement between arms and
   plots it, so that whether the effect fired at all — and whether any disagreement is isolated
   or banded — is visible next to the cosine curves. Three correction rules are on the table for
   a later decision, once the diagnostic and the curves are both on disk: leave as-is; align
   `PC1_l` to `mean_l` at every layer in both arms; or align the control's `PC1_l` to the
   deployed's. The third makes the cosine partly a product of the convention and should be
   treated as contaminated, not merely less preferred. **No follow-up run may adopt a correction
   rule without a new or amended design spec**, because changing the rule changes what the
   primary measurement is.

2. **Whether the plots should carry the existing PC1-fraction annotation.** The demos_v2
   comparison plot annotates layers at or above the 90th percentile of max PC1-share, with a
   caption saying the cosine sign is less trustworthy there. That threshold is not in this spec,
   so this plan reports the shares in the CSV and in their own figure but does not annotate the
   cosine plots. Carry the annotation over, or leave the cosine plots clean?

3. **Which within-band statistic to read.** The prediction table gives one number per band; the
   plan reports median, mean, min, and max within each band. If one of those is the intended
   readout, say which and the band CSV can lead with it.
