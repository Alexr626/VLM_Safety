# Hallucination checkout open checklist (2026-08-16)

Factual record of what was on disk and what this session ran, from `hallucination_and_safety_checkout_state.md` plus the harness files named there.

## Layout (verified)

| Item | Value |
|---|---|
| Path | `/home/alex/dev/vlm_hallucination` |
| Branch | `VLM_hallucination_mitigation` |
| HEAD | `7034999` (ahead of `origin/VLM_hallucination_mitigation` by 1, not pushed) |
| Conda env on disk | `vlm_hallucination_mitigation` at `/home/alex/miniconda3/envs/vlm_hallucination_mitigation` |
| Interactive terminal | already `conda activate vlm_hallucination_mitigation` |
| GPU | NVIDIA GeForce RTX 5080, 941 MiB / 16303 MiB used at check time |
| Shared COCO | `/home/alex/data/coco`; both checkouts’ `data/coco` are symlinks; `val2014` has 40504 files |
| Sibling safety tree | `/home/alex/dev/VLM_Safety` (not edited) |

GitHub SSH from this host: `Permission denied (publickey)`. Fetch/push still blocked.

## Listed steps

### 1. `data/coco/.gitkeep` deletion — not committed

Working tree still shows `deleted: data/coco/.gitkeep`. HEAD stores `data/coco` as a **directory tree**; the working tree replaced that directory with a **symlink** to `/home/alex/data/coco`. Git therefore reports the tracked `.gitkeep` as deleted even though `/home/alex/data/coco/.gitkeep` exists on the shared volume.

Restore is the wrong action: recreating a tracked directory at `data/coco` would replace the symlink. The deletion is the change that matches the layout. Left uncommitted (commit only on explicit request). `ml-vlsu/` left untracked.

### 2. Review of `7034999` path remaps — facts only

Tracked JSON/JSONL added lines use repo-relative paths (`data/coco/val2014/...`, `data/amber/images/...`). No `/home/alex/...` in those added JSON/JSONL lines. Removed lines were `/home/romanus/dev/vlm_hallucination_mitigation_summer_2026/...`.

`git grep` on current tracked `*.json` / `*.jsonl` found no `/home/alex`. Remaining `/home/romanus` hits: `.claude/settings.json` (allow-list paths), plus three diagnostic JSON files under `diagnostic_experiments/`.

Push still waiting on GitHub SSH.

### 3. Harness — hook suite passed; Cursor write-deny did not

From this repo root:

```bash
bash .claude/hooks/verify_harness.sh
```

Output: `passed=14 failed=0` / `gate behaving correctly`.

The script does not check Claude Code permission denies. This Cursor session **did** write probe files `analysis/__harness_write_probe.md` and `designs/__harness_write_probe.md`; both writes succeeded. Both files were deleted immediately. `.claude/settings.json` still lists `Write(analysis/**)` and `Write(designs/**)` under `deny`; Cursor does not load that file (`TOOLING.md`). Claude Code refusal of those writes is still unverified in this window.

### 4. Cloud H100 / Nokia remap

Not done (listed as later).

## What this window can do from here

Implementation work in this tree, using conda `vlm_hallucination_mitigation`, site `personal-workstation` (`config/sites.md`). Do not edit `/home/alex/dev/VLM_Safety`. Do not treat lambdab2 / RunAI rows as live. Do not push until GitHub SSH works.
