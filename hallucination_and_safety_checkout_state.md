# State of both checkouts (2026-08-16)

Handoff from the **safety** Cursor window (`/home/alex/dev/VLM_Safety`) for a new agent opening the **hallucination** checkout (`/home/alex/dev/vlm_hallucination`). Facts from disk this session and from the 2026-08-13 setup work in this chat. GitHub fetch failed (`Permission denied (publickey)`); remote-tracking refs are whatever was last recorded locally.

Approved plan (do not edit it): `/home/alex/.cursor/plans/two-project_site_setup_4ac12eaa.plan.md`. All five plan todos are marked completed.

---

## Why two environments exist

Until 2026-08-13 this workstation had **one clone** at `/home/alex/dev/VLM_Safety` on branch `VLM_hallucination_mitigation`. That tree mixed hallucination mitigation (VTI / POPE / AMBER / CHAIR) with leftover safety datasets (HoliSafe, MSSBench, FigStep, MM-SafetyBench, SIUO, CatQA). One GitHub repo, two research lines.

The internship at Nokia ended. Hallucination work continues with the former manager on this personal workstation. Alex returns later to a **different** Nokia team; lambdab2 and RunAI access are expected then, not now. The setup therefore splits along **two axes**:

1. **Research line** — hallucination vs safety — two git checkouts, two conda envs, two Cursor windows.
2. **Compute site** — personal workstation (canonical now) vs Nokia (lambdab2 + RunAI later). Same commits; paths remapped per machine. Cloud H100 is a stub site until a vendor is chosen. W&B is logging, not a GPU site.

There is **one GitHub remote** (`git@github.com:Alexr626/VLM_Safety.git`). No Nokia GitLab requirement. No internal Nokia schema or customer data in either tree.

---

## What this chat did (2026-08-13)

1. Restored Drive artifact packs into the then-`VLM_Safety` / now-`vlm_hallucination` tree (Pack B and related; Pack E skipped as overlapping). Archives deleted after the restore checklist.
2. Renamed that tree to `/home/alex/dev/vlm_hallucination`. Cloned `main` to `/home/alex/dev/VLM_Safety` (SSH clone failed; local `git clone --branch main` then `remote set-url origin git@github.com:Alexr626/VLM_Safety.git`).
3. Moved safety `data/` trees (holisafe-bench, mssbench, figstep, mm-safetybench, siuo, catqa-contrastive) into the safety checkout. Relocated COCO to `/home/alex/data/coco`; both `data/coco` are symlinks.
4. Hallucination commit `7034999`: `config/sites.md`; remap script `--relative` applied to **tracked** JSON (romanus prefix → `data/coco/val2014/...`); `src/dataset.py` `_resolve_image_path`; `IMPLEMENTATION.md` compute section. Gitignored manifests remapped to `/home/alex/dev/vlm_hallucination/...` (427 files), logged in `RESEARCH_LOG.md`, **not** committed.
5. Copied the research-agent loop into safety (`8b796e0`), then retargeted those files so they do not name VTI/POPE/`BENCHMARK_REGISTRY` (`e5c2e2f`). Safety `.gitignore` hides the rsynced dataset trees and keeps SIUO SSS JSON (`f261a03`).

This Cursor window stayed attached to `/home/alex/dev/VLM_Safety`. Further hallucination edits belong in a **new window** on `/home/alex/dev/vlm_hallucination`. Do not write hallucination code into the safety tree, and do not copy hallucination `IMPLEMENTATION.md` / `RESEARCH_LOG.md` / `ABSTRACT.md` / `designs/` / `analysis/` into safety.

---

## Layout (locked 2026-08-13)

| | Hallucination | Safety |
|---|---|---|
| Path | `/home/alex/dev/vlm_hallucination` | `/home/alex/dev/VLM_Safety` |
| Branch | `VLM_hallucination_mitigation` | `main` |
| Conda | `vlm_hallucination_mitigation` | `vlm_safety` |
| HEAD | `7034999` (ahead of origin by 1, not pushed) | `f261a03` (ahead of recorded `origin/main` by 3, not pushed) |

- Shared COCO: `/home/alex/data/coco`
- `HF_HOME` unset (default Hugging Face cache), shared by both envs
- Site table in both trees: `config/sites.md` (`personal-workstation` filled; `lambdab2`, `runai-nfs`, `cloud-h100` stubs)

---

## Personal workstation vs Nokia (when Alex returns)

**Now (personal-workstation):** Pop!_OS, 1× RTX 5080, Cursor over SSH from the laptop. Canonical git checkouts are the two paths above. Small exploratory runs here; large grids wait for a cloud-H100 site or Nokia return.

**Later (Nokia):** same GitHub remote, two checkouts, two conda envs. Intended lambdab2 paths (confirm on return): `/home/romanus/dev/vlm_hallucination` and `/home/romanus/dev/VLM_Safety`. Shared COCO prefer `/data/romanus/coco`. Historical `HF_HOME=/data/romanus/huggingface`. RunAI remains the hallucination-grid site (submit from lambdab2; NFS copy of the **hallucination** checkout). Safety jobs on RunAI only if added later. Keep one A6000 free for iteration.

On return, remap **gitignored** files only:

```bash
python helper_scripts/runai/remap_lambdab2_paths.py \
  --old /home/alex/dev/vlm_hallucination \
  --new "$(pwd)" \
  --apply
```

If a gitignored file still has the old lambdab2 prefix, use `--old /home/romanus/dev/vlm_hallucination_mitigation_summer_2026`. Tracked files should already be repo-relative after `7034999` — do not re-apply `--relative` in a way that dirties them. Log the command in that checkout’s `RESEARCH_LOG.md`. Fill the stub rows in `config/sites.md` with live GPU/NFS facts.

Do not treat lambdab2 / RunAI rows as live until then.

---

## Hallucination checkout (your workspace)

- `data/`: amber, captions, chair, coco→symlink, hallusionbench, mmhal-bench, pope, vti. Safety trees were moved out.
- Unstaged: `D data/coco/.gitkeep` (directory became a symlink). Untracked: `ml-vlsu/` (~4M). Leave `ml-vlsu/` unless Alex says otherwise.
- Loader smokes (POPE / CHAIR / AMBER-1500 pin) were run 2026-08-13 after the remap.

**Still for Alex / this agent, not blockers for opening the tree:**

1. Commit or restore the `data/coco/.gitkeep` deletion.
2. Alex reviews `7034999` (tracked diffs must be romanus prefix → `data/coco/...`, not `/home/alex/...`), then push when GitHub SSH works.
3. `bash .claude/hooks/verify_harness.sh` from **this** repo root; then confirm writes to `analysis/` and `designs/` are refused (the script cannot check those).
4. Cloud H100 vendor: later. Nokia remap: later.

Activate conda `vlm_hallucination_mitigation` in this window.

---

## Safety checkout (the other window; do not edit from here)

- Working tree clean at `f261a03`.
- No `IMPLEMENTATION.md` (not copied; create there when safety code changes).
- Large dataset trees on disk, gitignored except `data/siuo/siuo_sss_claude.json` and `siuo_sss_editted.json`.
- Recorded `origin/main` is still PR #2 (`e78c37d`). Whether GitHub `main` has moved is unverified until `git fetch origin` works.
- Loop files were copied from hallucination and then retargeted (`REFERENCE_REGISTRY`, HoliSafe/MSSBench names, conda `vlm_safety`). Do not copy hallucination `IMPLEMENTATION.md` or result logs into that tree.
