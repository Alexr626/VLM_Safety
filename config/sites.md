# Compute sites

Machine and path facts for this hallucination checkout and the sibling safety
checkout. Not a result log. Fill stub rows when that host is in use.

Canonical layout (personal workstation, 2026-08-13):

- Hallucination: `/home/alex/dev/vlm_hallucination` (branch `VLM_hallucination_mitigation`)
- Safety: `/home/alex/dev/VLM_Safety` (branch `main`)
- Shared COCO: `/home/alex/data/coco` (each checkout’s `data/coco` is a symlink)
- Hugging Face weights: default cache (`HF_HOME` unset)

Path remaps: `helper_scripts/runai/remap_lambdab2_paths.py`. Tracked JSON/JSONL
should store repo-relative paths (`data/coco/val2014/...`). Gitignored manifests
may keep a checkout-absolute prefix and are remapped per site (do not commit).

---

## personal-workstation

| Fact | Value |
|------|-------|
| Role | Current canonical git checkouts; Cursor + Claude Code |
| OS | Pop!_OS (Linux) |
| GPU | 1× RTX 5080; check `nvidia-smi`; `CUDA_VISIBLE_DEVICES` |
| Hallucination checkout | `/home/alex/dev/vlm_hallucination` |
| Safety checkout | `/home/alex/dev/VLM_Safety` |
| Shared COCO | `/home/alex/data/coco` |
| Hallucination conda | `vlm_hallucination_mitigation` |
| Safety conda | `vlm_safety` |
| `HF_HOME` | unset (Hugging Face default cache), shared by both envs |
| Typical N | Small exploratory runs on the 5080; large grids → `cloud-h100` when that site exists |
| Laptop | SSH client only (not a working tree) |

## lambdab2

| Fact | Value |
|------|-------|
| Role | Nokia lab server (when access returns) |
| GPU | 4× RTX A6000 48GB Ampere, shared; keep one slot free for iteration |
| Hallucination checkout | `/home/romanus/dev/vlm_hallucination` (intended; confirm on return) |
| Safety checkout | `/home/romanus/dev/VLM_Safety` (intended; confirm on return) |
| Shared COCO | Prefer `/data/romanus/coco` or one checkout + symlink; confirm on return |
| Conda | Per-checkout: `vlm_hallucination_mitigation` / `vlm_safety` |
| `HF_HOME` | `/data/romanus/huggingface` (historical; confirm on return) |
| `WANDB_DIR` | `/data/romanus/wandb` (historical) |
| Host (historical) | Tailscale `100.71.123.214` |

**[stub]** Re-verify GPU inventory, disk, and Tailscale address on return. Do not treat this row as live until then.

## runai-nfs

| Fact | Value |
|------|-------|
| Role | Bell Labs H100 cluster + persistent NFS (hallucination grids) |
| Submit from | lambdab2 `runai` CLI; project `nlm-mh`; pool `h100-pool` |
| GPU | 1× H100 80GB HBM3 per job |
| Image (historical) | `blsr-docker-virtual.artifactory-fpark1.int.net.nokia.com/llm_image14:0.1` |
| NFS extract (historical) | `/home/datalake/romanus/vlm_hallucination` |
| Python on NFS | micromamba `envs/vlm_hal` |
| Helpers | `helper_scripts/runai/` |

**[stub]** Confirm quota, image tag, and NFS paths on return. Safety jobs on RunAI only if added later.

## cloud-h100

| Fact | Value |
|------|-------|
| Role | Rented H100 (or similar) for large hallucination grids while off Nokia compute |
| Vendor | Runpod |
| Agent tooling (2026-08-16) | Cursor: hosted MCP `https://mcp.getrunpod.io/` in repo `.cursor/mcp.json` and `~/.cursor/mcp.json`. Skills in `~/.agents/skills/` (symlinked into `~/.cursor/skills/`). OAuth is per machine, not in git. `runpodctl` / Flash / API keys not installed yet. |
| Checkout / data | **[stub]** not a live GPU site until a Pod + volume exist |
| `HF_HOME` | **[stub]** |

W&B is logging, not a GPU site. Do not conflate the two.
