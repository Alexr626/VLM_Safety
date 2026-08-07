# Handoff Q&A: packs contents, GitHub key, activation_steering close

Date: 2026-08-07.

## What’s already in the artifact packs

**Pack A directions** (`packA_directions_no_act_cache.tar.gz`, 276M) was built from all of `experiment_artifacts/vti/` with only `_act_cache/` directories excluded. It **does** include:

- `textual_v2/` direction dirs (meandiff + r2 partition, demosv2, etc.)
- `textual_v2_perlayer/`
- `shuffled_control/` (direction dirs + reports; not act caches)
- `shuffled_control_demos850/` (same)
- `shuffled_control_demos850_perlayer/`
- `visual/`
- legacy `textual_directions_nd70_*.npz`
- derangement JSONs / manifests at the `vti/` root

**Pack A act caches** (`packA_act_caches_textual_v2.tar.gz`, 2.9G) only has `textual_v2/_act_cache/` for llava / qwen2.5 / qwen2 — **not** shuffled-control act caches.

**Not packed yet (worth optional Drive upload for low-compute restore):**

| Missing | Approx size | Why it matters |
|---------|-------------|----------------|
| `shuffled_control/_act_cache/` (llava+qwen2.5) | ~215M | Re-fit shuffled control without forwards |
| `shuffled_control_demos850/_act_cache/` | ~911M | Same for demos850 shuffled |
| `evaluation/results/2026-07-13/` | ~2.4G | Largest other eval tree |
| Other eval dates (06-18, 06-22, 07-02, …) | ~100M combined | Historical cells |
| `data/amber/dumps/`, `data/pope/dumps/` | ~8.9G | Steered-capture dumps; expensive to regenerate |

Pack B (`packB_full_data.tar.gz`, 48G) already covers raw benchmarks + dumps under `data/`.

## GitHub push from lambdab2

Remote `github` → `git@github.com:Alexr626/VLM_Safety.git` is already configured. Failure was `Permission denied (publickey)`.

Lambdab2 public key to add under GitHub → Settings → SSH and GPG keys → New SSH key:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFUg5yCYE3FmgNM+DpuWGSV30U5gVSWat+zyzjNncsKV romanus@lambdab2
```

Then:

```bash
ssh -T git@github.com   # expect: Hi Alexr626!
cd /home/romanus/dev/vlm_hallucination_mitigation_summer_2026
git fetch github
git push -u github new_research_workflow:VLM_hallucination_mitigation
```

No force unless fetch shows divergent commits you intentionally overwrite. Offline fallback remains `exports/new_research_workflow_for_github.bundle`.

## Closing `activation_steering`

**Do not treat GitLab `VLM_hallucination_mitigation` as the merged tip.** That remote branch is still at `840ad24` (early “Repurpose repo…”). Current work is on:

- `origin/new_research_workflow` (`c8cbaf7`) — full tree
- `origin/manager_handoff` (`99c6cef`) — filtered manager tree (built from `activation_steering`)

`activation_steering` tip `21b0dfa` is an **ancestor of `manager_handoff`** (safe). It is **not** a git ancestor of `new_research_workflow` (3 commits on a divergent line), but every unique *file blob* from those three commits is already **SAME** on `new_research_workflow` except `.gitignore` (DIFF). Closing/archiving `activation_steering` after pointing people at `manager_handoff` / `new_research_workflow` is content-safe; optionally leave the branch as a label rather than deleting.

## Follow-ups worth deciding

1. Build optional packs: shuffled act caches (~1.1G) + remaining eval results (esp. 2026-07-13)?
2. Prefer updating GitLab default / `VLM_hallucination_mitigation` to track `manager_handoff` or `new_research_workflow` tip vs leaving three parallel branches?
3. Archive `diagnostic_experiments/*/results` and large HTML galleries separately?
4. After Drive upload, paste folder URL into `ARTIFACT_ARCHIVE.md` and commit on both NRW + manager_handoff.
