# Clarify act-cache packs and GitLab single working branch

Date: 2026-08-07.

## Act caches

`packA_directions_no_act_cache.tar.gz` intentionally **omits** every `_act_cache/` directory. Directions / metadata / components / visual / perlayer / shuffled **direction dirs** are inside it; forward-pass activation stacks are not.

Act caches are in **separate** tarballs that must also be uploaded:

| Pack | What’s inside |
|------|----------------|
| `packA_act_caches_textual_v2.tar.gz` (2.9G) | `…/textual_v2/_act_cache/` for llava, qwen2.5, qwen2 — demos850 / demos_v2 last-token stacks used for PCA + meandiff |
| `packC_shuffled_act_caches.tar.gz` (1.1G) | `…/shuffled_control/_act_cache/` and `…/shuffled_control_demos850/_act_cache/` |

Upload priority in `ARTIFACT_ARCHIVE.md`: items 4 and 5. Restoring only the directions pack is not enough to avoid re-running forwards.

## GitLab “one working branch”

Earlier “update default / VLM_hallucination_mitigation” meant: the remote branch named `VLM_hallucination_mitigation` is still an old tip (`840ad24`), which is confusing if people think that’s current.

What Alex wants: close `activation_steering`, leave **`manager_handoff`** as the single manager-facing working branch on GitLab. That is fine: `activation_steering` is a full ancestor of `manager_handoff`. Optional: delete or protect `activation_steering`; optionally rename or retarget GitLab default branch to `manager_handoff`.

Note: `new_research_workflow` may still exist on GitLab as a full-tree backup (includes answers/harness). If the goal is literally one remote branch total, decide whether to keep NRW private-only (GitHub) and delete it from GitLab, or leave it as a non-default backup.
