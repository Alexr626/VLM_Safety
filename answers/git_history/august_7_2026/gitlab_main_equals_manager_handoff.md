# GitLab: main == manager_handoff; collapse to one branch

Date: 2026-08-07.

## Done

- Pushed `origin/main` at `704663c` (= tip of former `manager_handoff`).
- Deleted remote: `activation_steering`, `manager_handoff`, `new_research_workflow`.
- Local checkout now on `main` tracking `origin/main`.

## Blocked on GitLab UI (no API token on lambdab2)

Cannot delete `VLM_hallucination_mitigation` while it is still the **default** branch (`The default branch of a project cannot be deleted`).

User action:

1. GitLab → project → **Settings → Repository → Default branch** → set to **`main`** → Save.
2. Then either delete `VLM_hallucination_mitigation` in the UI, or ask the agent to run:
   `git push origin --delete VLM_hallucination_mitigation`

Remaining remotes until that step: `origin/main`, `origin/VLM_hallucination_mitigation`.
