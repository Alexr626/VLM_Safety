# How to put the three shuffled-control commits on `activation_steering`

Date: 2026-07-28

## Situation

`new_research_workflow` and `activation_steering` share the same tip parent
`f1aaf63` ("Fixed flip plots"). On top of that, `new_research_workflow` has four
commits; only the first three should also live on `activation_steering`:

| SHA | Keep on activation_steering? | Message |
|---|---|---|
| `0c14e14` | yes | Ignore shuffled activation caches… |
| `64d9b4c` | yes | Add shuffled-image control direction extraction… |
| `e925d1b` | yes | Record deployed-vs-shuffled-control geometric comparison… |
| `2c3dd46` | no | Add agent research workflow docs… |

Cherry-pick is the right tool: it copies those commits onto another branch
without moving or rewriting `new_research_workflow`.

## Commands

You currently have uncommitted edits on `new_research_workflow`. Stash (or
commit) them before switching branches.

```bash
# from new_research_workflow, with a dirty tree:
git stash push -u -m "wip research workflow"

git checkout activation_steering
git cherry-pick 0c14e14 64d9b4c e925d1b

# verify: tip should be the geometric-comparison commit, not the workflow one
git log --oneline -5

git checkout new_research_workflow
git stash pop
```

Equivalent range form (same three commits, inclusive):

```bash
git cherry-pick 0c14e14^..e925d1b
```

Because both branches forked from the same commit, this should apply cleanly.

## What this does / does not do

- **Does:** append the three commits onto `activation_steering` (new SHAs if
  authorship/dates rewrite, but usually identical patches).
- **Does not:** remove anything from `new_research_workflow`. That branch still
  has all four commits, including the research-workflow one — which is what you
  want.
- **Does not:** push. Push `activation_steering` only when you intend to.

## If you instead wanted activation_steering to be the only home for those three

You would still cherry-pick onto `activation_steering`, then optionally rebuild
`new_research_workflow` as `activation_steering` + only the workflow commit.
That is a rewrite of `new_research_workflow` and is unnecessary for the ask
"add these to the other branch as well."
