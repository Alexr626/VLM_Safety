# Viewing concatenated changes since a base commit

## Question

How to view a “concatenated” / net version of all changes since the `Demo generation pipeline changes` commit, to decide how to squash the unpushed commits.

## Answer

Yes. Git range diffs collapse intermediate commits into the net tree difference:

```bash
# File list + line counts (best overview)
git diff --stat 1c85002..HEAD

# Short summary
git diff --shortstat 1c85002..HEAD

# Adds / modifies / deletes per path
git diff --name-status 1c85002..HEAD

# Full patch if you want to read the net diff
git diff 1c85002..HEAD
```

`1c85002` is `Demo generation pipeline changes`. The range `A..B` is “what’s in B that isn’t in A” as a single combined diff — adds then deletes in later commits cancel out.

### State on 2026-07-20

- Branch `activation_steering` is **9 commits ahead** of `origin/activation_steering` (nothing from this range is pushed yet).
- Commits in range: `4804a97` … `5d60d3d` (diagnostic scripts → deletion cleanup).
- **Net effect of all 9 commits:** **33 files changed, +3350 / −32**
  - **21 added**, **12 modified**, **0 deleted** in the final tree vs `1c85002`
- Intermediate work that was later deleted largely **cancels out** of this aggregated view; what remains is mostly the surviving `perception_diag` / data / small eval+src tweaks.

### Squash options (once you decide)

Because these 9 commits are **unpushed**, soft reset + recommit is safe and simple:

```bash
# Keep working tree; rewrite history of the 9 commits into one (or stage selectively into two)
git reset --soft 1c85002
# then commit once, or unstage and make 2 commits
```

Or interactive rebase (`git rebase -i 1c85002`) to squash/fixup. Do **not** force-push unless you intentionally rewrite something already on the remote (these aren’t).
