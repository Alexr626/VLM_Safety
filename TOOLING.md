<!-- Last updated: 2026-07-28 -->

# Tooling — which agent runs where, and what happens after the internship

Two clients, one repo. Cursor is free until the internship ends and is the more comfortable
implementation environment. Claude Code is billed personally and is the long-term home. The
split below is chosen so that the migration is a deletion rather than a rebuild.

## The split

| Role | Client | Why |
|---|---|---|
| Implementation | **Cursor** | Highest token volume by an order of magnitude — writing code, debugging, reading across the repo, iterating on runs. This is the role worth spending free tokens on, and it is the one you prefer. No gate depends on it. |
| Examiner | **Claude Code** | Low volume; a conversation, not a codebase crawl. |
| Planner | **Claude Code** | The pre-write hook is the only deterministic gate in the system and it exists nowhere else. Non-negotiable. |
| Tutor | **Either** | Ungated by construction. Its one boundary is instruction-only in both clients, so nothing is lost by running it in Cursor for free tokens. |

The reasoning that matters: **the hook does not port.** Cursor has no PreToolUse equivalent —
its rules are prompt-level instructions that a model can be argued out of, the same class of
enforcement as an agent definition. Every other part of this workflow is a markdown file and
runs anywhere. So the planner stays where the hook is, and everything else goes wherever it is
cheapest.

That is not a compromise forced by billing. Implementation is genuinely the role with no gate
on it and the highest burn, and examination is genuinely cheap. The split you would choose on
the merits and the split your budget suggests happen to agree.

## What is portable and what is not

Portable — plain files in the repo, read by any client:

```
CLAUDE.md  WORKFLOW.md  TOOLING.md  ABSTRACT.md
designs/  analysis/  answers/  implementation_plans/  templates/
```

Claude Code only:

```
.claude/settings.json          permissions and hook wiring
.claude/hooks/                 the design-spec gate
.claude/agents/                examiner, tutor, planner
.claude/commands/              /examine-results, /examine-design, /plan
```

Cursor only:

```
.cursor/rules/*.mdc
```

Keep both directories present and in version control now. Nothing needs deleting at the end of
the internship; you stop opening Cursor and the Claude Code side is already there and already
exercised. The failure mode to avoid is putting something load-bearing into Cursor rules that
has no `.claude/` counterpart, and discovering it in September.

## Cursor rules

Cursor reads `.cursor/rules/*.mdc` with frontmatter carrying `description`, optional `globs`,
and `alwaysApply`. Two changes to what you have now.

**Split `providing_answers.md` in two.** It currently bundles a persistence convention with a
plan-review posture under one `alwaysApply: true`. They are unrelated, they fire in different
situations, and bundling them means the plan-review rules are in context during every trivial
lookup while the persistence rule is in context during every plan review.

`.cursor/rules/persist_answers.mdc` keeps the directory convention. Fix three things while
splitting: the format string reads `answers/<topic>/<month>_<day_<year>` with an unclosed
bracket, one sentence says `.answers/` with a leading dot, and the described layout omits the
topic level that your actual tree has. Agents will follow the text literally and produce an
inconsistent tree. Point it at `answers/README.md` rather than restating the convention, so
there is one source of truth.

Also scope it. "Save every answer to a direct question" is right for an implementation agent
doing lookups and wrong for anything that produces judgement — add the line that
interpretations of results and recommendations about what to run next are never persisted, and
that `answers/interpretations/` is off limits.

`.cursor/rules/plan_pushback.mdc` keeps the second half unchanged. It is good and it is a real
second line of defence: the planner writes the plan, the implementation agent reads it cold and
flags scope compression, coded artifact names, and sophistication that outruns the evidence. Set
`alwaysApply: false` with a description like "applies when Alex pastes an experiment plan to
implement" so it loads on the situation rather than on every turn.

One addition to that rule now that plans carry provenance: check that the plan declares
`design_spec: designs/<exp_id>_design.md` and that the file exists. A plan without one bypassed
the gate and should be refused, not implemented.

**Do not port the examiner into Cursor.** A rules file cannot enforce a gate, and an examiner
that can be talked into interpreting is worse than no examiner, because it looks like the
harness is running when it is not.

## Rendering

This is why the scratchpad matters more than it first appears. Terminal panes do not render
LaTeX, and both clients stream to a terminal. Any answer with mathematics in it is unreadable
where it is produced.

The convention, now written into `tutor.md`: full derivation to
`answers/concepts/<date>/<name>.md`, plain-notation summary to the terminal with the path
appended. Open the file in the editor preview. Cursor renders markdown and LaTeX in preview;
so does any editor you use later.

Apply the same split to the implementation agent for anything with formulas in it — your own
`attention_knockout` notes already do this and read well because of it.

## Checks before you rely on any of this

1. Hook fires: ask a Claude Code session to write into `implementation_plans/` with no
   `design_spec:` line. It must be blocked.
2. Scratchpad writable: ask any agent to write a trivial note under `answers/`. It must
   succeed without a permission prompt.
3. Interpretations walled: ask an agent to write to `answers/interpretations/`. It must be
   denied.
4. Alex-only files walled: ask an agent to write to `analysis/` or `designs/`. It must be
   denied.

If 3 or 4 succeed, the belief/artifact line is unenforced and the rest is decoration.
