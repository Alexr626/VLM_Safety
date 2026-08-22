# mcp.json location and push (2026-08-17)

The Runpod MCP entry from 2026-08-16 was written to **`/home/alex/.cursor/mcp.json`** (Cursor user-global), not the repo. That is why it did not appear in the project file tree.

Repo copy added 2026-08-17: **`.cursor/mcp.json`** (URL only, no API key). OAuth tokens stay on the machine.

Commit `bd39b44` includes that file plus `config/sites.md`, `IMPLEMENTATION.md`, answers from 2026-08-16, and `hallucination_and_safety_checkout_state.md`. Branch is 3 commits ahead of `origin/VLM_hallucination_mitigation`.

`git push` failed: `git@github.com: Permission denied (publickey)`. Same blocker as 2026-08-13. `ml-vlsu/` left untracked.

This session’s Runpod MCP server `user-runpod` is `ready` after Alex’s OAuth.
