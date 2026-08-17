# Runpod agent setup (2026-08-16)

Followed `https://docs.runpod.io/agent-setup.md` for Cursor, global scope (Alex’s choice).

## What was run

1. `npx -y skills add runpod/runpod-plugins-official --skill '*' --yes --global`
   - Installed 6 skills under `~/.agents/skills/{runpod,runpod-mcp,runpodctl,flash,runpod-usage,companion-clis}`
   - `npx skills list -g` lists all six for Cursor
   - PromptScript global-install failures are unrelated (not this agent)
2. `npx @runpod/mcp-server@latest add` is interactive (`@clack/prompts`); not used (agent-setup forbids blocking prompts).
3. Wrote hosted MCP config by hand (same JSON the Runpod README gives for Cursor): `~/.cursor/mcp.json` → `{"mcpServers":{"runpod":{"url":"https://mcp.getrunpod.io/"}}}`
4. Symlinked those six skills into `~/.cursor/skills/` so Cursor’s personal-skill path resolves them.

Did **not** install `runpodctl`, Flash, or an API key (agent-setup: later, on demand).

## Repo files updated

- `config/sites.md` `cloud-h100`: vendor Runpod; checkout/`HF_HOME` still stub
- `IMPLEMENTATION.md`: Last updated + `### cloud-h100 (Runpod)` subsection

## Not verified

MCP OAuth. Listing Pods is the end-to-end check after Alex signs in (empty list is a pass).
