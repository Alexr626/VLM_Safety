#!/usr/bin/env bash
# Export the Claude Code / Cloud Agentic research-workflow harness into one directory.
#
# Copies agents, slash commands, hooks, settings, workflow docs, templates, and the
# Cursor-side rules that belong to the same loop — everything listed in WORKFLOW.md
# Install, plus newer harness files that Install has not caught up to yet.
#
# Does not copy live experiment content (designs/, extractions specs, analysis readings,
# implementation_plans/, experiment_artifacts/, ABSTRACT.md drafts).
# Does not copy .claude/settings.local.json (machine-local).
#
# Usage:
#   bash helper_scripts/export_agentic_workflow_harness.sh
#   bash helper_scripts/export_agentic_workflow_harness.sh --out /path/to/dir
#   bash helper_scripts/export_agentic_workflow_harness.sh --flat
#   bash helper_scripts/export_agentic_workflow_harness.sh --zip
#
# --flat   write every file at the destination root with __ path separators
#          (easier to upload to a chat web UI that only accepts a file list)
# --zip    also write <out>.zip beside the destination directory
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y_%m_%d)"
OUT="${ROOT}/exports/agentic_workflow_harness_${STAMP}"
FLAT=0
ZIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="${2:?--out requires a path}"
      shift 2
      ;;
    --flat)
      FLAT=1
      shift
      ;;
    --zip)
      ZIP=1
      shift
      ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Relative paths from repo root. Keep in sync with WORKFLOW.md Install + TOOLING.md.
FILES=(
  # Shared workflow docs
  CLAUDE.md
  WORKFLOW.md
  TOOLING.md
  WORKFLOW_MAP.md
  STEERING_MATH_REFERENCE.md
  answers/README.md

  # Claude Code harness
  .claude/settings.json
  .claude/agents/examiner.md
  .claude/agents/tutor.md
  .claude/agents/planner.md
  .claude/hooks/require_design_spec.py
  .claude/hooks/verify_harness.sh
  .claude/commands/examine-results.md
  .claude/commands/examine-design.md
  .claude/commands/examine-extraction.md
  .claude/commands/plan.md
  .claude/commands/tutor.md

  # Spec / reading templates (copy-from; do not edit in place in the live repo)
  templates/abstract_template.md
  templates/bypass_log_starter.md
  templates/design_template.md
  templates/extraction_template.md
  templates/reading_template.md

  # Cursor side of the same loop (see TOOLING.md)
  .cursor/rules/providing_answers.mdc
  .cursor/rules/research_workflow.mdc
)

copy_one() {
  local rel="$1"
  local src="${ROOT}/${rel}"
  if [[ ! -e "$src" ]]; then
    echo "MISSING (skipped): ${rel}" >&2
    return 1
  fi
  if [[ "$FLAT" -eq 1 ]]; then
    local dest_name
    dest_name="$(printf '%s' "$rel" | sed 's#^\./##; s#/#__#g')"
    mkdir -p "$OUT"
    cp -a "$src" "${OUT}/${dest_name}"
    printf '%s\n' "$dest_name"
  else
    local dest="${OUT}/${rel}"
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
    # Preserve executable bits already set on hooks; force +x on hook scripts.
    if [[ "$rel" == .claude/hooks/* ]]; then
      chmod +x "$dest"
    fi
    printf '%s\n' "$rel"
  fi
  return 0
}

rm -rf "$OUT"
mkdir -p "$OUT"

copied=()
missing=()
for rel in "${FILES[@]}"; do
  if path="$(copy_one "$rel")"; then
    copied+=("$path")
  else
    missing+=("$rel")
  fi
done

MANIFEST="${OUT}/MANIFEST.md"
{
  cat <<EOF
# Agentic research-workflow harness export

Exported from \`${ROOT}\` on $(date -Is).

This directory is a self-contained copy of the Claude Code / Cloud Agentic research
workflow harness so you can iterate on it with other chat web UI agents. It is the
install set from \`WORKFLOW.md\`, plus the tutor slash command and Cursor rules that
belong to the same loop (\`TOOLING.md\`).

## What is included

EOF
  for p in "${copied[@]}"; do
    printf -- '- `%s`\n' "$p"
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo
    echo "## Missing at export time (skipped)"
    echo
    for p in "${missing[@]}"; do
      printf -- '- `%s`\n' "$p"
    done
  fi

  cat <<'EOF'

## What is deliberately excluded

- Live Alex-authored content: `designs/`, `extractions/`, `analysis/`, `ABSTRACT.md`
- Planner output: `implementation_plans/`
- Machine-local Claude settings: `.claude/settings.local.json`
- Experiment code, data, and artifacts

## How to use with a chat web UI

1. Upload this whole folder, or attach the files listed above.
2. Start from `WORKFLOW.md` (the loop) and `CLAUDE.md` (shared context every role inherits).
3. Role prompts live under `.claude/agents/` (or the flat `__` names if `--flat` was used).
4. Slash-command wrappers live under `.claude/commands/`.
5. Spec shapes live under `templates/` — copy them; do not treat the templates as filled specs.
6. The deterministic gate is `.claude/hooks/require_design_spec.py`; exercise it with
   `.claude/hooks/verify_harness.sh` when you are back in a Claude Code checkout.

## Re-install into a repo root

From this export (tree mode, not `--flat`):

```bash
cp -a CLAUDE.md WORKFLOW.md TOOLING.md WORKFLOW_MAP.md STEERING_MATH_REFERENCE.md /path/to/repo/
mkdir -p /path/to/repo/{.claude,.cursor/rules,templates,answers}
cp -a .claude/. /path/to/repo/.claude/
cp -a .cursor/rules/. /path/to/repo/.cursor/rules/
cp -a templates/. /path/to/repo/templates/
cp -a answers/README.md /path/to/repo/answers/
chmod +x /path/to/repo/.claude/hooks/*.py /path/to/repo/.claude/hooks/*.sh
bash /path/to/repo/.claude/hooks/verify_harness.sh
```

EOF
} >"$MANIFEST"

echo "Wrote ${#copied[@]} files (+ MANIFEST.md) -> ${OUT}"
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Warning: ${#missing[@]} listed path(s) were missing:" >&2
  printf '  %s\n' "${missing[@]}" >&2
fi

if [[ "$ZIP" -eq 1 ]]; then
  zip_path="${OUT}.zip"
  rm -f "$zip_path"
  (
    cd "$(dirname "$OUT")"
    zip -qr "$(basename "$zip_path")" "$(basename "$OUT")"
  )
  echo "Wrote ${zip_path}"
fi
