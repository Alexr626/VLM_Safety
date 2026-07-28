#!/usr/bin/env python3
"""PreToolUse hook: no plan lands in implementation_plans/ without a spec behind it.

Two spec types are accepted, declared within the first 10 lines of the outgoing content:

    design_spec: designs/<exp_id>_design.md          experiments: comparisons, measurements
    extraction_spec: extractions/<ext_id>_extraction.md   primitives: activations, directions

Exactly one must be present. The referenced file must exist and must not be an unfilled
template. Exit 0 to allow, exit 2 to block (stderr is shown to the model).

Wire up in .claude/settings.json with an absolute path, or a cd by any agent breaks every
write in the repo:

  "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/require_design_spec.py\""

Verify with .claude/hooks/verify_harness.sh after any change.
"""

import json
import os
import re
import sys

PLAN_DIR = "implementation_plans"

SPEC_KINDS = {
    "design_spec": "designs/",
    "extraction_spec": "extractions/",
}
DECL = re.compile(r"^\s*(design_spec|extraction_spec):\s*(\S+)\s*$")

MIN_SPEC_CHARS = 400

# Angle-bracket placeholders as they appear in the templates: <answer>, <prediction>,
# <condition 1>, <comparison - and which artifacts above support it>.
PLACEHOLDER = re.compile(r"<[a-z][a-z0-9 ,._/|+-]{2,70}>")
MAX_PLACEHOLDERS = 2


def fail(msg):
    sys.stderr.write(msg.rstrip() + "\n")
    sys.exit(2)


def main():
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # malformed event: never block real work on a parse failure

    if event.get("tool_name", "") not in ("Write", "Edit", "MultiEdit"):
        sys.exit(0)

    ti = event.get("tool_input", {}) or {}
    path = ti.get("file_path") or ti.get("path") or ""
    if PLAN_DIR not in path.replace("\\", "/"):
        sys.exit(0)

    content = ti.get("content") or ti.get("new_string") or ""

    # Edits to an existing plan inherit its declaration.
    if not content and os.path.exists(path):
        sys.exit(0)

    found = []
    for line in content.splitlines()[:10]:
        m = DECL.match(line)
        if m:
            found.append((m.group(1), m.group(2)))

    if not found:
        fail(
            "BLOCKED: no spec declared.\n"
            "Every file under implementation_plans/ must carry, within its first 10 lines,\n"
            "exactly one of:\n"
            "    design_spec: designs/<exp_id>_design.md\n"
            "        for experiments - any comparison, or any number read as evidence\n"
            "    extraction_spec: extractions/<ext_id>_extraction.md\n"
            "        for producing primitives - activations, attention, directions, caches\n"
            "If neither spec exists, it has not been written yet. Stop and return questions\n"
            "to Alex rather than drafting one."
        )

    if len(found) > 1:
        kinds = ", ".join(sorted({k for k, _ in found}))
        fail(
            f"BLOCKED: {len(found)} spec declarations found ({kinds}).\n"
            "A plan is one or the other. Producing primitives and measuring something are\n"
            "separate plans with separate gates; splitting them is the point."
        )

    kind, declared = found[0]
    expected_dir = SPEC_KINDS[kind]
    if not declared.replace("\\", "/").startswith(expected_dir):
        fail(
            f"BLOCKED: {kind} must point inside {expected_dir}, got: {declared}"
        )

    if not os.path.exists(declared):
        fail(
            f"BLOCKED: declared {kind} does not exist: {declared}\n"
            "The plan cannot precede the spec. Do not create the spec yourself."
        )

    try:
        with open(declared, "r", encoding="utf-8") as fh:
            spec = fh.read()
    except OSError as exc:
        fail(f"BLOCKED: cannot read declared {kind} {declared}: {exc}")

    if len(spec.strip()) < MIN_SPEC_CHARS:
        fail(
            f"BLOCKED: {kind} {declared} is a stub "
            f"({len(spec.strip())} chars, minimum {MIN_SPEC_CHARS}).\n"
            "An unfilled template is not a spec."
        )

    placeholders = PLACEHOLDER.findall(spec)
    if len(placeholders) > MAX_PLACEHOLDERS:
        shown = ", ".join(sorted(set(placeholders))[:5])
        fail(
            f"BLOCKED: {kind} {declared} still contains {len(placeholders)} unfilled "
            f"template placeholders ({shown}...).\n"
            "An unfilled template is not a spec."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
