#!/usr/bin/env python3
"""PreToolUse hook: no plan lands in implementation_plans/ without a spec behind it.

Two spec types are accepted, declared within the first 10 lines of the outgoing content:

    design_spec: designs/<path to spec>          experiments: comparisons, measurements
    extraction_spec: extractions/<path to spec>  primitives: activations, directions

The path is whatever the spec is actually called, repo-relative: any filename, at any depth
under its directory, so `designs/07_30_26/toward_yes_grid.md` and `designs/foo_design.md` are
both fine. Only the leading directory is constrained.

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
# <condition 1>, <name — source — n — purpose>, <primitive — output path — what it must not
# overwrite>. Any angle-bracket span on one line counts, so em-dashes, capitals, quotes, and
# digits do not smuggle an unfilled field past the gate. Bracketed URLs are exempt; a literal
# HTML tag would be flagged, which has not come up and errs toward blocking.
PLACEHOLDER = re.compile(r"<(?!https?://)[^<>\n]{3,90}>")
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

    # An Edit carries a fragment, not the whole file, so the declaration to validate is the
    # one already on disk. Read it and check that instead. This keeps the gate honest — a plan
    # whose spec was deleted or emptied still fails, and an edit that strips line 2 fails on
    # the next write — without demanding that every fragment repeat the declaration. Checking
    # the fragment was the old behaviour and it blocked every real edit to an existing plan.
    tool = event.get("tool_name", "")
    if tool in ("Edit", "MultiEdit") and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
        except OSError as exc:
            fail(f"BLOCKED: cannot read existing plan {path}: {exc}")
    elif not content and os.path.exists(path):
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
            "    design_spec: designs/<path to the spec, as it sits on disk>\n"
            "        for experiments - any comparison, or any number read as evidence\n"
            "    extraction_spec: extractions/<path to the spec, as it sits on disk>\n"
            "        for producing primitives - activations, attention, directions, caches\n"
            "Any filename, at any depth under that directory.\n"
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
