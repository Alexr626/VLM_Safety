#!/usr/bin/env python3
# Last updated: 2026-08-05
"""Insert / refresh the `Last updated:` stamp at the top of every harness file.

The stamp exists because the harness is iterated on in a chat web UI that has no git history
and no repo access: a file uploaded there has to say for itself when it last changed. See
WORKFLOW_MAP.md, "Last-updated stamps", for the convention this script implements.

Idempotent — re-running replaces an existing stamp rather than adding a second one.

Placement by file type:
  .md / .mdc   HTML comment, first line, or first line after the YAML frontmatter block
  .py / .sh    `# Last updated: ...` on the line after the shebang
  .json        `"_last_updated": "..."` key (Claude Code re-serialises settings.json on a
               permission prompt and moves the key to the end; that is expected)

Date per file = the later of (last git commit touching it) and (its mtime), so the stamp
records the last content change rather than the moment this script ran. `--date` overrides
that for the files that actually changed in the edit you are stamping.

Usage:
  python3 helper_scripts/stamp_harness_dates.py                 # stamp / refresh everything
  python3 helper_scripts/stamp_harness_dates.py --check         # report only, write nothing
  python3 helper_scripts/stamp_harness_dates.py --date 2026-08-05 WORKFLOW.md .claude/agents/tutor.md

With paths given, only those files are touched; they must be in HARNESS_FILES below or
discovered under .cursor/rules/. Add new harness files to that list — and to the FILES list in
export_agentic_workflow_harness.sh, which is maintained by hand for the same reason.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE_NOTE = " — harness template; delete this line in your copy"

# Keep in sync with the FILES list in export_agentic_workflow_harness.sh. Deliberately absent:
# ABSTRACT.md, RESEARCH_LOG.md, readme.md, STEERING_MATH_REFERENCE.md (project content that
# dates its own entries) and IMPLEMENTATION.md (older form of the same convention on line 5,
# maintained under .cursor/rules/research_workflow.mdc).
HARNESS_FILES = [
    # Shared workflow docs
    "CLAUDE.md",
    "WORKFLOW.md",
    "TOOLING.md",
    "WORKFLOW_MAP.md",
    "answers/README.md",

    # Claude Code harness
    ".claude/settings.json",
    ".claude/agents/examiner.md",
    ".claude/agents/tutor.md",
    ".claude/agents/planner.md",
    ".claude/hooks/require_design_spec.py",
    ".claude/hooks/verify_harness.sh",
    ".claude/commands/examine-results.md",
    ".claude/commands/examine-design.md",
    ".claude/commands/examine-extraction.md",
    ".claude/commands/plan.md",
    ".claude/commands/tutor.md",

    # Spec / reading templates
    "templates/abstract_template.md",
    "templates/bypass_log_starter.md",
    "templates/design_template.md",
    "templates/extraction_template.md",
    "templates/reading_template.md",

    # Harness tooling
    "helper_scripts/export_agentic_workflow_harness.sh",
    "helper_scripts/stamp_harness_dates.py",
]

MD_STAMP = re.compile(r"^<!--\s*Last updated:.*-->\s*$")
HASH_STAMP = re.compile(r"^#\s*Last updated:.*$")
JSON_STAMP = re.compile(r'^\s*"_last_updated":\s*"([^"]*)",?\s*$', re.M)


def harness_files():
    files = list(HARNESS_FILES)
    rules = os.path.join(ROOT, ".cursor", "rules")
    if os.path.isdir(rules):
        for name in sorted(os.listdir(rules)):
            if name.endswith((".mdc", ".md")):
                files.append(os.path.join(".cursor", "rules", name))
    seen, unique = set(), []
    for rel in files:
        if rel not in seen:
            seen.add(rel)
            unique.append(rel)
    return unique


def content_date(rel):
    """Later of the file's last commit date and its mtime."""
    committed = subprocess.run(
        ["git", "-C", ROOT, "log", "-1", "--format=%cs", "--", rel],
        capture_output=True, text=True,
    ).stdout.strip()
    dates = [datetime.fromtimestamp(os.path.getmtime(os.path.join(ROOT, rel))).date()]
    if committed:
        dates.append(date.fromisoformat(committed))
    return max(dates).isoformat()


def current_stamp(rel, text):
    if rel.endswith(".json"):
        m = JSON_STAMP.search(text)
        return m.group(1) if m else None
    pat = MD_STAMP if rel.endswith((".md", ".mdc")) else HASH_STAMP
    for line in text.split("\n")[:12]:
        if pat.match(line):
            m = re.search(r"Last updated:\s*(\d{4}-\d{2}-\d{2})", line)
            return m.group(1) if m else "unparsed"
    return None


def stamp_md(lines, stamp_line):
    lines = [l for i, l in enumerate(lines) if not (i < 12 and MD_STAMP.match(l))]
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                insert = i + 1
                break
        else:
            raise ValueError("unterminated frontmatter")
        block = [stamp_line]
        if insert < len(lines) and lines[insert].strip() != "":
            block.append("")
        if lines[insert - 1].strip() == "---":
            block.insert(0, "")
        return lines[:insert] + block + lines[insert:]
    block = [stamp_line]
    if lines and lines[0].strip() != "":
        block.append("")
    return block + lines


def stamp_hash(lines, stamp_line):
    lines = [l for i, l in enumerate(lines) if not (i < 4 and HASH_STAMP.match(l))]
    insert = 1 if lines and lines[0].startswith("#!") else 0
    return lines[:insert] + [stamp_line] + lines[insert:]


def stamp_json(text, when):
    json.loads(text)  # fail loudly before rewriting a live settings file
    stripped = JSON_STAMP.sub("", text, count=1)
    stripped = re.sub(r",(\s*\n\s*)\}", r"\1}", stripped, count=0)  # no trailing comma if last
    new = re.sub(r"^\{\n", '{\n  "_last_updated": "%s",\n' % when, stripped, count=1)
    if new == stripped:
        raise ValueError("could not place the stamp: file does not open with '{'")
    json.loads(new)
    return new


def render(rel, text, when):
    if rel.endswith(".json"):
        return stamp_json(text, when)
    note = TEMPLATE_NOTE if rel.startswith("templates/") else ""
    lines = text.split("\n")
    if rel.endswith((".md", ".mdc")):
        return "\n".join(stamp_md(lines, "<!-- Last updated: %s%s -->" % (when, note)))
    if rel.endswith((".py", ".sh")):
        return "\n".join(stamp_hash(lines, "# Last updated: %s" % when))
    raise ValueError("no stamp rule for %s" % rel)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("paths", nargs="*", help="repo-relative harness files; default all of them")
    ap.add_argument("--check", action="store_true", help="report stamps, write nothing")
    ap.add_argument("--date", help="force this date (YYYY-MM-DD) instead of the derived one")
    args = ap.parse_args()

    if args.date:
        date.fromisoformat(args.date)

    known = harness_files()
    if args.paths:
        targets = []
        for p in args.paths:
            rel = os.path.relpath(os.path.abspath(p), ROOT) if os.path.isabs(p) else p.lstrip("./")
            if rel not in known:
                sys.exit("not a harness file (add it to HARNESS_FILES first): %s" % rel)
            targets.append(rel)
    else:
        targets = known

    written, unchanged, missing = [], [], []
    for rel in targets:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        had = current_stamp(rel, text)
        when = args.date or (had if (args.check and had) else content_date(rel))

        if args.check:
            print("%-52s %s" % (rel, had or "NO STAMP"))
            continue

        new_text = render(rel, text, when)
        if new_text == text:
            unchanged.append((rel, when))
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        written.append((rel, when, had))

    if args.check:
        if missing:
            print("\nmissing: %s" % ", ".join(missing), file=sys.stderr)
        return

    for rel, when, had in written:
        print("%-52s %s%s" % (rel, when, "" if had is None else "  (was %s)" % had))
    for rel, when in unchanged:
        print("%-52s %s  (unchanged)" % (rel, when))
    print("\nstamped %d, unchanged %d, missing %d" % (len(written), len(unchanged), len(missing)))
    if missing:
        print("missing: %s" % ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
