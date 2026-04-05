#!/usr/bin/env python3
"""
Add captions to the behavioral_ground_truth holisafe_responses.json file.

For each row in holisafe_responses.json, look up the corresponding caption
from data/captions/holisafe.json (keyed by master id) and add it to the row.

A .bak backup is created before modifying the responses file.
"""

import json
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_PATH = _PROJECT_ROOT / "data" / "captions" / "holisafe.json"
RESPONSES_PATH = (_PROJECT_ROOT / "diagnostic_experiments" / "llava-1.5-7b-hf"
                  / "behavioral_ground_truth" / "outputs" / "results"
                  / "holisafe_responses.json")


def main():
    with open(CAPTIONS_PATH) as f:
        captions = json.load(f)
    with open(RESPONSES_PATH) as f:
        responses = json.load(f)

    print(f"Captions : {len(captions)}")
    print(f"Responses: {len(responses)}")

    # ── Join by master id ───────────────────────────────────────────────────
    missing = []
    for r in responses:
        caption = captions.get(str(r["id"]))
        if caption is None:
            missing.append(r["id"])
            r["caption"] = ""
        else:
            r["caption"] = caption

    matched = len(responses) - len(missing)
    print(f"\nMatched  : {matched}")
    print(f"Missing  : {len(missing)}")
    if missing:
        print(f"  First 10 missing ids: {missing[:10]}")

    # ── Backup and rewrite ──────────────────────────────────────────────────
    backup = RESPONSES_PATH.with_suffix(RESPONSES_PATH.suffix + ".bak_pre_caption")
    shutil.copy2(RESPONSES_PATH, backup)
    print(f"\nBackup: {backup.name}")

    with open(RESPONSES_PATH, "w") as f:
        json.dump(responses, f, indent=2)
    print(f"Rewrote: {RESPONSES_PATH}")


if __name__ == "__main__":
    main()