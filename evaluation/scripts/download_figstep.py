#!/usr/bin/env python3
"""
Clone the FigStep dataset from GitHub to data/figstep/.

Uses a shallow git clone (only the tip of main). FigStep distributes its
images under data/images/SafeBench/ and questions under
data/question/SafeBench.csv.
"""

import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/ThuCCSLab/FigStep.git"


def main() -> None:
    dest = Path("data/figstep")
    if dest.exists():
        print(f"FigStep already present at {dest}.")
        return
    print(f"Cloning {REPO_URL} -> {dest} ...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
            check=True,
        )
    except FileNotFoundError:
        print("git is not installed.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"git clone failed: {e}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
