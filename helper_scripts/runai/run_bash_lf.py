#!/usr/bin/env python3
"""Run a shell script after normalizing Windows CRLF line endings.

Used by RunAI jobs to execute helper_scripts/runai/*.sh from the extracted
repo tarball without CRLF failures from WinSCP uploads.
"""
import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: run_bash_lf.py <script.sh>", file=sys.stderr)
    sys.exit(2)

path = Path(sys.argv[1])
text = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
subprocess.run(["bash"], input=text, check=True)
