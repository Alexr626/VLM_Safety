#!/usr/bin/env python3
"""Run a shell script after normalizing Windows CRLF line endings."""
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
subprocess.run(["bash"], input=text, check=True)
