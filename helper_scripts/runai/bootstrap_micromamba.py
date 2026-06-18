#!/usr/bin/env python3
"""Download micromamba binary onto the NFS (no curl/wget required).

Called by helper_scripts/runai/setup_vlm.sh during the one-time RunAI setup job.
"""
import os
import sys
import tarfile
import urllib.request

base = sys.argv[1]
url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
tmp = "/tmp/mm.tar.bz2"
urllib.request.urlretrieve(url, tmp)
with tarfile.open(tmp, "r:bz2") as archive:
    archive.extract("bin/micromamba", path=base)
mm = os.path.join(base, "bin", "micromamba")
os.chmod(mm, 0o755)
print("micromamba ->", mm)
