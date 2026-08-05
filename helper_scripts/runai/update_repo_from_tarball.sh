#!/usr/bin/env bash
# Lightweight post-extract checks — delegates to verify_steering_nfs_layout.sh.
set -eu
BASE="${BASE:-/home/datalake/romanus}"
REPO="${REPO:-$BASE/vlm_hallucination}"
export BASE REPO
bash "$REPO/helper_scripts/runai/verify_steering_nfs_layout.sh"
echo "update_repo_from_tarball checks OK"
