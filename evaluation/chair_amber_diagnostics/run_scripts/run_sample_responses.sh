#!/usr/bin/env bash
# Thin wrapper — delegates to helper_scripts/run_sample_responses.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$ROOT/helper_scripts/run_sample_responses.sh"
