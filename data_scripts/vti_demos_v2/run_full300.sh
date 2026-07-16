#!/usr/bin/env bash
# Compatibility wrapper: use the parameterized v2.1 driver with a 300-row pool.
N_CANDIDATES="${N_CANDIDATES:-300}" exec "$(dirname "$0")/run_full.sh"
