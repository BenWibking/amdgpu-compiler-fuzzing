#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLS_DIR="${ROOT_DIR}/tools/spill_fuzz"

HIPCC=${HIPCC:-hipcc}
OUT="${TOOLS_DIR}/hip_runner"

${HIPCC} -O2 -std=c++17 -o "${OUT}" "${TOOLS_DIR}/hip_runner.cpp"
echo "built ${OUT}"
