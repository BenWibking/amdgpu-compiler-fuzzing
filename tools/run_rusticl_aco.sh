#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_SRC_DEFAULT="${ROOT_DIR}/kernels/pele/pc_cmpflx_opencl.cl"
ENTRY_DEFAULT="pc_cmpflx_launch"
BUILD_OPTS_DEFAULT="-cl-std=CL1.2"

KERNEL_SRC="${1:-$KERNEL_SRC_DEFAULT}"
ENTRY="${2:-$ENTRY_DEFAULT}"
BUILD_OPTS="${3:-$BUILD_OPTS_DEFAULT}"

if [[ ! -f "${KERNEL_SRC}" ]]; then
  echo "error: kernel source not found: ${KERNEL_SRC}" >&2
  exit 1
fi

# Rusticl requires explicit driver enablement.
if [[ -z "${RUSTICL_ENABLE:-}" ]]; then
  export RUSTICL_ENABLE="radeonsi"
fi

# Default kernel uses fp64; enable unless the user overrides.
if [[ -z "${RUSTICL_FEATURES:-}" ]]; then
  export RUSTICL_FEATURES="fp64"
fi

# Build OpenCL compile-only harness if needed.
if [[ ! -x "${ROOT_DIR}/tools/ocl_aco_compile/ocl_aco_compile" ]]; then
  make -C "${ROOT_DIR}/tools/ocl_aco_compile"
fi

"${ROOT_DIR}/tools/ocl_aco_compile/ocl_aco_compile" \
  "${KERNEL_SRC}" "${ENTRY}" "${BUILD_OPTS}"
