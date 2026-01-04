#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_SRC_DEFAULT="${ROOT_DIR}/kernels/pele/pc_cmpflx_opencl.cl"
ENTRY_DEFAULT="pc_cmpflx_launch"
OUT_DEFAULT="${ROOT_DIR}/pc_cmpflx.spv"

KERNEL_SRC="${1:-$KERNEL_SRC_DEFAULT}"
ENTRY="${2:-$ENTRY_DEFAULT}"
OUT_SPV="${3:-$OUT_DEFAULT}"

if [[ ! -f "${KERNEL_SRC}" ]]; then
  echo "error: kernel source not found: ${KERNEL_SRC}" >&2
  exit 1
fi

if ! command -v clspv >/dev/null 2>&1; then
  echo "error: clspv not found in PATH" >&2
  exit 1
fi

# Compile OpenCL C to Vulkan-compatible SPIR-V (clspv requires CL1.2 here).
clspv --cl-std=CL1.2 --spv-version=1.6 --fp64 -inline-entry-points -O0 \
  -I "${ROOT_DIR}/kernels/pele" \
  -o "${OUT_SPV}" "${KERNEL_SRC}"

# Build Vulkan compile-only harness if needed.
if [[ ! -x "${ROOT_DIR}/tools/vk_aco_compile/vk_aco_compile" ]]; then
  make -C "${ROOT_DIR}/tools/vk_aco_compile"
fi

RADV_PERFTEST=aco \
RADV_DEBUG=shaders \
ACO_DEBUG=validate,info \
"${ROOT_DIR}/tools/vk_aco_compile/vk_aco_compile" "${OUT_SPV}" "${ENTRY}" 0 0
