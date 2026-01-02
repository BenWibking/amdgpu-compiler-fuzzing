#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/spill_fuzz/pele_hip_to_ll.sh [options]

Options:
  -t, --target <name>   Makefile target name (default: pelec_repro2_dodecane_lu)
  -k, --kernel <name>   Kernel function name to extract (optional)
  -r, --rocm <path>     ROCm install prefix (default: /opt/rocm)
  -h, --help            Show this help

This script builds LLVM bitcode via hipcc and converts it to .ll.
USAGE
}

target="pelec_repro2_dodecane_lu"
kernel=""
rocm_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--target)
      target="$2"
      shift 2
      ;;
    -k|--kernel)
      kernel="$2"
      shift 2
      ;;
    -r|--rocm)
      rocm_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

find_rocm_prefix() {
  local candidate

  if [[ -n "${rocm_path}" ]]; then
    echo "${rocm_path}"
    return 0
  fi

  if [[ -f "/opt/rocm/include/hip/hip_runtime.h" ]]; then
    echo "/opt/rocm"
    return 0
  fi

  for candidate in /opt/rocm-*; do
    if [[ -f "${candidate}/include/hip/hip_runtime.h" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

if ! rocm_path="$(find_rocm_prefix)"; then
  echo "Could not find ROCm install with hip headers. Use --rocm <path>." >&2
  exit 1
fi

export ROCM_PATH="${rocm_path}"

kernels_dir="kernels/pele"
obj_path="${kernels_dir}/${target}.o"
ll_path="${kernels_dir}/${target}.ll"

llvm_dis_candidates=(
  "${ROCM_PATH}/lib/llvm/bin/llvm-dis"
  "${ROCM_PATH}/llvm/bin/llvm-dis"
  /opt/rocm-*/lib/llvm/bin/llvm-dis
  /opt/rocm-*/llvm/bin/llvm-dis
  "llvm-dis"
)
llvm_extract_candidates=(
  "${ROCM_PATH}/lib/llvm/bin/llvm-extract"
  "${ROCM_PATH}/llvm/bin/llvm-extract"
  /opt/rocm-*/lib/llvm/bin/llvm-extract
  /opt/rocm-*/llvm/bin/llvm-extract
  "llvm-extract"
)

llvm_dis=""
for candidate in "${llvm_dis_candidates[@]}"; do
  if [[ -x "${candidate}" ]]; then
    llvm_dis="${candidate}"
    break
  fi
done

llvm_extract=""
for candidate in "${llvm_extract_candidates[@]}"; do
  if [[ -x "${candidate}" ]]; then
    llvm_extract="${candidate}"
    break
  fi
done

if [[ -z "${llvm_dis}" ]]; then
  echo "Could not find llvm-dis. Install LLVM or point --rocm to a full ROCm." >&2
  exit 1
fi

if [[ -z "${llvm_extract}" ]]; then
  llvm_extract="llvm-extract"
fi

make -C "${kernels_dir}" "${target}.o" \
  FLAGS="--cuda-device-only -emit-llvm"

"${llvm_dis}" "${obj_path}" -o "${ll_path}"

echo "Wrote ${ll_path}"

if [[ -n "${kernel}" ]]; then
  out_kernel_ll="${kernels_dir}/${target}.${kernel}.ll"
  "${llvm_extract}" -func "${kernel}" "${ll_path}" -o "${out_kernel_ll}"
  echo "Wrote ${out_kernel_ll}"
fi
