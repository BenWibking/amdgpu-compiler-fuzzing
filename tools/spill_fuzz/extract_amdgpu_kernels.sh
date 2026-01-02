#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/spill_fuzz/extract_amdgpu_kernels.sh [options] <input.ll>

Options:
  -o, --out <dir>     Output directory (default: directory of input)
  -r, --rocm <path>   ROCm install prefix (default: auto-detect)
  -l, --list          List kernel names and exit
  -d, --demangle      Demangle kernel names for listing/output files
  -h, --help          Show this help

This extracts each amdgpu_kernel into its own .ll using opt internalize+GDC.
USAGE
}

out_dir=""
rocm_path=""
input_ll=""
list_only="false"
demangle="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out)
      out_dir="$2"
      shift 2
      ;;
    -r|--rocm)
      rocm_path="$2"
      shift 2
      ;;
    -l|--list)
      list_only="true"
      shift
      ;;
    -d|--demangle)
      demangle="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      input_ll="$1"
      shift
      ;;
  esac
done

if [[ -z "${input_ll}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${input_ll}" ]]; then
  echo "Input not found: ${input_ll}" >&2
  exit 1
fi

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

opt_candidates=(
  "${rocm_path}/lib/llvm/bin/opt"
  "${rocm_path}/llvm/bin/opt"
  /opt/rocm-*/lib/llvm/bin/opt
  /opt/rocm-*/llvm/bin/opt
  "opt"
)

opt_tool=""
for candidate in "${opt_candidates[@]}"; do
  if [[ -x "${candidate}" ]]; then
    opt_tool="${candidate}"
    break
  fi
done

if [[ -z "${opt_tool}" ]]; then
  echo "Could not find opt. Install ROCm LLVM tools or update --rocm." >&2
  exit 1
fi

if [[ -z "${out_dir}" ]]; then
  out_dir="$(dirname "${input_ll}")"
fi

mkdir -p "${out_dir}"

demangler_candidates=(
  "c++filt"
  "${rocm_path}/lib/llvm/bin/llvm-cxxfilt"
  "${rocm_path}/llvm/bin/llvm-cxxfilt"
  /opt/rocm-*/lib/llvm/bin/llvm-cxxfilt
  /opt/rocm-*/llvm/bin/llvm-cxxfilt
)

demangler=""
for candidate in "${demangler_candidates[@]}"; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    demangler="${candidate}"
    break
  fi
done

if [[ "${demangle}" == "true" && -z "${demangler}" ]]; then
  echo "Could not find c++filt or llvm-cxxfilt for demangling." >&2
  exit 1
fi

kernel_names="$(rg -n '^define .*amdgpu_kernel' "${input_ll}" | sed -E 's/.*@([^ (]+).*/\1/' | sort -u)"
if [[ -z "${kernel_names}" ]]; then
  echo "No amdgpu_kernel definitions found in ${input_ll}" >&2
  exit 1
fi

if [[ "${list_only}" == "true" ]]; then
  if [[ "${demangle}" == "true" ]]; then
    while IFS= read -r kernel; do
      demangled="$("${demangler}" "${kernel}")"
      echo "${kernel}: ${demangled}"
    done <<< "${kernel_names}"
  else
    echo "${kernel_names}"
  fi
  exit 0
fi

map_file=""
if [[ "${demangle}" == "true" ]]; then
  map_file="${out_dir}/kernel-map.txt"
  : > "${map_file}"
fi

while IFS= read -r kernel; do
  name_hash=""
  if [[ "${demangle}" == "true" ]]; then
    demangled="$("${demangler}" "${kernel}")"
    demangled_base="${demangled%%(*}"
    safe_kernel="$(echo "${demangled_base}" | sed -E 's/[^A-Za-z0-9_.-]/_/g')"
  else
    safe_kernel="$(echo "${kernel}" | sed -E 's/[^A-Za-z0-9_.-]/_/g')"
    demangled=""
  fi
  if [[ ${#safe_kernel} -gt 120 ]]; then
    if command -v sha1sum >/dev/null 2>&1; then
      name_hash="$(echo -n "${kernel}" | sha1sum | awk '{print $1}')"
    elif command -v md5sum >/dev/null 2>&1; then
      name_hash="$(echo -n "${kernel}" | md5sum | awk '{print $1}')"
    else
      name_hash="$(echo -n "${kernel}" | cksum | awk '{print $1}')"
    fi
    safe_kernel="${safe_kernel:0:100}-${name_hash}"
  fi
  out_ll="${out_dir}/kernel-${safe_kernel}.ll"
  if [[ -e "${out_ll}" ]]; then
    if [[ -z "${name_hash}" ]]; then
      if command -v sha1sum >/dev/null 2>&1; then
        name_hash="$(echo -n "${kernel}" | sha1sum | awk '{print $1}')"
      elif command -v md5sum >/dev/null 2>&1; then
        name_hash="$(echo -n "${kernel}" | md5sum | awk '{print $1}')"
      else
        name_hash="$(echo -n "${kernel}" | cksum | awk '{print $1}')"
      fi
    fi
    out_ll="${out_dir}/kernel-${safe_kernel}-${name_hash}.ll"
  fi
  "${opt_tool}" -S -passes="internalize,globaldce" \
    -internalize-public-api-list="${kernel}" \
    "${input_ll}" -o "${out_ll}"
  echo "Wrote ${out_ll}"
  if [[ -n "${map_file}" ]]; then
    echo "${kernel}\t${demangled}\t${out_ll}" >> "${map_file}"
  fi
done <<< "${kernel_names}"

if [[ -n "${map_file}" ]]; then
  echo "Wrote ${map_file}"
fi
