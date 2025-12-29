#!/usr/bin/env bash
set -euo pipefail

MIR_PATH=${1:-}
if [[ -z "${MIR_PATH}" ]]; then
  echo "usage: $0 <file.mir>" >&2
  exit 2
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TOOLS_DIR="${ROOT_DIR}/tools/spill_fuzz"

LLC=${SPILL_FUZZ_LLC:-${LLC:-llc}}
LD_LLD=${SPILL_FUZZ_LLD:-${LD_LLD:-ld.lld}}
LLVM_READOBJ=${SPILL_FUZZ_LLVM_READOBJ:-${LLVM_READOBJ:-llvm-readobj}}
MCPU=${SPILL_FUZZ_MCPU:-gfx90a}
BUFFER_SIZE=${SPILL_FUZZ_BUFFER_SIZE:-4096}
KERNEL_NAME=${SPILL_FUZZ_KERNEL:-}
GPU_STRICT=${SPILL_FUZZ_GPU_STRICT:-0}

HIP_RUNNER="${TOOLS_DIR}/hip_runner"
META_PARSER="${TOOLS_DIR}/parse_metadata.py"

if [[ ! -x "${HIP_RUNNER}" ]]; then
  "${TOOLS_DIR}/build_hip_runner.sh"
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/spill_fuzz_gpu.XXXXXX")
trap 'rm -rf "${WORK_DIR}"' EXIT

REF_MIR="${WORK_DIR}/ref.mir"
TEST_MIR="${WORK_DIR}/test.mir"
REF_OBJ="${WORK_DIR}/ref.o"
TEST_OBJ="${WORK_DIR}/test.o"
REF_HSACO="${WORK_DIR}/ref.hsaco"
TEST_HSACO="${WORK_DIR}/test.hsaco"
SPEC="${WORK_DIR}/kernel.spec"

python3 - <<'PY' "${MIR_PATH}" "${TEST_MIR}"
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
Path(sys.argv[2]).write_text(src, encoding="utf-8")
PY

python3 - <<'PY' "${MIR_PATH}" "${REF_MIR}"
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
if "--- |" not in src:
    Path(sys.argv[2]).write_text(src, encoding="utf-8")
    sys.exit(0)

pre, rest = src.split("--- |", 1)
ir, post = rest.split("...", 1)

def rewrite(line: str) -> str:
    if line.startswith("define "):
        if "amdgpu-num-vgpr" in line or "amdgpu-num-sgpr" in line:
            line = re.sub(r'"amdgpu-num-vgpr"="\\d+"', '"amdgpu-num-vgpr"="256"', line)
            line = re.sub(r'"amdgpu-num-sgpr"="\\d+"', '"amdgpu-num-sgpr"="256"', line)
        else:
            insert = ' "amdgpu-num-vgpr"="256" "amdgpu-num-sgpr"="256"'
            if "{" in line:
                line = line.replace("{", insert + " {")
            else:
                line = line + insert
    return line

ir_lines = [rewrite(line) for line in ir.splitlines()]
out = pre + "--- |" + "\n".join(ir_lines) + "..." + post
Path(sys.argv[2]).write_text(out, encoding="utf-8")
PY

if ! ${LLC} -mtriple=amdgcn-amd-amdhsa -mcpu="${MCPU}" -filetype=obj -o "${REF_OBJ}" "${REF_MIR}"; then
  if [[ "${GPU_STRICT}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

if ! ${LLC} -mtriple=amdgcn-amd-amdhsa -mcpu="${MCPU}" -filetype=obj -o "${TEST_OBJ}" "${TEST_MIR}"; then
  if [[ "${GPU_STRICT}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

if ! ${LD_LLD} -shared -o "${REF_HSACO}" "${REF_OBJ}"; then
  if [[ "${GPU_STRICT}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

if ! ${LD_LLD} -shared -o "${TEST_HSACO}" "${TEST_OBJ}"; then
  if [[ "${GPU_STRICT}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

KERNEL_ARG=()
if [[ -n "${KERNEL_NAME}" ]]; then
  KERNEL_ARG=(--kernel "${KERNEL_NAME}")
fi

if ! "${META_PARSER}" --llvm-readobj "${LLVM_READOBJ}" "${REF_HSACO}" --out "${SPEC}" "${KERNEL_ARG[@]}"; then
  status=$?
  if [[ ${status} -eq 3 ]]; then
    exit 0
  fi
  exit ${status}
fi

"${HIP_RUNNER}" \
  --hsaco-a "${REF_HSACO}" \
  --hsaco-b "${TEST_HSACO}" \
  --spec "${SPEC}" \
  --buffer-size "${BUFFER_SIZE}"
