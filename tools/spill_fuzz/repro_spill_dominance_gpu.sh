#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MIR_PATH="${ROOT_DIR}/kernels/pele/spill_fuzz_out/kernel-pc_cmpflx_launch-a7d5b88a0dde5efe7b96045874e05330cd93917d.vgpr20.sgpr108.ll"
KERNEL_NAME="_Z16pc_cmpflx_launchiiiiiiiiiiPKdiiiiiiS0_iiiiiiPdiiiiiiS1_iiiiiiS0_iiiiiiS0_iiiiiiS1_iiiiiiS1_iiiiiiS0_iiiiiii"
INPUT_SPEC="${ROOT_DIR}/tools/spill_fuzz/input_specs/pc_cmpflx_launch.json"

if [[ ! -f "${MIR_PATH}" ]]; then
  echo "missing MIR input: ${MIR_PATH}" >&2
  exit 2
fi

export SPILL_FUZZ_MCPU=${SPILL_FUZZ_MCPU:-gfx942}
export SPILL_FUZZ_KERNEL=${SPILL_FUZZ_KERNEL:-${KERNEL_NAME}}
export SPILL_FUZZ_INPUT_SPEC=${SPILL_FUZZ_INPUT_SPEC:-${INPUT_SPEC}}

"${ROOT_DIR}/tools/spill_fuzz/run_on_gpu.sh" "${MIR_PATH}"
