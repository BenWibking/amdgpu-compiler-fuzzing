# AMDGPU Spill Fuzz Harness

This is a configuration fuzzer for AMDGPU MIR. It varies register limits and
pass settings, runs `llc`, and checks a non-GPU oracle (spill dominance) plus
optional `-verify-machineinstrs`.

## Quick start

```
./tools/spill_fuzz/spill_fuzz.py \
  --corpus extern/llvm-project/llvm/test/CodeGen/AMDGPU \
  --llc extern/llvm-project/build/bin/llc \
  --mcpu gfx90a \
  --passes greedy \
  --verify-machineinstrs \
  --iterations 50
```

## What it does

- Picks a random `.ll` file from the corpus and lowers it to MIR with the
  selected `llc` to keep formats in sync.
- Skips IR that targets non-HSA shader calling conventions.
- Skips IR that uses WMMA intrinsics unless `--mcpu` is gfx11/gfx12.
- Skips IR that uses OpenCL `printf` intrinsics.
- Skips IR that uses r600 or legacy FMA intrinsics.
- Skips IR containing `CODE_OBJECT_VERSION` metadata tokens.
- Skips IR with dynamic alloca.
- Skips IR that uses SMFMAC intrinsics unless `--mcpu` is gfx95.
- Skips IR that defines LDS/GDS globals when non-kernel functions are present.
- Skips IR that uses MFMA intrinsics unless `--mcpu` is gfx90/gfx94/gfx95.
- Skips IR that contains "invalid addrspacecast" diagnostics.
- Skips IR that uses `amdgpu_gfx` calling convention.
- Skips IR that uses `llvm.amdgcn.fdot2.*` unless `--mcpu` is gfx94/gfx95.
- Skips workgroup attribute error-check tests.
- Skips invalid `read_register` tests.
- Skips atomic fmax intrinsics unless `--mcpu` is gfx10/gfx11/gfx94/gfx95.
- Injects `"amdgpu-num-vgpr"`/`"amdgpu-num-sgpr"` into the IR.
- Verifies machine state after ISel with `llc -stop-after=finalize-isel`.
- Runs `llc -stop-after=<passes> -print-after=<passes>` and captures the machine
  dump for the spill-dominance oracle.
- Runs a spill-dominance oracle on `SI_SPILL_*_SAVE/RESTORE` pairs.
- Optionally calls a GPU runner.

## Non-GPU oracles

- `-verify-machineinstrs` from `llc` (use `--verify-machineinstrs`).
- Pre-pass machine verifier (`-stop-after=finalize-isel`) to catch invalid inputs.
- Spill-dominance check (always on). It ensures each restore of `%stack.N` has
  a dominating save of the same slot.

## GPU oracle hook (optional, HIP)

You can provide a command that takes the mutated MIR path and returns non-zero
on failure:

```
./tools/spill_fuzz/spill_fuzz.py \
  --corpus extern/llvm-project/llvm/test/CodeGen/AMDGPU \
  --llc extern/llvm-project/build/bin/llc \
  --mcpu gfx90a \
  --gpu-cmd "./tools/spill_fuzz/run_on_gpu.sh"
```

The HIP runner does differential execution between a high-register version and
the current fuzzed version. It compares outputs of all pointer arguments and
returns non-zero on mismatch.

The metadata parser only accepts kernels whose explicit arguments are limited to
`global_buffer`, `by_value`, and `value`. If no compatible kernel is found, the
GPU step is skipped for that input.

The runner is built automatically the first time `run_on_gpu.sh` is invoked. You
can also build it manually:

```
./tools/spill_fuzz/build_hip_runner.sh
```

Environment overrides:

```
SPILL_FUZZ_LLC=/path/to/llc
SPILL_FUZZ_LLD=/path/to/ld.lld
SPILL_FUZZ_LLVM_READOBJ=/path/to/llvm-readobj
SPILL_FUZZ_MCPU=gfx90a
SPILL_FUZZ_BUFFER_SIZE=4096
SPILL_FUZZ_KERNEL=my_kernel_name
SPILL_FUZZ_GPU_STRICT=1
SPILL_FUZZ_INPUT_SPEC=/path/to/input.json
HIPCC=/opt/rocm/bin/hipcc
```

Input spec (optional)

The GPU runner can consume a JSON file to set deterministic argument values,
buffer sizes, and launch dimensions. When set via `SPILL_FUZZ_INPUT_SPEC` or
`--input-spec`, the JSON is converted into a flat spec and fed to `hip_runner`.

Example JSON:

```json
{
  "seed": 12345,
  "launch": { "grid": [1, 1, 1], "block": [1, 1, 1] },
  "buffers": { "10": { "size_bytes": 65536 } },
  "values": {
    "0": 0,
    "5": { "int": 64 },
    "73": { "bytes": [0, 0, 0, 0] }
  }
}
```

Notes:
- `buffers`/`values` use argument indices from the kernel metadata order.
- `values` supports integer, `hex`, or explicit `bytes` entries.

## HIP kernel to LLVM IR helper

The `tools/spill_fuzz/pele_hip_to_ll.sh` wrapper builds a HIP reproducer using
`kernels/pele/Makefile`, emits device-only LLVM bitcode, and writes a `.ll`
next to the kernel source. Optional `--kernel` extraction keeps a single
function.

What it does:

- Builds the target with `--cuda-device-only -emit-llvm` so the output object is
  LLVM bitcode rather than an ELF.
- Converts the bitcode object to textual LLVM IR via `llvm-dis` (or `opt -S` if
  `llvm-dis` is unavailable).
- Optionally extracts a single kernel to a smaller `.ll` with `llvm-extract`.

How to use the resulting `.ll`:

- Feed it to `llc` for codegen/regalloc repros (set `-mtriple`/`-mcpu` to match
  the target).
- Prefer the single-kernel `.ll` for reduction and faster iteration.

Examples:

```
./tools/spill_fuzz/pele_hip_to_ll.sh -t pelec_repro2_dodecane_lu
./tools/spill_fuzz/pele_hip_to_ll.sh -t pelec_repro2_dodecane_lu -k my_kernel
```

Devcontainer one-liner:

```
devcontainer exec --workspace-folder . -- bash -lc "./tools/spill_fuzz/pele_hip_to_ll.sh -t pelec_repro2_dodecane_lu"
```

Output:

- `kernels/pele/pelec_repro2_dodecane_lu.ll` is device IR for `llc`.
- `kernels/pele/pelec_repro2_dodecane_lu.<kernel>.ll` is produced when using
  `--kernel`.

Example `llc` invocation:

```
/opt/rocm-6.4.4/lib/llvm/bin/llc \
  -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 \
  kernels/pele/pelec_repro2_dodecane_lu.ll -o /tmp/pele.s
```

Extract each kernel into its own `.ll` (no `llvm-extract` required):

```
./tools/spill_fuzz/extract_amdgpu_kernels.sh \
  kernels/pele/pelec_repro2_dodecane_lu.ll
```

This writes `kernel-<kernel_name>.ll` files in the same directory (non-filename
characters are replaced with `_`).

List available kernels without extracting:

```
./tools/spill_fuzz/extract_amdgpu_kernels.sh --list \
  kernels/pele/pelec_repro2_dodecane_lu.ll
```

Demangle kernel names (default; also used for output filenames when extracting):

```
./tools/spill_fuzz/extract_amdgpu_kernels.sh --demangle \
  kernels/pele/pelec_repro2_dodecane_lu.ll
```

When `--demangle` is used, the script writes a `kernel-map.txt` file that maps
mangled names to demangled names and output paths. The output filenames use the
demangled base name (without parameter lists). Collisions get a hash suffix, and
long names are truncated with a hash to avoid filename length limits.

Strip debug info from extracted kernels:

```
./tools/spill_fuzz/extract_amdgpu_kernels.sh --strip-debug \
  kernels/pele/pelec_repro2_dodecane_lu.ll
```

Example: Pele repro → fuzz the `pc_cmpflx_launch` kernel

```
./tools/spill_fuzz/pele_hip_to_ll.sh -t pelec_repro2_dodecane_lu
./tools/spill_fuzz/extract_amdgpu_kernels.sh --demangle --strip-debug \
  kernels/pele/pelec_repro2_dodecane_lu.ll
mkdir -p /tmp/pele-fuzz
cp kernels/pele/kernel-pc_cmpflx_launch*.ll /tmp/pele-fuzz/
./tools/spill_fuzz/spill_fuzz.py \
  --corpus /tmp/pele-fuzz \
  --llc /opt/rocm-6.4.4/lib/llvm/bin/llc \
  --mcpu gfx942 \
  --passes greedy \
  --iterations 1
```

Check spill-dominance on original vs mutated IR:

```
./tools/spill_fuzz/check_spill_dominance_repro.sh \
  --llc /opt/rocm-6.4.4/lib/llvm/bin/llc \
  --mcpu gfx942 \
  --input kernels/pele/kernel-pc_cmpflx_launch-a7d5b88a0dde5efe7b96045874e05330cd93917d.ll \
  --vgpr 20 \
  --sgpr 108
```

Devcontainer alias (optional):

```
alias dcv='devcontainer exec --workspace-folder . -- bash -lc'
```

You can override the ROCm prefix (default `/opt/rocm`) with `--rocm`.

## Notes

- This is a configuration fuzzer, not a structural MIR mutator yet.
- The spill-dominance check is conservative and may report false positives on
  exotic control flow. If you hit one, save the mutated MIR and minimize.
