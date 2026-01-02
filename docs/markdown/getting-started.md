# Getting Started

This page walks through running the spill fuzz harness included in this repo.
For deeper details, see `tools/spill_fuzz/README.md`.

## Prerequisites

- A local LLVM build that provides `llc` for the AMDGPU backend.
- A corpus of AMDGPU `.mir` files (the LLVM test suite works well).

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

## What you get

- The fuzzer mutates register limits and pass settings in a random `.mir` file.
- It runs `llc` with the requested passes and checks spill-dominance.
- Failures and minimized outputs are written under `spill_fuzz_out/` by default.

## Optional GPU oracle (HIP)

You can also compare GPU execution between a high-register baseline and the
current fuzzed variant:

```
./tools/spill_fuzz/spill_fuzz.py \
  --corpus extern/llvm-project/llvm/test/CodeGen/AMDGPU \
  --llc extern/llvm-project/build/bin/llc \
  --mcpu gfx90a \
  --gpu-cmd "./tools/spill_fuzz/run_on_gpu.sh"
```

The GPU runner is built automatically the first time it is invoked. You can
also build it manually:

```
./tools/spill_fuzz/build_hip_runner.sh
```

## Example: Pele kernel to LLVM IR

The Pele reproducers under `kernels/pele/` can be lowered to device-only LLVM
IR using the helper script:

```
./tools/spill_fuzz/pele_hip_to_ll.sh -t pelec_repro2_dodecane_lu
```

Devcontainer one-liner:

```
devcontainer exec --workspace-folder . -- bash -lc "./tools/spill_fuzz/pele_hip_to_ll.sh -t pelec_repro2_dodecane_lu"
```

This writes `kernels/pele/pelec_repro2_dodecane_lu.ll`, which you can feed to
`llc` for regalloc/codegen repros.

Example `llc` invocation:

```
/opt/rocm-6.4.4/lib/llvm/bin/llc \
  -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 \
  kernels/pele/pelec_repro2_dodecane_lu.ll -o /tmp/pele.s
```

Extract each kernel into its own `.ll`:

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

Demangle kernel names (also used for output filenames when extracting):

```
./tools/spill_fuzz/extract_amdgpu_kernels.sh --demangle \
  kernels/pele/pelec_repro2_dodecane_lu.ll
```

When `--demangle` is used, the script writes a `kernel-map.txt` file that maps
mangled names to demangled names and output paths. The output filenames use the
demangled base name (without parameter lists). Collisions get a hash suffix, and
long names are truncated with a hash to avoid filename length limits.
