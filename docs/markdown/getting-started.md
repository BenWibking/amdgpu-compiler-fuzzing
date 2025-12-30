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
