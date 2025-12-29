# Installation

This project relies on a local LLVM build with the AMDGPU backend, plus optional
ROCm tooling for GPU execution. The steps below outline a typical setup.

## Build LLVM with AMDGPU

1. Clone LLVM and configure a build that includes AMDGPU.
2. Build `llc` and its dependencies.

Example:

```
mkdir -p llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=clang \
  -DLLVM_TARGETS_TO_BUILD=AMDGPU \
  -DCMAKE_BUILD_TYPE=Release
ninja llc
```

Set `LLC` if `llc` is not on your `PATH`:

```
export LLC=/path/to/llvm-project/build/bin/llc
```

## Optional: ROCm + HIP runner

The GPU oracle uses HIP to compile and run a kernel. Install ROCm and ensure
`hipcc` is available on your `PATH`. You can also override it with `HIPCC`.

Build the runner manually if desired:

```
./tools/spill_fuzz/build_hip_runner.sh
```

## Sanity check

Use the quick start in `docs/markdown/getting-started.md` to confirm the fuzzer
can execute with your LLVM build and corpus.
