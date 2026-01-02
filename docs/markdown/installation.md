# Installation

This project relies on a local LLVM build with the AMDGPU backend, plus ROCm
tooling for GPU execution. The steps below outline a typical setup.

## Build LLVM with AMDGPU

1. Clone LLVM and configure a build that includes AMDGPU.
2. Build `llc` and its dependencies.

Example:

```
mkdir -p extern/llvm-project/build
cd extern/llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=clang \
  -DLLVM_TARGETS_TO_BUILD=AMDGPU \
  -DCMAKE_BUILD_TYPE=Release
ninja llc
```

Set `LLC` if `llc` is not on your `PATH`:

```
export LLC=/path/to/extern/llvm-project/build/bin/llc
```

## ROCm + HIP runner (required)

The GPU oracle uses HIP to compile and run a kernel. Install ROCm and ensure
`hipcc` is available on your `PATH`. You can also override it with `HIPCC`.

Build the runner manually if desired:

```
./tools/spill_fuzz/build_hip_runner.sh
```

## ROCm devcontainer

There is a preconfigured ROCm devcontainer under `.devcontainer/rocm-container`.
It builds an OpenSUSE + ROCm image and sets up a `leap` user with ROCm tooling
in `PATH`.

Prerequisites:

- Docker (or compatible container runtime).
- VS Code with the Dev Containers extension, or the Dev Containers CLI.

Build the image locally:

```
cd .devcontainer/rocm-container
./build-by-step.sh
```

Then open the repository in VS Code and run "Dev Containers: Open Folder in
Container". The devcontainer configuration uses the image tag
`linux-amd64-rocm-fuzzer:main` and runs `git submodule update --init` on create.

CLI example (requires the Dev Containers CLI):

```
devcontainer up --workspace-folder .
```

Install the CLI with npm if needed:

```
npm install -g @devcontainers/cli
```

## Sanity check

Use the quick start in `docs/markdown/getting-started.md` to confirm the fuzzer
can execute with your LLVM build and corpus.
