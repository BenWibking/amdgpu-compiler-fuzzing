# AMDGPU Compiler Fuzzing

This repository contains tooling and notes for fuzzing the AMDGPU
compiler toolchain.

## Prerequisites

- A local LLVM build with the AMDGPU backend (for `llc`).
- A corpus of AMDGPU `.mir` files (the LLVM test suite works well).
- Optional: ROCm + `hipcc` if you want to use the GPU oracle.

## Documentation

Docs are in `docs/markdown/` and can be built with MkDocs:

```sh
mkdocs build -f docs/mkdocs.yml
mkdocs serve -f docs/mkdocs.yml
```
