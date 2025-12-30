# Repository Guidelines

## Project Structure & Module Organization

- `tools/spill_fuzz/` holds the main fuzz harness, GPU runner scripts, and helper utilities (Python, shell, C++).
- `docs/markdown/` contains the MkDocs source; site config lives in `docs/mkdocs.yml`.
- `extern/` vendors large upstream dependencies (e.g., `llvm-project/`, `mesa/`); avoid editing unless you are intentionally updating a dependency.
- `spill_fuzz_out/` is used for fuzz outputs and artifacts; treat it as generated data.

## Build, Test, and Development Commands

- Build/serve docs:
  - `mkdocs build -f docs/mkdocs.yml` builds the static site into `site/`.
  - `mkdocs serve -f docs/mkdocs.yml` runs a local docs server.
- Run the spill fuzzer (example):
  - `./tools/spill_fuzz/spill_fuzz.py --corpus llvm-project/llvm/test/CodeGen/AMDGPU --llc llvm-project/build/bin/llc --mcpu gfx90a --passes greedy --iterations 50`
- GPU oracle (optional, HIP):
  - `./tools/spill_fuzz/run_on_gpu.sh` builds/runs the HIP runner on first use.

## Coding Style & Naming Conventions

- Match the style of nearby files; keep diffs minimal.
- Python: 4-space indentation, snake_case for functions/variables; keep scripts runnable from repo root.
- Shell: prefer `bash`-compatible syntax; keep scripts in `tools/spill_fuzz/`.
- C++: follow existing LLVM/clang-style formatting in `tools/spill_fuzz/hip_runner.cpp`.

## Testing Guidelines

- There is no dedicated test harness; validate changes by running the fuzz script on a small corpus and/or the GPU oracle.
- When adding new checks, document how to exercise them in `tools/spill_fuzz/README.md`.

## Commit & Pull Request Guidelines

- Commits are short, imperative, and lowercase (e.g., `fix build`, `add docs build`).
- PRs should describe the change, list commands run, and include a minimal repro or sample input if relevant (e.g., a failing `.mir`).
- If changes affect fuzzing behavior, note the corpus, `llc` build, and `--mcpu` used.

## Configuration & Dependency Notes

- Required: local LLVM build with AMDGPU backend and a `.mir` corpus.
- Optional: ROCm + `hipcc` for the GPU oracle; see `tools/spill_fuzz/README.md` for environment overrides.
