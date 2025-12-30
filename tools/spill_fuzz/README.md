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

- Picks a random `.mir` file from the corpus.
- Injects `"amdgpu-num-vgpr"`/`"amdgpu-num-sgpr"` into the IR section.
- Runs `llc -run-pass=<passes>` and captures MIR output.
- Runs a spill-dominance oracle on `SI_SPILL_*_SAVE/RESTORE` pairs.
- Optionally calls a GPU runner.

## Non-GPU oracles

- `-verify-machineinstrs` from `llc` (use `--verify-machineinstrs`).
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
HIPCC=/opt/rocm/bin/hipcc
```

## Notes

- This is a configuration fuzzer, not a structural MIR mutator yet.
- The spill-dominance check is conservative and may report false positives on
  exotic control flow. If you hit one, save the mutated MIR and minimize.
