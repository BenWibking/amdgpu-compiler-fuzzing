# vk_aco_compile

Minimal Vulkan compute pipeline creator to force SPIR-V compilation through
Mesa/RADV and ACO.

Build:

```sh
make
```

Run (example):

```sh
RADV_PERFTEST=aco \
RADV_DEBUG=shaders \
ACO_DEBUG=validate,info \
./vk_aco_compile ../../pc_cmpflx.spv pc_cmpflx_launch 0 0
```

OpenCL note:
- Use OpenCL 1.2 for clspv builds of `pc_cmpflx_opencl.cl`; CL2.0 currently
  triggers a clspv segfault in this environment.

SPIR-V environment check:

```sh
python3 ../../tools/spirv_env_check.py ../../pc_cmpflx_from_llvm.spv
```

Arguments:
- `spv_path`: SPIR-V module to load.
- `entry`: entry point name (default `pc_cmpflx_launch`).
- `num_bindings`: number of storage buffer bindings to declare (set to `0` to
  enable reflection for set 0).
- `push_constant_bytes`: push constant range size (set to `0` to enable
  reflection).

Notes:
- Reflection only handles common buffer/push-constant patterns; if pipeline
  creation fails, try a manual `num_bindings` or `push_constant_bytes`.
