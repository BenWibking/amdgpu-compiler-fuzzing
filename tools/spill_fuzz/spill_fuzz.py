#!/usr/bin/env python3
"""AMDGPU spill fuzzing harness with a GPU oracle.

This is a configuration fuzzer: it varies register limits and pass settings
against a MIR corpus, then relies on a GPU runner to detect issues.
"""

import argparse
import glob
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
NON_HSA_SHADER_CC_RE = re.compile(r"\bamdgpu_(ps|vs|gs|hs|es|ls|cs)\b")
NON_HSA_SHADER_ATTR_RE = re.compile(r'"amdgpu-shader-type"\s*=\s*"\w+"')
NON_HSA_FUNC_RE = re.compile(r"\bamdgpu_cs_chain_func\b")
WMMA_INTRINSIC_RE = re.compile(r"\bllvm\.amdgcn\.wmma\.")
OPENCL_PRINTF_RE = re.compile(r"\bllvm\.amdgcn\.printf\b")
FLAT_ATOMIC_FADD_RE = re.compile(r"\bllvm\.amdgcn\.flat\.atomic\.fadd\b")
R600_INTRINSIC_RE = re.compile(r"\bllvm\.r600\.")
LEGACY_FMA_RE = re.compile(r"\bllvm\.amdgcn\.fma\.legacy\b")
CODE_OBJECT_VERSION_RE = re.compile(r"\bCODE_OBJECT_VERSION\b")
DYNAMIC_ALLOCA_RE = re.compile(r"\balloca\b.*\baddrspace\(5\)\b", re.IGNORECASE)
SMFMAC_INTRINSIC_RE = re.compile(r"\bllvm\.amdgcn\.smfmac\.")
GLOBAL_LDS_GDS_RE = re.compile(r"@[\w\.\$]+.*addrspace\((2|3)\)")
NON_KERNEL_DEFINE_RE = re.compile(r"^define\b(?!.*\bamdgpu_kernel\b)", re.MULTILINE)
MFMA_INTRINSIC_RE = re.compile(r"\bllvm\.amdgcn\.mfma\.")
INVALID_ADDRSPACECAST_RE = re.compile(r"\binvalid addrspacecast\b", re.IGNORECASE)
GFX_CALLING_CONV_RE = re.compile(r"\bamdgpu_gfx\b")
FDOT2_INTRINSIC_RE = re.compile(r"\bllvm\.amdgcn\.fdot2\.")
WORKGROUP_ATTR_TEST_RE = re.compile(r"\bamdgpu-max-num-workgroups\b")
READ_REGISTER_INVALID_RE = re.compile(r"\btest_invalid_read_m0\b")
ATOMIC_FMAX_INTRINSIC_RE = re.compile(r"\bllvm\.amdgcn\.(raw_ptr_buffer_atomic_fmax|raw_buffer_atomic_fmax|struct_ptr_buffer_atomic_fmax|struct_buffer_atomic_fmax|image_atomic_fmax|flat_atomic_fmax|global_atomic_fmax)\b")


class FuzzConfig:
    def __init__(
        self,
        llc: str,
        mcpu: str,
        passes: str,
        verify_machine_instrs: bool,
        num_vgpr: int,
        num_sgpr: int,
        spill_sgpr_to_vgpr: Optional[bool],
        gpu_cmd: List[str],
    ) -> None:
        self.llc = llc
        self.mcpu = mcpu
        self.passes = passes
        self.verify_machine_instrs = verify_machine_instrs
        self.num_vgpr = num_vgpr
        self.num_sgpr = num_sgpr
        self.spill_sgpr_to_vgpr = spill_sgpr_to_vgpr
        self.gpu_cmd = gpu_cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Directory with .mir files")
    parser.add_argument("--llc", default=os.environ.get("LLC", "llc"))
    parser.add_argument("--mcpu", default="gfx90a")
    parser.add_argument("--passes", default="greedy", help="Comma-separated llc passes")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-vgpr", type=int, default=8)
    parser.add_argument("--max-vgpr", type=int, default=128)
    parser.add_argument("--min-sgpr", type=int, default=8)
    parser.add_argument("--max-sgpr", type=int, default=128)
    parser.add_argument("--verify-machineinstrs", action="store_true")
    parser.add_argument("--spill-sgpr-to-vgpr", choices=["on", "off"], default="on")
    parser.add_argument("--gpu-cmd", required=True,
                        help="Command to run a GPU oracle. It receives the MIR path.")
    parser.add_argument("--out-dir", default="spill_fuzz_out")
    return parser.parse_args()


def collect_inputs(corpus_dir: Path) -> List[Path]:
    return sorted(p for p in corpus_dir.rglob("*.ll") if p.is_file())


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def resolve_llc(llc_arg: str) -> str:
    if os.path.isfile(llc_arg) and os.access(llc_arg, os.X_OK):
        return llc_arg
    if os.sep in llc_arg:
        candidates = [
            "/opt/rocm/lib/llvm/bin/llc",
        ]
        candidates.extend(sorted(glob.glob("/opt/rocm-*/lib/llvm/bin/llc")))
        candidates.extend(sorted(glob.glob("/opt/rocm-*/llvm/bin/llc")))
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                sys.stderr.write(f"llc not found at {llc_arg}; using {candidate}\n")
                return candidate
        fallback = shutil.which("llc")
        if fallback:
            sys.stderr.write(f"llc not found at {llc_arg}; using {fallback}\n")
            return fallback
        return llc_arg
    resolved = shutil.which(llc_arg)
    if resolved:
        if resolved != llc_arg:
            sys.stderr.write(f"using llc from PATH: {resolved}\n")
        return resolved
    return llc_arg


def resolve_gpu_cmd(gpu_cmd: str) -> List[str]:
    argv = shlex.split(gpu_cmd)
    if not argv:
        raise ValueError("GPU oracle command is empty")
    exe = argv[0]
    if os.path.isfile(exe) and os.access(exe, os.X_OK):
        return argv
    resolved = shutil.which(exe)
    if resolved is None:
        raise ValueError(f"GPU oracle command not found or not executable: {exe}")
    argv[0] = resolved
    return argv


def apply_reg_limits_to_ir(ir_text: str, num_vgpr: int, num_sgpr: int) -> str:
    def insert_attrs_before_metadata(line: str, attrs: str) -> str:
        meta_idx = line.find(" !")
        if meta_idx != -1:
            return line[:meta_idx] + attrs + line[meta_idx:]
        return line + attrs

    lines = ir_text.splitlines()
    out_lines = []
    in_define = False
    pending_insert = False
    for line in lines:
        if line.startswith("define "):
            in_define = True
            pending_insert = True

        if in_define:
            if "amdgpu-num-vgpr" in line or "amdgpu-num-sgpr" in line:
                line = re.sub(r'"amdgpu-num-vgpr"="\d+"',
                              f'"amdgpu-num-vgpr"="{num_vgpr}"', line)
                line = re.sub(r'"amdgpu-num-sgpr"="\d+"',
                              f'"amdgpu-num-sgpr"="{num_sgpr}"', line)
                pending_insert = False
            elif pending_insert and "{" in line:
                insert = f' "amdgpu-num-vgpr"="{num_vgpr}" "amdgpu-num-sgpr"="{num_sgpr}"'
                brace_idx = line.find("{")
                meta_idx = line.find(" !")
                if meta_idx != -1 and meta_idx < brace_idx:
                    line = line[:meta_idx] + insert + line[meta_idx:]
                else:
                    line = line[:brace_idx] + insert + line[brace_idx:]
                pending_insert = False
            elif pending_insert and line.strip() == "{":
                insert = f' "amdgpu-num-vgpr"="{num_vgpr}" "amdgpu-num-sgpr"="{num_sgpr}"'
                if out_lines:
                    out_lines[-1] = insert_attrs_before_metadata(out_lines[-1], insert)
                pending_insert = False
        if in_define and line.startswith("}"):
            in_define = False
            pending_insert = False
        out_lines.append(line)
    return "\n".join(out_lines)


def has_non_hsa_shader_cc(ir_text: str) -> bool:
    return (NON_HSA_SHADER_CC_RE.search(ir_text) is not None
            or NON_HSA_SHADER_ATTR_RE.search(ir_text) is not None
            or NON_HSA_FUNC_RE.search(ir_text) is not None)


def has_unsupported_wmma(ir_text: str, mcpu: str) -> bool:
    if not WMMA_INTRINSIC_RE.search(ir_text):
        return False
    return not (mcpu.startswith("gfx11") or mcpu.startswith("gfx12"))


def has_unsupported_flat_atomic_fadd(ir_text: str, mcpu: str) -> bool:
    if not FLAT_ATOMIC_FADD_RE.search(ir_text):
        return False
    return not (mcpu.startswith("gfx94") or mcpu.startswith("gfx95"))


def has_unsupported_smfmac(ir_text: str, mcpu: str) -> bool:
    if not SMFMAC_INTRINSIC_RE.search(ir_text):
        return False
    return not mcpu.startswith("gfx95")


def has_opencl_printf(ir_text: str) -> bool:
    return OPENCL_PRINTF_RE.search(ir_text) is not None


def has_r600_intrinsics(ir_text: str) -> bool:
    return R600_INTRINSIC_RE.search(ir_text) is not None


def has_legacy_fma(ir_text: str) -> bool:
    return LEGACY_FMA_RE.search(ir_text) is not None


def has_code_object_version_token(ir_text: str) -> bool:
    return CODE_OBJECT_VERSION_RE.search(ir_text) is not None


def has_dynamic_alloca(ir_text: str) -> bool:
    return DYNAMIC_ALLOCA_RE.search(ir_text) is not None


def has_lds_gds_in_non_kernel(ir_text: str) -> bool:
    return GLOBAL_LDS_GDS_RE.search(ir_text) is not None and NON_KERNEL_DEFINE_RE.search(ir_text) is not None


def has_unsupported_mfma(ir_text: str, mcpu: str) -> bool:
    if not MFMA_INTRINSIC_RE.search(ir_text):
        return False
    return not (mcpu.startswith("gfx90") or mcpu.startswith("gfx94") or mcpu.startswith("gfx95"))


def has_invalid_addrspacecast(ir_text: str) -> bool:
    return INVALID_ADDRSPACECAST_RE.search(ir_text) is not None


def has_gfx_calling_conv(ir_text: str) -> bool:
    return GFX_CALLING_CONV_RE.search(ir_text) is not None


def has_unsupported_fdot2(ir_text: str, mcpu: str) -> bool:
    if not FDOT2_INTRINSIC_RE.search(ir_text):
        return False
    return not (mcpu.startswith("gfx94") or mcpu.startswith("gfx95"))


def has_workgroup_attr_tests(ir_text: str) -> bool:
    return WORKGROUP_ATTR_TEST_RE.search(ir_text) is not None


def has_invalid_read_register_tests(ir_text: str) -> bool:
    return READ_REGISTER_INVALID_RE.search(ir_text) is not None


def has_unsupported_atomic_fmax(ir_text: str, mcpu: str) -> bool:
    if not ATOMIC_FMAX_INTRINSIC_RE.search(ir_text):
        return False
    return not (mcpu.startswith("gfx10") or mcpu.startswith("gfx11") or mcpu.startswith("gfx94") or mcpu.startswith("gfx95"))


def rewrite_mir_with_limits(mir_text: str, num_vgpr: int, num_sgpr: int) -> str:
    if "--- |" not in mir_text:
        return mir_text
    pre, rest = mir_text.split("--- |", 1)
    ir, post = rest.split("...", 1)
    ir = apply_reg_limits_to_ir(ir, num_vgpr, num_sgpr)
    return pre + "--- |" + ir + "..." + post


def choose_limits(rng: random.Random, min_vgpr: int, max_vgpr: int,
                  min_sgpr: int, max_sgpr: int) -> Tuple[int, int]:
    num_vgpr = rng.randint(min_vgpr, max_vgpr)
    num_sgpr = rng.randint(min_sgpr, max_sgpr)
    return num_vgpr, num_sgpr


def resolve_pass_name(passes: str) -> str:
    pass_list = [p.strip() for p in passes.split(",") if p.strip()]
    if not pass_list:
        return "greedy"
    if len(pass_list) > 1:
        sys.stderr.write(f"using only first pass for -stop-after/print-after: {pass_list[0]}\n")
    return pass_list[0]


def build_llc_cmd_for_pass(cfg: FuzzConfig, ir_path: Path, pass_name: str) -> List[str]:
    cmd = [
        cfg.llc,
        f"-mtriple=amdgcn-amd-amdhsa",
        f"-mcpu={cfg.mcpu}",
        f"-stop-after={pass_name}",
        f"-print-after={pass_name}",
        "-o",
        "/dev/null",
        str(ir_path),
    ]
    if cfg.verify_machine_instrs:
        cmd.append("-verify-machineinstrs")
    if cfg.spill_sgpr_to_vgpr is not None:
        cmd.append(f"-amdgpu-spill-sgpr-to-vgpr={'1' if cfg.spill_sgpr_to_vgpr else '0'}")
    return cmd


def build_llc_cmd(cfg: FuzzConfig, ir_path: Path) -> List[str]:
    pass_name = resolve_pass_name(cfg.passes)
    return build_llc_cmd_for_pass(cfg, ir_path, pass_name)


def build_pre_ra_verifier_cmd(cfg: FuzzConfig, ir_path: Path) -> List[str]:
    cmd = [
        cfg.llc,
        "-mtriple=amdgcn-amd-amdhsa",
        f"-mcpu={cfg.mcpu}",
        "-stop-after=finalize-isel",
        "-verify-machineinstrs",
        "-o",
        "/dev/null",
        str(ir_path),
    ]
    if cfg.spill_sgpr_to_vgpr is not None:
        cmd.append(f"-amdgpu-spill-sgpr-to-vgpr={'1' if cfg.spill_sgpr_to_vgpr else '0'}")
    return cmd


def run_iteration(rng: random.Random, inputs: List[Path], out_dir: Path, args: argparse.Namespace) -> int:
    input_path = rng.choice(inputs)
    num_vgpr, num_sgpr = choose_limits(rng, args.min_vgpr, args.max_vgpr,
                                       args.min_sgpr, args.max_sgpr)
    spill_sgpr = None
    if args.spill_sgpr_to_vgpr == "on":
        spill_sgpr = True
    elif args.spill_sgpr_to_vgpr == "off":
        spill_sgpr = False

    cfg = FuzzConfig(
        llc=args.llc,
        mcpu=args.mcpu,
        passes=args.passes,
        verify_machine_instrs=args.verify_machineinstrs,
        num_vgpr=num_vgpr,
        num_sgpr=num_sgpr,
        spill_sgpr_to_vgpr=spill_sgpr,
        gpu_cmd=args.gpu_cmd,
    )

    ir_text = input_path.read_text(encoding="utf-8")
    if has_non_hsa_shader_cc(ir_text):
        sys.stderr.write(f"Skipping non-HSA shader module: {input_path}\n")
        return 0
    if has_unsupported_wmma(ir_text, cfg.mcpu):
        sys.stderr.write(f"Skipping WMMA module for mcpu {cfg.mcpu}: {input_path}\n")
        return 0
    if has_unsupported_flat_atomic_fadd(ir_text, cfg.mcpu):
        sys.stderr.write(f"Skipping flat atomic fadd module for mcpu {cfg.mcpu}: {input_path}\n")
        return 0
    if has_unsupported_smfmac(ir_text, cfg.mcpu):
        sys.stderr.write(f"Skipping smfmac module for mcpu {cfg.mcpu}: {input_path}\n")
        return 0
    if has_unsupported_mfma(ir_text, cfg.mcpu):
        sys.stderr.write(f"Skipping mfma module for mcpu {cfg.mcpu}: {input_path}\n")
        return 0
    if has_opencl_printf(ir_text):
        sys.stderr.write(f"Skipping OpenCL printf module: {input_path}\n")
        return 0
    if has_r600_intrinsics(ir_text):
        sys.stderr.write(f"Skipping r600 intrinsic module: {input_path}\n")
        return 0
    if has_legacy_fma(ir_text):
        sys.stderr.write(f"Skipping legacy fma intrinsic module: {input_path}\n")
        return 0
    if has_code_object_version_token(ir_text):
        sys.stderr.write(f"Skipping CODE_OBJECT_VERSION module: {input_path}\n")
        return 0
    if has_dynamic_alloca(ir_text):
        sys.stderr.write(f"Skipping dynamic alloca module: {input_path}\n")
        return 0
    if has_lds_gds_in_non_kernel(ir_text):
        sys.stderr.write(f"Skipping LDS/GDS globals in non-kernel module: {input_path}\n")
        return 0
    if has_invalid_addrspacecast(ir_text):
        sys.stderr.write(f"Skipping invalid addrspacecast module: {input_path}\n")
        return 0
    if has_gfx_calling_conv(ir_text):
        sys.stderr.write(f"Skipping amdgpu_gfx calling convention module: {input_path}\n")
        return 0
    if has_unsupported_fdot2(ir_text, cfg.mcpu):
        sys.stderr.write(f"Skipping fdot2 module for mcpu {cfg.mcpu}: {input_path}\n")
        return 0
    if has_workgroup_attr_tests(ir_text):
        sys.stderr.write(f"Skipping workgroup attribute error test module: {input_path}\n")
        return 0
    if has_invalid_read_register_tests(ir_text):
        sys.stderr.write(f"Skipping invalid read_register test module: {input_path}\n")
        return 0
    if has_unsupported_atomic_fmax(ir_text, cfg.mcpu):
        sys.stderr.write(f"Skipping atomic fmax module for mcpu {cfg.mcpu}: {input_path}\n")
        return 0

    mutated_text = apply_reg_limits_to_ir(ir_text, num_vgpr, num_sgpr)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"{input_path.stem}.vgpr{num_vgpr}.sgpr{num_sgpr}.ll"
    tmp_path.write_text(mutated_text, encoding="utf-8")

    verify_cmd = build_pre_ra_verifier_cmd(cfg, tmp_path)
    vcode, _, vstderr = run_cmd(verify_cmd)
    if vcode != 0:
        sys.stderr.write(f"IR failed machine verifier before passes: {verify_cmd}\n{vstderr}\n")
        return 1

    cmd = build_llc_cmd(cfg, tmp_path)
    code, _, stderr = run_cmd(cmd)
    if code != 0:
        sys.stderr.write(f"llc failed: {cmd}\n{stderr}\n")
        return 1

    gpu_cmd = cfg.gpu_cmd + [str(tmp_path)]
    gcode, _, gerr = run_cmd(gpu_cmd)
    if gcode != 0:
        sys.stderr.write(f"GPU runner failed: {gpu_cmd}\n{gerr}\n")
        return 1

    return 0


def main() -> int:
    args = parse_args()
    args.llc = resolve_llc(args.llc)
    try:
        args.gpu_cmd = resolve_gpu_cmd(args.gpu_cmd)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    corpus_dir = Path(args.corpus)
    inputs = collect_inputs(corpus_dir)
    if not inputs:
        sys.stderr.write(f"No .ll inputs found in {corpus_dir}\n")
        return 2

    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    failures = 0
    for _ in range(args.iterations):
        failures += run_iteration(rng, inputs, out_dir, args)

    if failures:
        sys.stderr.write(f"Failures: {failures}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
