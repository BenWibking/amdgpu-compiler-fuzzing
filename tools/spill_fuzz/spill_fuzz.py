#!/usr/bin/env python3
"""AMDGPU spill fuzzing harness with non-GPU oracles.

This is a configuration fuzzer: it varies register limits and pass settings
against a MIR corpus, then applies static oracles (machine verifier and a
spill-dominance check). It can optionally hand off to a GPU runner.
"""

import argparse
import glob
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


SPILL_SAVE_RE = re.compile(r"\bSI_SPILL_\w+_SAVE\b")
SPILL_RESTORE_RE = re.compile(r"\bSI_SPILL_\w+_RESTORE\b")
STACK_SLOT_RE = re.compile(r"%stack\.(\d+)")
BB_RE = re.compile(r"^\s*bb\.(\S+):")
SUCCESSORS_RE = re.compile(r"\bsuccessors:\s*(.*)$")
SUCCESSOR_BB_RE = re.compile(r"%bb\.([A-Za-z0-9_\.]+)")
FUNC_NAME_RE = re.compile(r"^name:\s*(\S+)")


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
        gpu_cmd: Optional[str],
    ) -> None:
        self.llc = llc
        self.mcpu = mcpu
        self.passes = passes
        self.verify_machine_instrs = verify_machine_instrs
        self.num_vgpr = num_vgpr
        self.num_sgpr = num_sgpr
        self.spill_sgpr_to_vgpr = spill_sgpr_to_vgpr
        self.gpu_cmd = gpu_cmd


class SpillIssue:
    def __init__(self, function: str, block: str, slot: str, reason: str) -> None:
        self.function = function
        self.block = block
        self.slot = slot
        self.reason = reason


class MirFunction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.blocks: List[str] = []
        self.succs: Dict[str, List[str]] = {}
        self.instrs: Dict[str, List[str]] = {}

    def add_block(self, bb: str) -> None:
        if bb not in self.instrs:
            self.blocks.append(bb)
            self.instrs[bb] = []
            self.succs.setdefault(bb, [])

    def add_instr(self, bb: str, instr: str) -> None:
        self.instrs.setdefault(bb, []).append(instr)

    def add_succs(self, bb: str, succs: List[str]) -> None:
        self.succs[bb] = succs


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
    parser.add_argument("--spill-sgpr-to-vgpr", choices=["on", "off"], default=None)
    parser.add_argument("--gpu-cmd", default=None,
                        help="Command to run a GPU oracle. It receives the MIR path.")
    parser.add_argument("--out-dir", default="spill_fuzz_out")
    return parser.parse_args()


def collect_inputs(corpus_dir: Path) -> List[Path]:
    return sorted(p for p in corpus_dir.rglob("*.mir") if p.is_file())


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


def apply_reg_limits_to_ir(ir_text: str, num_vgpr: int, num_sgpr: int) -> str:
    lines = ir_text.splitlines()
    out_lines = []
    in_define = False
    for line in lines:
        if line.startswith("define "):
            in_define = True
            if "amdgpu-num-vgpr" in line or "amdgpu-num-sgpr" in line:
                line = re.sub(r'"amdgpu-num-vgpr"="\d+"',
                              f'"amdgpu-num-vgpr"="{num_vgpr}"', line)
                line = re.sub(r'"amdgpu-num-sgpr"="\d+"',
                              f'"amdgpu-num-sgpr"="{num_sgpr}"', line)
            else:
                insert = f' "amdgpu-num-vgpr"="{num_vgpr}" "amdgpu-num-sgpr"="{num_sgpr}"'
                if "{" in line:
                    line = line.replace("{", insert + " {")
                else:
                    line = line + insert
        elif in_define and line.startswith("}"):
            in_define = False
        out_lines.append(line)
    return "\n".join(out_lines)


def rewrite_mir_with_limits(mir_text: str, num_vgpr: int, num_sgpr: int) -> str:
    if "--- |" not in mir_text:
        return mir_text
    pre, rest = mir_text.split("--- |", 1)
    ir, post = rest.split("...", 1)
    ir = apply_reg_limits_to_ir(ir, num_vgpr, num_sgpr)
    return pre + "--- |" + ir + "..." + post


def parse_mir_functions(mir_text: str) -> List[MirFunction]:
    functions: List[MirFunction] = []
    current_func: Optional[MirFunction] = None
    in_body = False
    current_bb: Optional[str] = None

    for raw_line in mir_text.splitlines():
        line = raw_line.rstrip("\n")
        name_match = FUNC_NAME_RE.match(line.strip())
        if name_match:
            current_func = MirFunction(name_match.group(1))
            functions.append(current_func)
            in_body = False
            current_bb = None
            continue

        if line.strip().startswith("body:"):
            in_body = True
            continue

        if not in_body or current_func is None:
            continue

        bb_match = BB_RE.match(line)
        if bb_match:
            current_bb = f"bb.{bb_match.group(1)}"
            current_func.add_block(current_bb)
            continue

        if current_bb is None:
            continue

        succ_match = SUCCESSORS_RE.search(line)
        if succ_match:
            succs = SUCCESSOR_BB_RE.findall(succ_match.group(1))
            succ_blocks = [f"bb.{s}" for s in succs]
            current_func.add_succs(current_bb, succ_blocks)
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue

        # Skip metadata-like lines that are not instructions.
        if stripped.startswith("liveins:"):
            continue

        current_func.add_instr(current_bb, stripped)

    return functions


def compute_dominators(func: MirFunction) -> Dict[str, Set[str]]:
    if not func.blocks:
        return {}
    preds: Dict[str, List[str]] = {b: [] for b in func.blocks}
    for b in func.blocks:
        for succ in func.succs.get(b, []):
            preds.setdefault(succ, []).append(b)

    entry = func.blocks[0]
    dom: Dict[str, Set[str]] = {b: set(func.blocks) for b in func.blocks}
    dom[entry] = {entry}

    changed = True
    while changed:
        changed = False
        for b in func.blocks:
            if b == entry:
                continue
            pred_sets = [dom[p] for p in preds.get(b, []) if p in dom]
            if pred_sets:
                new_dom = set.intersection(*pred_sets)
            else:
                new_dom = set()
            new_dom.add(b)
            if new_dom != dom[b]:
                dom[b] = new_dom
                changed = True

    return dom


def check_spill_dominance(mir_text: str) -> List[SpillIssue]:
    issues: List[SpillIssue] = []
    for func in parse_mir_functions(mir_text):
        dom = compute_dominators(func)
        slot_saves: Dict[str, Dict[str, List[int]]] = {}
        slot_restore_sites: List[Tuple[str, str, int]] = []

        for bb in func.blocks:
            instrs = func.instrs.get(bb, [])
            for idx, instr in enumerate(instrs):
                slot_match = STACK_SLOT_RE.search(instr)
                if not slot_match:
                    continue
                slot = slot_match.group(1)
                if SPILL_SAVE_RE.search(instr):
                    slot_saves.setdefault(slot, {}).setdefault(bb, []).append(idx)
                elif SPILL_RESTORE_RE.search(instr):
                    slot_restore_sites.append((slot, bb, idx))

        for slot, bb, idx in slot_restore_sites:
            saves_in_bb = slot_saves.get(slot, {}).get(bb, [])
            if any(s < idx for s in saves_in_bb):
                continue
            dominating_blocks = [b for b in slot_saves.get(slot, {}) if b in dom.get(bb, set())]
            if not dominating_blocks:
                issues.append(
                    SpillIssue(
                        function=func.name,
                        block=bb,
                        slot=slot,
                        reason="restore not dominated by spill save",
                    )
                )
    return issues


def choose_limits(rng: random.Random, min_vgpr: int, max_vgpr: int,
                  min_sgpr: int, max_sgpr: int) -> Tuple[int, int]:
    num_vgpr = rng.randint(min_vgpr, max_vgpr)
    num_sgpr = rng.randint(min_sgpr, max_sgpr)
    return num_vgpr, num_sgpr


def build_llc_cmd(cfg: FuzzConfig, mir_path: Path) -> List[str]:
    cmd = [
        cfg.llc,
        f"-mtriple=amdgcn-amd-amdhsa",
        f"-mcpu={cfg.mcpu}",
        f"-run-pass={cfg.passes}",
        "-o",
        "-",
        str(mir_path),
    ]
    if cfg.verify_machine_instrs:
        cmd.append("-verify-machineinstrs")
    if cfg.spill_sgpr_to_vgpr is not None:
        cmd.append(f"-amdgpu-spill-sgpr-to-vgpr={'1' if cfg.spill_sgpr_to_vgpr else '0'}")
    return cmd


def build_verifier_cmd(cfg: FuzzConfig, mir_path: Path) -> List[str]:
    cmd = [
        cfg.llc,
        f"-mtriple=amdgcn-amd-amdhsa",
        f"-mcpu={cfg.mcpu}",
        "-run-pass=machineverifier",
        "-o",
        "-",
        str(mir_path),
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

    mir_text = input_path.read_text(encoding="utf-8")
    mutated_text = rewrite_mir_with_limits(mir_text, num_vgpr, num_sgpr)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"{input_path.stem}.vgpr{num_vgpr}.sgpr{num_sgpr}.mir"
    tmp_path.write_text(mutated_text, encoding="utf-8")

    verify_cmd = build_verifier_cmd(cfg, tmp_path)
    vcode, _, vstderr = run_cmd(verify_cmd)
    if vcode != 0:
        sys.stderr.write(f"MIR failed machine verifier before passes: {verify_cmd}\n{vstderr}\n")
        return 1

    cmd = build_llc_cmd(cfg, tmp_path)
    code, stdout, stderr = run_cmd(cmd)
    if code != 0:
        sys.stderr.write(f"llc failed: {cmd}\n{stderr}\n")
        return 1

    issues = check_spill_dominance(stdout)
    if issues:
        sys.stderr.write("Spill dominance issues:\n")
        for issue in issues:
            sys.stderr.write(f"  {issue.function} {issue.block} %stack.{issue.slot}: {issue.reason}\n")
        return 1

    if cfg.gpu_cmd:
        gpu_cmd = cfg.gpu_cmd.split() + [str(tmp_path)]
        gcode, _, gerr = run_cmd(gpu_cmd)
        if gcode != 0:
            sys.stderr.write(f"GPU runner failed: {gpu_cmd}\n{gerr}\n")
            return 1

    return 0


def main() -> int:
    args = parse_args()
    args.llc = resolve_llc(args.llc)
    corpus_dir = Path(args.corpus)
    inputs = collect_inputs(corpus_dir)
    if not inputs:
        sys.stderr.write(f"No .mir inputs found in {corpus_dir}\n")
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
