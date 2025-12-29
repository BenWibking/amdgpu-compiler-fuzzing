#!/usr/bin/env python3
"""Parse AMDGPU code object metadata and emit a simple kernel spec."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


KERNELS_RE = re.compile(r"^\\s*(amdhsa\\.kernels|kernels):\\s*$")
KERNEL_NAME_RE = re.compile(r"^\\s*-\\s*\\.name:\\s*(.+)$")
ARGS_RE = re.compile(r"^\\s*\\.args:\\s*$")
ARG_ITEM_KV_RE = re.compile(r"^\\s*-\\s*\\.([A-Za-z0-9_]+):\\s*(.+)$")
KEY_VALUE_RE = re.compile(r"^\\s*\\.([A-Za-z0-9_]+):\\s*(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hsaco", help="HSACO path")
    parser.add_argument("--llvm-readobj", default="llvm-readobj")
    parser.add_argument("--kernel", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def run_readobj(path: str, llvm_readobj: str) -> str:
    cmd = [llvm_readobj, "--amdgpu-code-object-metadata", path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError("llvm-readobj failed")
    return proc.stdout


def clean_value(val: str) -> str:
    val = val.strip()
    if val.startswith(("\"", "'")) and val.endswith(("\"", "'")) and len(val) >= 2:
        return val[1:-1]
    return val


def parse_metadata(text: str):
    kernels = []
    in_kernels = False
    in_args = False
    args_indent = None
    current_kernel = None
    current_arg = None

    for line in text.splitlines():
        if KERNELS_RE.match(line):
            in_kernels = True
            continue
        if not in_kernels:
            continue

        name_match = KERNEL_NAME_RE.match(line)
        if name_match:
            current_kernel = {"name": clean_value(name_match.group(1)), "args": []}
            kernels.append(current_kernel)
            in_args = False
            current_arg = None
            continue

        if current_kernel is None:
            continue

        if ARGS_RE.match(line):
            in_args = True
            args_indent = len(line) - len(line.lstrip())
            current_arg = None
            continue

        if in_args:
            indent = len(line) - len(line.lstrip())
            if args_indent is not None and indent <= args_indent and line.strip():
                in_args = False
                current_arg = None
                args_indent = None
            else:
                kv_inline = ARG_ITEM_KV_RE.match(line)
                if kv_inline:
                    current_arg = {}
                    current_kernel["args"].append(current_arg)
                    current_arg[kv_inline.group(1)] = clean_value(kv_inline.group(2))
                    continue
                kv = KEY_VALUE_RE.match(line)
                if kv and current_arg is not None:
                    current_arg[kv.group(1)] = clean_value(kv.group(2))

    return kernels


def arg_supported(arg) -> bool:
    kind = arg.get("value_kind", "")
    if kind.startswith("hidden_"):
        return False
    return kind in ("global_buffer", "by_value", "value")


def emit_spec(path: Path, kernel_name: str, args) -> None:
    lines = [f"kernel {kernel_name}"]
    for arg in args:
        lines.append(
            "arg {kind} {size} {addr}".format(
                kind=arg.get("value_kind", "unknown"),
                size=arg.get("size", "0"),
                addr=arg.get("address_space", "unknown"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    text = run_readobj(args.hsaco, args.llvm_readobj)
    kernels = parse_metadata(text)

    for kernel in kernels:
        if args.kernel and kernel["name"] != args.kernel:
            continue
        explicit_args = [arg for arg in kernel["args"] if arg_supported(arg)]
        if len(explicit_args) == 0:
            continue
        emit_spec(Path(args.out), kernel["name"], explicit_args)
        return 0

    return 3


if __name__ == "__main__":
    raise SystemExit(main())
