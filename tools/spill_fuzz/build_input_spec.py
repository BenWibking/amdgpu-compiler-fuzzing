#!/usr/bin/env python3
"""Convert a JSON input spec into the flat format used by hip_runner."""

import argparse
import json
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    return parser.parse_args()


def emit_lines(data: dict) -> List[str]:
    lines: List[str] = []

    seed = data.get("seed")
    if seed is not None:
        lines.append(f"seed {int(seed)}")

    launch = data.get("launch")
    if launch is not None:
        grid = launch.get("grid", [1, 1, 1])
        block = launch.get("block", [1, 1, 1])
        if len(grid) != 3 or len(block) != 3:
            raise ValueError("launch grid/block must be length-3 arrays")
        lines.append(
            "launch {} {} {} {} {} {}".format(
                int(grid[0]), int(grid[1]), int(grid[2]),
                int(block[0]), int(block[1]), int(block[2]),
            )
        )

    buffers = data.get("buffers", {})
    for key, value in buffers.items():
        if isinstance(value, dict):
            size = value.get("size_bytes")
        else:
            size = value
        if size is None:
            raise ValueError(f"buffer {key} missing size_bytes")
        lines.append(f"buffer {int(key)} {int(size)}")

    values = data.get("values", {})
    for key, value in values.items():
        if isinstance(value, dict):
            if "int" in value:
                lines.append(f"value {int(key)} int {int(value['int'])}")
            elif "hex" in value:
                hex_str = str(value["hex"]).strip()
                lines.append(f"value {int(key)} hex {hex_str}")
            elif "bytes" in value:
                bytes_list = value["bytes"]
                if not isinstance(bytes_list, list):
                    raise ValueError(f"value {key} bytes must be a list")
                byte_text = " ".join(str(int(b)) for b in bytes_list)
                lines.append(f"value {int(key)} bytes {byte_text}")
            else:
                raise ValueError(f"value {key} dict must have int, hex, or bytes")
        else:
            lines.append(f"value {int(key)} int {int(value)}")

    return lines


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    lines = emit_lines(data)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
