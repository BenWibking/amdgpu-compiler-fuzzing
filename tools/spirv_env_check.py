#!/usr/bin/env python3
import struct
import sys


SOURCE_LANG = {
    0: "Unknown",
    1: "ESSL",
    2: "GLSL",
    3: "OpenCL_C",
    4: "OpenCL_CPP",
    5: "HLSL",
}

EXEC_MODEL = {
    0: "Vertex",
    1: "TessellationControl",
    2: "TessellationEvaluation",
    3: "Geometry",
    4: "Fragment",
    5: "GLCompute",
    6: "Kernel",
}

ADDRESSING_MODEL = {
    0: "Logical",
    1: "Physical32",
    2: "Physical64",
    5348: "PhysicalStorageBuffer64",
}

MEMORY_MODEL = {
    0: "Simple",
    1: "GLSL450",
    2: "OpenCL",
    3: "Vulkan",
}

CAPABILITY = {
    1: "Matrix",
    2: "Shader",
    3: "Geometry",
    4: "Tessellation",
    5: "Addresses",
    6: "Linkage",
    7: "Kernel",
    8: "Vector16",
    9: "Float16Buffer",
    10: "Float16",
    11: "Float64",
    12: "Int64",
    13: "Int64Atomics",
    14: "ImageBasic",
    15: "ImageReadWrite",
    16: "ImageMipmap",
}


def read_words(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("SPIR-V size is not a multiple of 4 bytes")
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    return words


def decode_string(words):
    b = struct.pack("<%dI" % len(words), *words)
    s = b.split(b"\x00", 1)[0]
    return s.decode("ascii", errors="replace")


def main():
    if len(sys.argv) != 2:
        print("usage: spirv_env_check.py <file.spv>")
        return 1
    words = read_words(sys.argv[1])
    if words[0] != 0x07230203:
        raise ValueError("unexpected SPIR-V magic")

    source_lang = None
    source_ver = None
    exec_models = []
    entry_points = []
    addr_model = None
    mem_model = None
    caps = []

    i = 5
    while i < len(words):
        word = words[i]
        wc = word >> 16
        op = word & 0xFFFF
        if wc == 0:
            break
        if op == 3 and wc >= 3:
            source_lang = words[i + 1]
            source_ver = words[i + 2]
        elif op == 14 and wc >= 3:
            addr_model = words[i + 1]
            mem_model = words[i + 2]
        elif op == 15 and wc >= 4:
            exec = words[i + 1]
            name = decode_string(words[i + 3 : i + wc])
            exec_models.append(exec)
            entry_points.append(name)
        elif op == 17 and wc >= 2:
            caps.append(words[i + 1])
        i += wc

    def map_or(val, table):
        return table.get(val, "Unknown(%s)" % val)

    print("source_language: %s" % (map_or(source_lang, SOURCE_LANG)))
    if source_ver is not None:
        print("source_version: %s" % source_ver)
    print("addressing_model: %s" % (map_or(addr_model, ADDRESSING_MODEL)))
    print("memory_model: %s" % (map_or(mem_model, MEMORY_MODEL)))
    if exec_models:
        print(
            "execution_models: %s"
            % ", ".join(map_or(m, EXEC_MODEL) for m in exec_models)
        )
    if entry_points:
        print("entry_points: %s" % ", ".join(entry_points))
    if caps:
        print("capabilities: %s" % ", ".join(map_or(c, CAPABILITY) for c in caps))

    if mem_model == 2 or (source_lang in (3, 4)) or (6 in exec_models):
        print("env_hint: OpenCL-like")
    elif mem_model in (1, 3) and 5 in exec_models:
        print("env_hint: Vulkan-like")
    else:
        print("env_hint: Unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
