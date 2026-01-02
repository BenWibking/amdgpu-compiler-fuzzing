// Simple HIP runner for differential HSACO execution.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct ArgSpec {
  std::string kind;
  size_t size = 0;
  std::string addr_space;
};

struct BufferArg {
  size_t size = 0;
  std::vector<uint8_t> init;
  std::vector<uint8_t> out_a;
  std::vector<uint8_t> out_b;
  void *device_ptr = nullptr;
};

struct LaunchDims {
  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};
};

struct ValueOverride {
  enum class Kind { kInt, kHex, kBytes };
  Kind kind = Kind::kInt;
  uint64_t int_value = 0;
  std::vector<uint8_t> bytes;
};

struct InputSpec {
  bool has_seed = false;
  uint32_t seed = 12345;
  bool has_launch = false;
  LaunchDims launch;
  std::unordered_map<size_t, size_t> buffer_sizes;
  std::unordered_map<size_t, ValueOverride> values;
};

static bool load_spec(const std::string &path, std::string &kernel,
                      std::vector<ArgSpec> &args) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::string token;
  while (in >> token) {
    if (token == "kernel") {
      in >> kernel;
    } else if (token == "arg") {
      ArgSpec arg;
      in >> arg.kind;
      in >> arg.size;
      in >> arg.addr_space;
      args.push_back(arg);
    }
  }
  return !kernel.empty();
}

static bool parse_hex_bytes(const std::string &hex_in,
                            std::vector<uint8_t> &out) {
  std::string hex = hex_in;
  if (hex.rfind("0x", 0) == 0 || hex.rfind("0X", 0) == 0) {
    hex = hex.substr(2);
  }
  if (hex.size() % 2 != 0) {
    return false;
  }
  out.clear();
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    auto hex_byte = hex.substr(i, 2);
    uint8_t value = static_cast<uint8_t>(std::stoul(hex_byte, nullptr, 16));
    out.push_back(value);
  }
  return true;
}

static bool parse_input_spec(const std::string &path, InputSpec &spec) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  std::string line;
  size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    std::istringstream iss(line);
    std::string tag;
    if (!(iss >> tag)) {
      continue;
    }
    if (!tag.empty() && tag[0] == '#') {
      continue;
    }
    if (tag == "seed") {
      uint64_t value = 0;
      if (!(iss >> value)) {
        std::cerr << "invalid seed at line " << line_no << "\n";
        return false;
      }
      spec.has_seed = true;
      spec.seed = static_cast<uint32_t>(value);
    } else if (tag == "launch") {
      unsigned gx = 1, gy = 1, gz = 1, bx = 1, by = 1, bz = 1;
      if (!(iss >> gx >> gy >> gz >> bx >> by >> bz)) {
        std::cerr << "invalid launch at line " << line_no << "\n";
        return false;
      }
      spec.has_launch = true;
      spec.launch.grid = dim3(gx, gy, gz);
      spec.launch.block = dim3(bx, by, bz);
    } else if (tag == "buffer") {
      size_t index = 0;
      size_t size = 0;
      if (!(iss >> index >> size)) {
        std::cerr << "invalid buffer at line " << line_no << "\n";
        return false;
      }
      spec.buffer_sizes[index] = size;
    } else if (tag == "value") {
      size_t index = 0;
      std::string kind;
      if (!(iss >> index >> kind)) {
        std::cerr << "invalid value at line " << line_no << "\n";
        return false;
      }
      ValueOverride ov;
      if (kind == "int") {
        long long value = 0;
        if (!(iss >> value)) {
          std::cerr << "invalid value int at line " << line_no << "\n";
          return false;
        }
        ov.kind = ValueOverride::Kind::kInt;
        ov.int_value = static_cast<uint64_t>(value);
      } else if (kind == "hex") {
        std::string hex;
        if (!(iss >> hex)) {
          std::cerr << "invalid value hex at line " << line_no << "\n";
          return false;
        }
        ov.kind = ValueOverride::Kind::kHex;
        if (!parse_hex_bytes(hex, ov.bytes)) {
          std::cerr << "invalid hex bytes at line " << line_no << "\n";
          return false;
        }
      } else if (kind == "bytes") {
        ov.kind = ValueOverride::Kind::kBytes;
        int byte_val = 0;
        while (iss >> byte_val) {
          if (byte_val < 0 || byte_val > 255) {
            std::cerr << "invalid byte value at line " << line_no << "\n";
            return false;
          }
          ov.bytes.push_back(static_cast<uint8_t>(byte_val));
        }
        if (ov.bytes.empty()) {
          std::cerr << "empty bytes at line " << line_no << "\n";
          return false;
        }
      } else {
        std::cerr << "unknown value kind at line " << line_no << "\n";
        return false;
      }
      spec.values[index] = std::move(ov);
    } else {
      std::cerr << "unknown input spec tag at line " << line_no << "\n";
      return false;
    }
  }
  return true;
}

static void fill_random(std::vector<uint8_t> &data, std::mt19937 &rng) {
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : data)
    b = static_cast<uint8_t>(dist(rng));
}

static bool apply_value_override(const ArgSpec &arg, size_t index,
                                 const InputSpec &spec,
                                 std::vector<uint8_t> &data,
                                 bool &had_error) {
  (void)arg;
  auto it = spec.values.find(index);
  if (it == spec.values.end()) {
    return false;
  }
  const ValueOverride &ov = it->second;
  if (ov.kind == ValueOverride::Kind::kInt) {
    uint64_t value = ov.int_value;
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(value & 0xFF);
      value >>= 8;
    }
    return true;
  }
  const std::vector<uint8_t> &bytes = ov.bytes;
  if (bytes.size() > data.size()) {
    std::cerr << "value override too large for arg " << index << "\n";
    had_error = true;
    return false;
  }
  std::fill(data.begin(), data.end(), 0);
  std::copy(bytes.begin(), bytes.end(), data.begin());
  return true;
}

static bool run_kernel(hipFunction_t func, const std::vector<ArgSpec> &args,
                       std::vector<BufferArg> &buffers,
                       const std::vector<std::vector<uint8_t>> &by_value,
                       const std::vector<void *> &param_values,
                       const LaunchDims &launch) {
  std::vector<void *> params = param_values;
  size_t buffer_index = 0;

  for (const auto &arg : args) {
    if (arg.kind != "global_buffer")
      continue;
    BufferArg &buf = buffers[buffer_index++];
    if (hipMemcpy(buf.device_ptr, buf.init.data(), buf.size,
                  hipMemcpyHostToDevice) != hipSuccess) {
      return false;
    }
  }

  hipError_t err = hipModuleLaunchKernel(
      func, launch.grid.x, launch.grid.y, launch.grid.z, launch.block.x,
      launch.block.y, launch.block.z, 0, nullptr, params.data(), nullptr);
  if (err != hipSuccess) {
    return false;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    return false;
  }

  buffer_index = 0;
  for (const auto &arg : args) {
    if (arg.kind != "global_buffer")
      continue;
    BufferArg &buf = buffers[buffer_index++];
    if (hipMemcpy(buf.out_a.data(), buf.device_ptr, buf.size,
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return false;
    }
  }

  (void)by_value;
  return true;
}

static bool run_kernel_b(hipFunction_t func, const std::vector<ArgSpec> &args,
                         std::vector<BufferArg> &buffers,
                         const std::vector<std::vector<uint8_t>> &by_value,
                         const std::vector<void *> &param_values,
                         const LaunchDims &launch) {
  std::vector<void *> params = param_values;
  size_t buffer_index = 0;

  for (const auto &arg : args) {
    if (arg.kind != "global_buffer")
      continue;
    BufferArg &buf = buffers[buffer_index++];
    if (hipMemcpy(buf.device_ptr, buf.init.data(), buf.size,
                  hipMemcpyHostToDevice) != hipSuccess) {
      return false;
    }
  }

  hipError_t err = hipModuleLaunchKernel(
      func, launch.grid.x, launch.grid.y, launch.grid.z, launch.block.x,
      launch.block.y, launch.block.z, 0, nullptr, params.data(), nullptr);
  if (err != hipSuccess) {
    return false;
  }

  if (hipDeviceSynchronize() != hipSuccess) {
    return false;
  }

  buffer_index = 0;
  for (const auto &arg : args) {
    if (arg.kind != "global_buffer")
      continue;
    BufferArg &buf = buffers[buffer_index++];
    if (hipMemcpy(buf.out_b.data(), buf.device_ptr, buf.size,
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return false;
    }
  }

  (void)by_value;
  return true;
}

int main(int argc, char **argv) {
  std::string hsaco_a;
  std::string hsaco_b;
  std::string spec_path;
  std::string input_spec_path;
  size_t buffer_size = 4096;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--hsaco-a" && i + 1 < argc) {
      hsaco_a = argv[++i];
    } else if (arg == "--hsaco-b" && i + 1 < argc) {
      hsaco_b = argv[++i];
    } else if (arg == "--spec" && i + 1 < argc) {
      spec_path = argv[++i];
    } else if (arg == "--input-spec" && i + 1 < argc) {
      input_spec_path = argv[++i];
    } else if (arg == "--buffer-size" && i + 1 < argc) {
      buffer_size = static_cast<size_t>(std::stoul(argv[++i]));
    }
  }

  if (hsaco_a.empty() || hsaco_b.empty() || spec_path.empty()) {
    std::cerr << "usage: hip_runner --hsaco-a <hsaco> --hsaco-b <hsaco> "
                 "--spec <spec> [--buffer-size N] [--input-spec path]\n";
    return 2;
  }

  std::string kernel;
  std::vector<ArgSpec> args;
  if (!load_spec(spec_path, kernel, args)) {
    std::cerr << "failed to read spec\n";
    return 2;
  }

  hipModule_t mod_a = nullptr;
  hipModule_t mod_b = nullptr;
  if (hipModuleLoad(&mod_a, hsaco_a.c_str()) != hipSuccess ||
      hipModuleLoad(&mod_b, hsaco_b.c_str()) != hipSuccess) {
    std::cerr << "hipModuleLoad failed\n";
    return 1;
  }

  hipFunction_t func_a = nullptr;
  hipFunction_t func_b = nullptr;
  if (hipModuleGetFunction(&func_a, mod_a, kernel.c_str()) != hipSuccess ||
      hipModuleGetFunction(&func_b, mod_b, kernel.c_str()) != hipSuccess) {
    std::cerr << "hipModuleGetFunction failed\n";
    return 1;
  }

  InputSpec input_spec;
  if (!input_spec_path.empty()) {
    if (!parse_input_spec(input_spec_path, input_spec)) {
      std::cerr << "failed to read input spec\n";
      return 2;
    }
  }
  std::mt19937 rng(input_spec.has_seed ? input_spec.seed : 12345);
  LaunchDims launch = input_spec.has_launch ? input_spec.launch : LaunchDims{};
  std::vector<BufferArg> buffers;
  std::vector<std::vector<uint8_t>> by_value;
  std::vector<void *> param_values;
  std::vector<void *> device_ptrs;

  size_t ptr_arg_count = 0;
  size_t by_value_count = 0;
  for (const auto &arg : args) {
    if (arg.kind == "global_buffer")
      ++ptr_arg_count;
    else if (arg.kind == "by_value" || arg.kind == "value")
      ++by_value_count;
  }
  device_ptrs.reserve(ptr_arg_count);
  by_value.reserve(by_value_count);

  for (size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
    const auto &arg = args[arg_index];
    if (arg.kind == "global_buffer") {
      BufferArg buf;
      auto size_it = input_spec.buffer_sizes.find(arg_index);
      buf.size = size_it == input_spec.buffer_sizes.end() ? buffer_size
                                                          : size_it->second;
      buf.init.resize(buf.size);
      buf.out_a.resize(buf.size);
      buf.out_b.resize(buf.size);
      fill_random(buf.init, rng);
      if (hipMalloc(&buf.device_ptr, buffer_size) != hipSuccess) {
        std::cerr << "hipMalloc failed\n";
        return 1;
      }
      buffers.push_back(std::move(buf));
      device_ptrs.push_back(buffers.back().device_ptr);
      param_values.push_back(&device_ptrs.back());
    } else if (arg.kind == "by_value" || arg.kind == "value") {
      std::vector<uint8_t> data(arg.size, 0);
      bool override_error = false;
      if (!apply_value_override(arg, arg_index, input_spec, data,
                                override_error)) {
        if (override_error) {
          return 1;
        }
        fill_random(data, rng);
      }
      by_value.push_back(std::move(data));
      param_values.push_back(by_value.back().data());
    } else {
      std::cerr << "unsupported arg kind: " << arg.kind << "\n";
      return 1;
    }
  }

  if (!run_kernel(func_a, args, buffers, by_value, param_values, launch)) {
    std::cerr << "kernel A failed\n";
    return 1;
  }

  if (!run_kernel_b(func_b, args, buffers, by_value, param_values, launch)) {
    std::cerr << "kernel B failed\n";
    return 1;
  }

  for (const auto &buf : buffers) {
    if (!std::equal(buf.out_a.begin(), buf.out_a.end(), buf.out_b.begin())) {
      std::cerr << "output mismatch\n";
      return 1;
    }
  }

  for (auto &buf : buffers) {
    hipFree(buf.device_ptr);
  }

  hipModuleUnload(mod_a);
  hipModuleUnload(mod_b);
  return 0;
}
