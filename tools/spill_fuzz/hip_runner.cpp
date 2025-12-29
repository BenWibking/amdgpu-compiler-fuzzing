// Simple HIP runner for differential HSACO execution.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
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

static void fill_random(std::vector<uint8_t> &data, std::mt19937 &rng) {
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : data)
    b = static_cast<uint8_t>(dist(rng));
}

static bool run_kernel(hipFunction_t func, const std::vector<ArgSpec> &args,
                       std::vector<BufferArg> &buffers,
                       const std::vector<std::vector<uint8_t>> &by_value,
                       const std::vector<void *> &param_values) {
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
      func, 1, 1, 1, 1, 1, 1, 0, nullptr, params.data(), nullptr);
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
                         const std::vector<void *> &param_values) {
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
      func, 1, 1, 1, 1, 1, 1, 0, nullptr, params.data(), nullptr);
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
  size_t buffer_size = 4096;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--hsaco-a" && i + 1 < argc) {
      hsaco_a = argv[++i];
    } else if (arg == "--hsaco-b" && i + 1 < argc) {
      hsaco_b = argv[++i];
    } else if (arg == "--spec" && i + 1 < argc) {
      spec_path = argv[++i];
    } else if (arg == "--buffer-size" && i + 1 < argc) {
      buffer_size = static_cast<size_t>(std::stoul(argv[++i]));
    }
  }

  if (hsaco_a.empty() || hsaco_b.empty() || spec_path.empty()) {
    std::cerr << "usage: hip_runner --hsaco-a <hsaco> --hsaco-b <hsaco> "
                 "--spec <spec> [--buffer-size N]\n";
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

  std::mt19937 rng(12345);
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

  for (const auto &arg : args) {
    if (arg.kind == "global_buffer") {
      BufferArg buf;
      buf.size = buffer_size;
      buf.init.resize(buffer_size);
      buf.out_a.resize(buffer_size);
      buf.out_b.resize(buffer_size);
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
      fill_random(data, rng);
      by_value.push_back(std::move(data));
      param_values.push_back(by_value.back().data());
    } else {
      std::cerr << "unsupported arg kind: " << arg.kind << "\n";
      return 1;
    }
  }

  if (!run_kernel(func_a, args, buffers, by_value, param_values)) {
    std::cerr << "kernel A failed\n";
    return 1;
  }

  if (!run_kernel_b(func_b, args, buffers, by_value, param_values)) {
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
