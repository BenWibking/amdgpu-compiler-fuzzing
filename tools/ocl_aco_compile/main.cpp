#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void die(const char *msg, cl_int err) {
  if (err == CL_SUCCESS) {
    std::fprintf(stderr, "error: %s\n", msg);
  } else {
    std::fprintf(stderr, "error: %s (cl_int=%d)\n", msg, err);
  }
  std::exit(1);
}

static std::string read_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "error: failed to open %s\n", path);
    std::exit(1);
  }
  std::streamsize size = file.tellg();
  if (size <= 0) {
    std::fprintf(stderr, "error: empty source file: %s\n", path);
    std::exit(1);
  }
  file.seekg(0, std::ios::beg);
  std::string data(static_cast<size_t>(size), '\0');
  if (!file.read(data.data(), size)) {
    std::fprintf(stderr, "error: failed to read %s\n", path);
    std::exit(1);
  }
  return data;
}

static std::string get_platform_info_str(cl_platform_id platform,
                                         cl_platform_info param) {
  size_t size = 0;
  cl_int err = clGetPlatformInfo(platform, param, 0, nullptr, &size);
  if (err != CL_SUCCESS || size == 0) {
    return {};
  }
  std::string out(size, '\0');
  err = clGetPlatformInfo(platform, param, size, out.data(), nullptr);
  if (err != CL_SUCCESS) {
    return {};
  }
  if (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
  return out;
}

static std::string get_device_info_str(cl_device_id device,
                                       cl_device_info param) {
  size_t size = 0;
  cl_int err = clGetDeviceInfo(device, param, 0, nullptr, &size);
  if (err != CL_SUCCESS || size == 0) {
    return {};
  }
  std::string out(size, '\0');
  err = clGetDeviceInfo(device, param, size, out.data(), nullptr);
  if (err != CL_SUCCESS) {
    return {};
  }
  if (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
  return out;
}

static size_t parse_index_env(const char *env_name, size_t max_index) {
  const char *value = std::getenv(env_name);
  if (!value || *value == '\0') {
    return 0;
  }
  char *end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (!end || *end != '\0') {
    std::fprintf(stderr, "error: invalid %s: %s\n", env_name, value);
    std::exit(1);
  }
  if (parsed >= max_index) {
    std::fprintf(stderr,
                 "error: %s index %lu out of range (max %zu)\n",
                 env_name, parsed, max_index - 1);
    std::exit(1);
  }
  return static_cast<size_t>(parsed);
}

static void print_build_log(cl_program program, cl_device_id device) {
  size_t log_size = 0;
  cl_int err =
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_size);
  if (err != CL_SUCCESS || log_size == 0) {
    return;
  }
  std::string log(log_size, '\0');
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                              log.data(), nullptr);
  if (err != CL_SUCCESS) {
    return;
  }
  if (!log.empty() && log.back() == '\0') {
    log.pop_back();
  }
  if (!log.empty()) {
    std::fprintf(stderr, "build log:\n%s\n", log.c_str());
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(
        stderr,
        "usage: %s <kernel_src> [entry] [build_opts]\n"
        "env: OCL_PLATFORM_INDEX, OCL_DEVICE_INDEX\n",
        argv[0]);
    return 2;
  }

  const char *src_path = argv[1];
  const char *entry = (argc > 2) ? argv[2] : "pc_cmpflx_launch";
  const char *build_opts = (argc > 3) ? argv[3] : "-cl-std=CL1.2";

  std::string source = read_file(src_path);
  const char *src_ptr = source.c_str();
  size_t src_len = source.size();

  cl_uint platform_count = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
  if (err != CL_SUCCESS || platform_count == 0) {
    die("no OpenCL platforms available", err);
  }
  std::vector<cl_platform_id> platforms(platform_count);
  err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    die("failed to enumerate OpenCL platforms", err);
  }
  size_t platform_index = parse_index_env("OCL_PLATFORM_INDEX", platform_count);
  cl_platform_id platform = platforms[platform_index];

  cl_uint device_count = 0;
  cl_device_type device_type = CL_DEVICE_TYPE_GPU;
  err = clGetDeviceIDs(platform, device_type, 0, nullptr, &device_count);
  if (err == CL_DEVICE_NOT_FOUND || device_count == 0) {
    device_type = CL_DEVICE_TYPE_ALL;
    err = clGetDeviceIDs(platform, device_type, 0, nullptr, &device_count);
  }
  if (err != CL_SUCCESS || device_count == 0) {
    die("no OpenCL devices available", err);
  }
  std::vector<cl_device_id> devices(device_count);
  err = clGetDeviceIDs(platform, device_type, device_count, devices.data(),
                       nullptr);
  if (err != CL_SUCCESS) {
    die("failed to enumerate OpenCL devices", err);
  }
  size_t device_index = parse_index_env("OCL_DEVICE_INDEX", device_count);
  cl_device_id device = devices[device_index];

  std::string platform_name =
      get_platform_info_str(platform, CL_PLATFORM_NAME);
  std::string platform_vendor =
      get_platform_info_str(platform, CL_PLATFORM_VENDOR);
  std::string device_name = get_device_info_str(device, CL_DEVICE_NAME);
  std::string device_version = get_device_info_str(device, CL_DEVICE_VERSION);

  std::fprintf(stdout, "platform[%zu]: %s (%s)\n", platform_index,
               platform_name.empty() ? "unknown" : platform_name.c_str(),
               platform_vendor.empty() ? "unknown" : platform_vendor.c_str());
  std::fprintf(stdout, "device[%zu]: %s (%s)\n", device_index,
               device_name.empty() ? "unknown" : device_name.c_str(),
               device_version.empty() ? "unknown" : device_version.c_str());
  if (platform_name.find("Rusticl") == std::string::npos) {
    std::fprintf(stderr,
                 "warning: OpenCL platform is not Rusticl (got: %s)\n",
                 platform_name.empty() ? "unknown" : platform_name.c_str());
  }

  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (!context || err != CL_SUCCESS) {
    die("clCreateContext failed", err);
  }

  cl_program program =
      clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
  if (!program || err != CL_SUCCESS) {
    die("clCreateProgramWithSource failed", err);
  }

  err = clBuildProgram(program, 1, &device, build_opts, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    print_build_log(program, device);
    die("clBuildProgram failed", err);
  }

  cl_kernel kernel = clCreateKernel(program, entry, &err);
  if (!kernel || err != CL_SUCCESS) {
    print_build_log(program, device);
    die("clCreateKernel failed", err);
  }

  std::fprintf(stdout, "build ok: %s\n", entry);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(context);
  return 0;
}
