#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static void die(const char *msg, VkResult res) {
  if (res == VK_SUCCESS) {
    std::fprintf(stderr, "error: %s\n", msg);
  } else {
    std::fprintf(stderr, "error: %s (VkResult=%d)\n", msg, res);
  }
  std::exit(1);
}

static std::vector<uint32_t> read_spirv(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "error: failed to open %s\n", path);
    std::exit(1);
  }
  std::streamsize size = file.tellg();
  if (size <= 0 || (size % 4) != 0) {
    std::fprintf(stderr, "error: invalid SPIR-V size for %s\n", path);
    std::exit(1);
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint32_t> data(static_cast<size_t>(size / 4));
  if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
    std::fprintf(stderr, "error: failed to read %s\n", path);
    std::exit(1);
  }
  return data;
}

enum SpvOp : uint16_t {
  OpEntryPoint = 15,
  OpDecorate = 71,
  OpMemberDecorate = 72,
  OpTypeInt = 21,
  OpTypeFloat = 22,
  OpTypeVector = 23,
  OpTypeMatrix = 24,
  OpTypeArray = 28,
  OpTypeRuntimeArray = 29,
  OpTypeStruct = 30,
  OpTypePointer = 32,
  OpConstant = 43,
  OpVariable = 59,
};

enum SpvDecoration : uint32_t {
  DecorationBinding = 33,
  DecorationDescriptorSet = 34,
  DecorationOffset = 35,
  DecorationArrayStride = 6,
};

enum SpvStorageClass : uint32_t {
  StorageClassUniform = 2,
  StorageClassUniformConstant = 0,
  StorageClassStorageBuffer = 12,
  StorageClassPushConstant = 9,
};

struct TypeInfo {
  uint32_t size = 0;
  uint32_t elem_type_id = 0;
  uint32_t count = 0;
  std::vector<uint32_t> member_types;
  std::vector<uint32_t> member_offsets;
};

struct ReflectionData {
  std::unordered_map<uint32_t, uint32_t> constants;
  std::unordered_map<uint32_t, TypeInfo> types;
  std::unordered_map<uint32_t, uint32_t> array_stride;
  std::unordered_map<uint32_t, uint32_t> binding;
  std::unordered_map<uint32_t, uint32_t> descriptor_set;
  std::unordered_map<uint32_t, std::vector<uint32_t>> member_offsets;
  std::unordered_map<uint32_t, uint32_t> storage_class;
  std::unordered_map<uint32_t, uint32_t> var_type;
  uint32_t push_constant_size = 0;
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, VkDescriptorType>>
      set_bindings;
};

static uint32_t type_size(const ReflectionData &refl, uint32_t type_id);

static uint32_t struct_size(const ReflectionData &refl, uint32_t type_id) {
  auto it = refl.types.find(type_id);
  if (it == refl.types.end()) {
    return 0;
  }
  const TypeInfo &t = it->second;
  if (t.member_types.empty()) {
    return 0;
  }
  uint32_t max_end = 0;
  for (size_t i = 0; i < t.member_types.size(); i++) {
    uint32_t offset = 0;
    if (i < t.member_offsets.size()) {
      offset = t.member_offsets[i];
    }
    uint32_t msize = type_size(refl, t.member_types[i]);
    uint32_t end = offset + msize;
    if (end > max_end) {
      max_end = end;
    }
  }
  return max_end;
}

static uint32_t type_size(const ReflectionData &refl, uint32_t type_id) {
  auto it = refl.types.find(type_id);
  if (it == refl.types.end()) {
    return 0;
  }
  const TypeInfo &t = it->second;
  if (t.size > 0) {
    return t.size;
  }
  if (!t.member_types.empty()) {
    return struct_size(refl, type_id);
  }
  if (t.elem_type_id != 0 && t.count != 0) {
    uint32_t elem_size = type_size(refl, t.elem_type_id);
    auto stride_it = refl.array_stride.find(type_id);
    uint32_t stride = (stride_it != refl.array_stride.end())
                          ? stride_it->second
                          : elem_size;
    return stride * t.count;
  }
  return 0;
}

static void reflect_spirv(const std::vector<uint32_t> &code,
                          ReflectionData *refl) {
  if (code.size() < 5) {
    return;
  }
  uint32_t idx = 5;
  while (idx < code.size()) {
    uint32_t word = code[idx];
    uint16_t op = static_cast<uint16_t>(word & 0xFFFF);
    uint16_t count = static_cast<uint16_t>(word >> 16);
    if (count == 0 || (idx + count) > code.size()) {
      return;
    }
    const uint32_t *inst = &code[idx];
    switch (op) {
    case OpDecorate: {
      uint32_t target_id = inst[1];
      uint32_t decoration = inst[2];
      if (decoration == DecorationBinding && count > 3) {
        refl->binding[target_id] = inst[3];
      } else if (decoration == DecorationDescriptorSet && count > 3) {
        refl->descriptor_set[target_id] = inst[3];
      } else if (decoration == DecorationArrayStride && count > 3) {
        refl->array_stride[target_id] = inst[3];
      }
      break;
    }
    case OpMemberDecorate: {
      uint32_t target_id = inst[1];
      uint32_t member = inst[2];
      uint32_t decoration = inst[3];
      if (decoration == DecorationOffset && count > 4) {
        auto &vec = refl->member_offsets[target_id];
        if (vec.size() <= member) {
          vec.resize(member + 1, 0);
        }
        vec[member] = inst[4];
      }
      break;
    }
    case OpTypeInt: {
      uint32_t result_id = inst[1];
      uint32_t width = inst[2];
      refl->types[result_id].size = width / 8;
      break;
    }
    case OpTypeFloat: {
      uint32_t result_id = inst[1];
      uint32_t width = inst[2];
      refl->types[result_id].size = width / 8;
      break;
    }
    case OpTypeVector: {
      uint32_t result_id = inst[1];
      uint32_t elem_type = inst[2];
      uint32_t count_val = inst[3];
      TypeInfo &t = refl->types[result_id];
      t.elem_type_id = elem_type;
      t.count = count_val;
      break;
    }
    case OpTypeMatrix: {
      uint32_t result_id = inst[1];
      uint32_t column_type = inst[2];
      uint32_t count_val = inst[3];
      TypeInfo &t = refl->types[result_id];
      t.elem_type_id = column_type;
      t.count = count_val;
      break;
    }
    case OpTypeArray: {
      uint32_t result_id = inst[1];
      uint32_t elem_type = inst[2];
      uint32_t len_id = inst[3];
      uint32_t len = 0;
      auto it = refl->constants.find(len_id);
      if (it != refl->constants.end()) {
        len = it->second;
      }
      TypeInfo &t = refl->types[result_id];
      t.elem_type_id = elem_type;
      t.count = len;
      break;
    }
    case OpTypeRuntimeArray: {
      uint32_t result_id = inst[1];
      uint32_t elem_type = inst[2];
      TypeInfo &t = refl->types[result_id];
      t.elem_type_id = elem_type;
      t.count = 0;
      break;
    }
    case OpTypeStruct: {
      uint32_t result_id = inst[1];
      TypeInfo &t = refl->types[result_id];
      t.member_types.clear();
      for (uint16_t i = 2; i < count; i++) {
        t.member_types.push_back(inst[i]);
      }
      auto mo = refl->member_offsets.find(result_id);
      if (mo != refl->member_offsets.end()) {
        t.member_offsets = mo->second;
      }
      break;
    }
    case OpTypePointer: {
      uint32_t result_id = inst[1];
      uint32_t storage = inst[2];
      uint32_t elem_type = inst[3];
      TypeInfo &t = refl->types[result_id];
      t.elem_type_id = elem_type;
      refl->storage_class[result_id] = storage;
      break;
    }
    case OpConstant: {
      uint32_t result_id = inst[2];
      uint32_t value = inst[3];
      refl->constants[result_id] = value;
      break;
    }
    case OpVariable: {
      uint32_t result_type = inst[1];
      uint32_t result_id = inst[2];
      uint32_t storage = inst[3];
      refl->storage_class[result_id] = storage;
      refl->var_type[result_id] = result_type;
      break;
    }
    default:
      break;
    }
    idx += count;
  }

  for (const auto &kv : refl->member_offsets) {
    auto it = refl->types.find(kv.first);
    if (it != refl->types.end()) {
      it->second.member_offsets = kv.second;
    }
  }

  for (const auto &kv : refl->var_type) {
    uint32_t var_id = kv.first;
    uint32_t type_id = kv.second;
    auto sc_it = refl->storage_class.find(var_id);
    if (sc_it == refl->storage_class.end()) {
      continue;
    }
    uint32_t storage = sc_it->second;
    if (storage == StorageClassPushConstant) {
      uint32_t struct_id = refl->types[type_id].elem_type_id;
      uint32_t size = struct_size(*refl, struct_id);
      if (size > refl->push_constant_size) {
        refl->push_constant_size = size;
      }
      continue;
    }
    if (storage != StorageClassUniform &&
        storage != StorageClassStorageBuffer &&
        storage != StorageClassUniformConstant) {
      continue;
    }
    uint32_t set = 0;
    uint32_t bind = 0;
    auto set_it = refl->descriptor_set.find(var_id);
    if (set_it != refl->descriptor_set.end()) {
      set = set_it->second;
    }
    auto bind_it = refl->binding.find(var_id);
    if (bind_it != refl->binding.end()) {
      bind = bind_it->second;
    }
    VkDescriptorType dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (storage == StorageClassUniform) {
      dtype = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
    refl->set_bindings[set][bind] = dtype;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <spv_path> [entry] [num_bindings] "
                 "[push_constant_bytes]\n"
                 "  num_bindings=0 enables reflection for set 0\n"
                 "  push_constant_bytes=0 enables reflection\n",
                 argv[0]);
    return 1;
  }

  const char *spv_path = argv[1];
  const char *entry = (argc > 2) ? argv[2] : "pc_cmpflx_launch";
  uint32_t num_bindings = (argc > 3) ? static_cast<uint32_t>(std::atoi(argv[3]))
                                     : 0;
  uint32_t push_constant_bytes =
      (argc > 4) ? static_cast<uint32_t>(std::atoi(argv[4])) : 0;

  std::vector<uint32_t> code = read_spirv(spv_path);
  ReflectionData refl;
  reflect_spirv(code, &refl);
  if (num_bindings == 0) {
    if (!refl.set_bindings.empty()) {
      uint32_t max_bind = 0;
      auto it = refl.set_bindings.find(0);
      if (it != refl.set_bindings.end()) {
        for (const auto &kv : it->second) {
          if (kv.first > max_bind) {
            max_bind = kv.first;
          }
        }
      }
      num_bindings = max_bind + 1;
    } else {
      num_bindings = 1;
    }
  }
  if (push_constant_bytes == 0) {
    push_constant_bytes = refl.push_constant_size;
  }
  std::printf("reflection: sets=%zu, push_constants=%u bytes\n",
              refl.set_bindings.size(), push_constant_bytes);

  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "vk_aco_compile";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "none";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo inst_info = {};
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  inst_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  VkResult res = vkCreateInstance(&inst_info, nullptr, &instance);
  if (res != VK_SUCCESS) {
    die("vkCreateInstance failed", res);
  }

  uint32_t phys_count = 0;
  res = vkEnumeratePhysicalDevices(instance, &phys_count, nullptr);
  if (res != VK_SUCCESS || phys_count == 0) {
    die("vkEnumeratePhysicalDevices failed", res);
  }

  std::vector<VkPhysicalDevice> phys(phys_count);
  res = vkEnumeratePhysicalDevices(instance, &phys_count, phys.data());
  if (res != VK_SUCCESS || phys_count == 0) {
    die("vkEnumeratePhysicalDevices failed", res);
  }

  VkPhysicalDevice physical_device = phys[0];

  uint32_t qf_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, nullptr);
  if (qf_count == 0) {
    die("no queue families found", VK_SUCCESS);
  }

  std::vector<VkQueueFamilyProperties> qf_props(qf_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count,
                                           qf_props.data());

  uint32_t compute_qf = UINT32_MAX;
  for (uint32_t i = 0; i < qf_count; i++) {
    if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_qf = i;
      break;
    }
  }
  if (compute_qf == UINT32_MAX) {
    die("no compute queue family found", VK_SUCCESS);
  }

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo q_info = {};
  q_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  q_info.queueFamilyIndex = compute_qf;
  q_info.queueCount = 1;
  q_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo dev_info = {};
  dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dev_info.queueCreateInfoCount = 1;
  dev_info.pQueueCreateInfos = &q_info;

  VkDevice device = VK_NULL_HANDLE;
  res = vkCreateDevice(physical_device, &dev_info, nullptr, &device);
  if (res != VK_SUCCESS) {
    die("vkCreateDevice failed", res);
  }

  VkShaderModuleCreateInfo sm_info = {};
  sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  sm_info.codeSize = code.size() * sizeof(uint32_t);
  sm_info.pCode = code.data();

  VkShaderModule shader = VK_NULL_HANDLE;
  res = vkCreateShaderModule(device, &sm_info, nullptr, &shader);
  if (res != VK_SUCCESS) {
    die("vkCreateShaderModule failed", res);
  }

  std::vector<VkDescriptorSetLayout> set_layouts;
  if (!refl.set_bindings.empty()) {
    uint32_t max_set = 0;
    for (const auto &set_kv : refl.set_bindings) {
      if (set_kv.first > max_set) {
        max_set = set_kv.first;
      }
    }
    set_layouts.resize(max_set + 1, VK_NULL_HANDLE);
    for (uint32_t set = 0; set <= max_set; set++) {
      std::vector<VkDescriptorSetLayoutBinding> bindings;
      auto it = refl.set_bindings.find(set);
      if (it != refl.set_bindings.end()) {
        for (const auto &kv : it->second) {
          VkDescriptorSetLayoutBinding b = {};
          b.binding = kv.first;
          b.descriptorType = kv.second;
          b.descriptorCount = 1;
          b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
          bindings.push_back(b);
        }
      }
      if (bindings.empty()) {
        VkDescriptorSetLayoutBinding b = {};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
      }
      VkDescriptorSetLayoutCreateInfo dsl_info = {};
      dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      dsl_info.bindingCount = static_cast<uint32_t>(bindings.size());
      dsl_info.pBindings = bindings.data();
      res = vkCreateDescriptorSetLayout(device, &dsl_info, nullptr,
                                        &set_layouts[set]);
      if (res != VK_SUCCESS) {
        die("vkCreateDescriptorSetLayout failed", res);
      }
    }
  } else {
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(num_bindings);
    for (uint32_t i = 0; i < num_bindings; i++) {
      VkDescriptorSetLayoutBinding b = {};
      b.binding = i;
      b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b.descriptorCount = 1;
      b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bindings.push_back(b);
    }
    VkDescriptorSetLayoutCreateInfo dsl_info = {};
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = static_cast<uint32_t>(bindings.size());
    dsl_info.pBindings = bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    res = vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &dsl);
    if (res != VK_SUCCESS) {
      die("vkCreateDescriptorSetLayout failed", res);
    }
    set_layouts.push_back(dsl);
  }

  VkPushConstantRange pc_range = {};
  pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc_range.offset = 0;
  pc_range.size = push_constant_bytes;

  VkPipelineLayoutCreateInfo pl_info = {};
  pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pl_info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
  pl_info.pSetLayouts = set_layouts.data();
  if (push_constant_bytes > 0) {
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &pc_range;
  }

  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  res = vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout);
  if (res != VK_SUCCESS) {
    die("vkCreatePipelineLayout failed", res);
  }

  VkPipelineShaderStageCreateInfo stage = {};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = entry;

  VkComputePipelineCreateInfo cp_info = {};
  cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cp_info.stage = stage;
  cp_info.layout = pipeline_layout;

  VkPipeline pipeline = VK_NULL_HANDLE;
  res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr,
                                 &pipeline);
  if (res != VK_SUCCESS) {
    die("vkCreateComputePipelines failed", res);
  }

  std::printf("pipeline created successfully\n");

  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  for (VkDescriptorSetLayout dsl : set_layouts) {
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  }
  vkDestroyShaderModule(device, shader, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);

  return 0;
}
