# Vulkan Compute: Modern Graphics API for Computation

*Welcome to the seventh installment of our GPU programming series! In this article, we'll explore Vulkan Compute, a modern approach to GPU computing that's part of the Vulkan graphics API. While Vulkan is primarily known for graphics rendering, its compute capabilities offer a powerful alternative to CUDA and OpenCL for general-purpose GPU programming.*

## Introduction to Vulkan Compute

Vulkan is a low-level, cross-platform 3D graphics and compute API developed by the Khronos Group (the same organization behind OpenCL). Released in 2016, Vulkan was designed as a modern successor to OpenGL, offering more direct control over the GPU and better multi-threading support.

While Vulkan is primarily known for graphics rendering, it includes robust compute capabilities that allow developers to leverage the parallel processing power of GPUs for non-graphics tasks.

### Key Features of Vulkan Compute

- **Low-level control**: Direct access to GPU resources with minimal driver overhead
- **Cross-platform support**: Works on Windows, Linux, Android, and more
- **Explicit synchronization**: Fine-grained control over resource dependencies
- **Integration with graphics**: Seamless interoperability between compute and graphics operations
- **Modern design**: Built for contemporary GPU architectures and multi-core CPUs

### When to Use Vulkan Compute

Vulkan Compute is particularly well-suited for:

- Applications that combine graphics and compute workloads
- Scenarios requiring fine-grained control over GPU execution
- Cross-platform applications targeting multiple GPU vendors
- Performance-critical applications that need to minimize driver overhead

## Vulkan Compute Architecture

Understanding Vulkan's architecture is essential for effective compute programming. Unlike CUDA or OpenCL, Vulkan uses a more explicit programming model that gives developers greater control but also requires more code.

### Core Concepts

- **Physical and Logical Devices**: Physical devices represent actual GPUs, while logical devices are interfaces to those GPUs
- **Command Buffers**: Pre-recorded sequences of commands that are submitted to queues
- **Queues and Queue Families**: Different types of queues (graphics, compute, transfer) for different operations
- **Pipelines**: Fixed configurations for how work will be processed
- **Descriptors**: References to resources (buffers, images) used by shaders
- **Shaders**: SPIR-V bytecode that runs on the GPU

## Setting Up Compute Pipelines

Creating a compute pipeline in Vulkan involves several steps:

### 1. Initialize Vulkan

```cpp
// Create Vulkan instance
VkInstance instance;
VkInstanceCreateInfo instanceInfo = {};
instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
// Fill in application info, enabled layers, etc.
vkCreateInstance(&instanceInfo, nullptr, &instance);

// Select physical device (GPU)
VkPhysicalDevice physicalDevice;
uint32_t deviceCount = 0;
vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
std::vector<VkPhysicalDevice> devices(deviceCount);
vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
physicalDevice = devices[0]; // Select first device or implement proper selection

// Create logical device with compute queue
VkDevice device;
VkDeviceQueueCreateInfo queueInfo = {};
queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
// Find compute queue family index
float queuePriority = 1.0f;
queueInfo.queueCount = 1;
queueInfo.pQueuePriorities = &queuePriority;

VkDeviceCreateInfo deviceInfo = {};
deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
deviceInfo.queueCreateInfoCount = 1;
deviceInfo.pQueueCreateInfos = &queueInfo;
// Enable device features and extensions as needed
vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);

// Get compute queue
VkQueue computeQueue;
vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
```

### 2. Create Compute Shader

Vulkan uses SPIR-V as its shader language format. You can write shaders in GLSL and compile them to SPIR-V using tools like glslangValidator:

```glsl
// compute.comp - GLSL compute shader
#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) buffer InputBuffer {
    float data[];
} inputBuffer;

layout(binding = 1) buffer OutputBuffer {
    float data[];
} outputBuffer;

void main() {
    uint index = gl_GlobalInvocationID.x;
    outputBuffer.data[index] = inputBuffer.data[index] * 2.0;
}
```

Compile to SPIR-V:
```bash
glslangValidator -V compute.comp -o comp.spv
```

Load the SPIR-V shader in your application:

```cpp
// Load shader from file
std::vector<char> shaderCode = readFile("comp.spv");

// Create shader module
VkShaderModule shaderModule;
VkShaderModuleCreateInfo moduleInfo = {};
moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
moduleInfo.codeSize = shaderCode.size();
moduleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule);
```

### 3. Create Descriptor Set Layout and Pipeline Layout

```cpp
// Create descriptor set layout (defines shader bindings)
VkDescriptorSetLayout descriptorSetLayout;
VkDescriptorSetLayoutBinding bindings[2] = {};

// Input buffer binding
bindings[0].binding = 0;
bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
bindings[0].descriptorCount = 1;
bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

// Output buffer binding
bindings[1].binding = 1;
bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
bindings[1].descriptorCount = 1;
bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

VkDescriptorSetLayoutCreateInfo layoutInfo = {};
layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
layoutInfo.bindingCount = 2;
layoutInfo.pBindings = bindings;
vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);

// Create pipeline layout
VkPipelineLayout pipelineLayout;
VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
pipelineLayoutInfo.setLayoutCount = 1;
pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
```

### 4. Create Compute Pipeline

```cpp
// Create compute pipeline
VkComputePipelineCreateInfo pipelineInfo = {};
pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

// Shader stage
pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
pipelineInfo.stage.module = shaderModule;
pipelineInfo.stage.pName = "main"; // Entry point

pipelineInfo.layout = pipelineLayout;

VkPipeline computePipeline;
vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
```

## Memory Management in Vulkan

Vulkan's memory management is explicit and requires careful handling. This gives developers fine-grained control but also increases complexity.

### Buffer Creation and Memory Allocation

```cpp
// Create buffer
VkBuffer buffer;
VkBufferCreateInfo bufferInfo = {};
bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
bufferInfo.size = bufferSize;
bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);

// Get memory requirements
VkMemoryRequirements memRequirements;
vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

// Find suitable memory type
uint32_t memoryTypeIndex = findMemoryType(physicalDevice, 
                                         memRequirements.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

// Allocate memory
VkDeviceMemory bufferMemory;
VkMemoryAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
allocInfo.allocationSize = memRequirements.size;
allocInfo.memoryTypeIndex = memoryTypeIndex;
vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);

// Bind memory to buffer
vkBindBufferMemory(device, buffer, bufferMemory, 0);

// Map memory and copy data
void* data;
vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &data);
memcpy(data, sourceData, bufferSize);
vkUnmapMemory(device, bufferMemory);
```

### Helper Function for Memory Type Selection

```cpp
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}
```

## Descriptor Sets for Resource Binding

Descriptor sets connect buffers and other resources to shader bindings:

```cpp
// Create descriptor pool
VkDescriptorPool descriptorPool;
VkDescriptorPoolSize poolSize = {};
poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
poolSize.descriptorCount = 2; // Two storage buffers

VkDescriptorPoolCreateInfo poolInfo = {};
poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
poolInfo.poolSizeCount = 1;
poolInfo.pPoolSizes = &poolSize;
poolInfo.maxSets = 1;
vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

// Allocate descriptor set
VkDescriptorSet descriptorSet;
VkDescriptorSetAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
allocInfo.descriptorPool = descriptorPool;
allocInfo.descriptorSetCount = 1;
allocInfo.pSetLayouts = &descriptorSetLayout;
vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

// Update descriptor set with buffer info
VkDescriptorBufferInfo inputBufferInfo = {};
inputBufferInfo.buffer = inputBuffer;
inputBufferInfo.offset = 0;
inputBufferInfo.range = bufferSize;

VkDescriptorBufferInfo outputBufferInfo = {};
outputBufferInfo.buffer = outputBuffer;
outputBufferInfo.offset = 0;
outputBufferInfo.range = bufferSize;

VkWriteDescriptorSet descriptorWrites[2] = {};

descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
descriptorWrites[0].dstSet = descriptorSet;
descriptorWrites[0].dstBinding = 0;
descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
descriptorWrites[0].descriptorCount = 1;
descriptorWrites[0].pBufferInfo = &inputBufferInfo;

descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
descriptorWrites[1].dstSet = descriptorSet;
descriptorWrites[1].dstBinding = 1;
descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
descriptorWrites[1].descriptorCount = 1;
descriptorWrites[1].pBufferInfo = &outputBufferInfo;

vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);
```

## Command Buffers and Dispatch

Command buffers record operations to be executed on the GPU:

```cpp
// Create command pool
VkCommandPool commandPool;
VkCommandPoolCreateInfo poolInfo = {};
poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

// Allocate command buffer
VkCommandBuffer commandBuffer;
VkCommandBufferAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
allocInfo.commandPool = commandPool;
allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
allocInfo.commandBufferCount = 1;
vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

// Begin command buffer recording
VkCommandBufferBeginInfo beginInfo = {};
beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
vkBeginCommandBuffer(commandBuffer, &beginInfo);

// Bind pipeline
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

// Bind descriptor sets
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

// Dispatch compute work
// For a 1D array of 1024 elements with local_size_x = 16, we need 1024/16 = 64 workgroups
vkCmdDispatch(commandBuffer, 64, 1, 1);

// End command buffer recording
vkEndCommandBuffer(commandBuffer);

// Submit command buffer to queue
VkSubmitInfo submitInfo = {};
submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
submitInfo.commandBufferCount = 1;
submitInfo.pCommandBuffers = &commandBuffer;

// Create fence for synchronization
VkFence fence;
VkFenceCreateInfo fenceInfo = {};
fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
vkCreateFence(device, &fenceInfo, nullptr, &fence);

// Submit work and wait for completion
vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
```

## Integrating Compute with Graphics Workloads

One of Vulkan's strengths is the seamless integration between compute and graphics operations. This allows for efficient pipelines where compute shaders process data that is then used in rendering.

### Resource Sharing Between Compute and Graphics

```cpp
// Create a buffer that will be used by both compute and graphics pipelines
VkBufferCreateInfo bufferInfo = {};
bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
bufferInfo.size = dataSize;
bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;

// Specify which queue families will access this buffer
uint32_t queueFamilyIndices[] = {computeQueueFamilyIndex, graphicsQueueFamilyIndex};
bufferInfo.queueFamilyIndexCount = 2;
bufferInfo.pQueueFamilyIndices = queueFamilyIndices;

vkCreateBuffer(device, &bufferInfo, nullptr, &sharedBuffer);
```

### Pipeline Barriers for Synchronization

When a resource is used by both compute and graphics operations, you need to ensure proper synchronization:

```cpp
// In compute command buffer, after compute work is done
VkBufferMemoryBarrier barrier = {};
barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
barrier.srcQueueFamilyIndex = computeQueueFamilyIndex;
barrier.dstQueueFamilyIndex = graphicsQueueFamilyIndex;
barrier.buffer = sharedBuffer;
barrier.offset = 0;
barrier.size = bufferSize;

vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
    0,
    0, nullptr,
    1, &barrier,
    0, nullptr
);
```

## Complete Vulkan Compute Example

Here's a simplified but complete example of a Vulkan compute application that doubles each element in a buffer:

```cpp
#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

int main() {
    // Application constants
    const uint32_t BUFFER_ELEMENTS = 1024;
    const uint32_t BUFFER_SIZE = BUFFER_ELEMENTS * sizeof(float);
    
    // Initialize Vulkan
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Compute Example";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    VkInstance instance;
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance");
    }
    
    // Select physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice physicalDevice = devices[0]; // Just take the first device
    
    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    uint32_t computeQueueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            break;
        }
    }
    
    if (computeQueueFamilyIndex == UINT32_MAX) {
        throw std::runtime_error("Failed to find a compute queue family");
    }
    
    // Create logical device with compute queue
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    
    VkDevice device;
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }
    
    // Get compute queue
    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    
    // Create buffers
    VkBuffer inputBuffer, outputBuffer;
    VkDeviceMemory inputBufferMemory, outputBufferMemory;
    
    // Input buffer
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = BUFFER_SIZE;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &inputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create input buffer");
    }
    
    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, inputBuffer, &memRequirements);
    
    // Allocate memory
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, 
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &inputBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate input buffer memory");
    }
    
    vkBindBufferMemory(device, inputBuffer, inputBufferMemory, 0);
    
    // Output buffer (similar process)
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &outputBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer");
    }
    
    vkGetBufferMemoryRequirements(device, outputBuffer, &memRequirements);
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &outputBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate output buffer memory");
    }
    
    vkBindBufferMemory(device, outputBuffer, outputBufferMemory, 0);
    
    // Fill input buffer with data
    float* inputData;
    vkMapMemory(device, inputBufferMemory, 0, BUFFER_SIZE, 0, (void**)&inputData);
    
    for (uint32_t i = 0; i < BUFFER_ELEMENTS; i++) {
        inputData[i] = static_cast<float>(i);
    }
    
    vkUnmapMemory(device, inputBufferMemory);
    
    // Create descriptor set layout
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayoutBinding bindings[2] = {};
    
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
    
    // Create pipeline layout
    VkPipelineLayout pipelineLayout;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    
    // Load shader
    auto shaderCode = readFile("comp.spv");
    
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo moduleCreateInfo = {};
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.codeSize = shaderCode.size();
    moduleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());
    
    if (vkCreateShaderModule(device, &moduleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
    
    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName = "main";
    pipelineCreateInfo.layout = pipelineLayout;
    
    VkPipeline computePipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
    
    // Create descriptor pool
    VkDescriptorPool descriptorPool;
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;
    
    VkDescriptorPoolCreateInfo poolCreateInfo = {};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &poolSize;
    poolCreateInfo.maxSets = 1;
    
    if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
    
    // Allocate descriptor set
    VkDescriptorSet descriptorSet;
    VkDescriptorSetAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    
    if (vkAllocateDescriptorSets(device, &allocateInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }
    
    // Update descriptor set
    VkDescriptorBufferInfo inputBufferInfo = {};
    inputBufferInfo.buffer = inputBuffer;
    inputBufferInfo.offset = 0;
    inputBufferInfo.range = BUFFER_SIZE;
    
    VkDescriptorBufferInfo outputBufferInfo = {};
    outputBufferInfo.buffer = outputBuffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = BUFFER_SIZE;
    
    VkWriteDescriptorSet descriptorWrites[2] = {};
    
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &inputBufferInfo;
    
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &outputBufferInfo;
    
    vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);
    
    // Create command pool
    VkCommandPool commandPool;
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    
    if (vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
    
    // Allocate command buffer
    VkCommandBuffer commandBuffer;
    VkCommandBufferAllocateInfo commandBufferInfo = {};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferInfo.commandPool = commandPool;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferInfo.commandBufferCount = 1;
    
    if (vkAllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }
    
    // Record command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer");
    }
    
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    
    // Dispatch compute work (64 workgroups of 16 threads each = 1024 total threads)
    vkCmdDispatch(commandBuffer, BUFFER_ELEMENTS / 16, 1, 1);
    
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer");
    }
    
    // Create fence for synchronization
    VkFence fence;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }
    
    // Submit command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }
    
    // Wait for the fence to signal that compute work has finished
    if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        throw std::runtime_error("Failed to wait for fence");
    }
    
    // Read back the output buffer
    float* outputData;
    vkMapMemory(device, outputBufferMemory, 0, BUFFER_SIZE, 0, (void**)&outputData);
    
    // Verify results
    bool success = true;
    for (uint32_t i = 0; i < BUFFER_ELEMENTS; i++) {
        if (outputData[i] != inputData[i] * 2.0f) {
            std::cout << "Verification failed at index " << i << ": " 
                      << outputData[i] << " != " << (inputData[i] * 2.0f) << std::endl;
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "Computation succeeded!" << std::endl;
    }
    
    vkUnmapMemory(device, outputBufferMemory);
    
    // Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyBuffer(device, inputBuffer, nullptr);
    vkDestroyBuffer(device, outputBuffer, nullptr);
    vkFreeMemory(device, inputBufferMemory, nullptr);
    vkFreeMemory(device, outputBufferMemory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    
    return 0;
}
```

## Conclusion

Vulkan Compute offers a powerful, cross-platform solution for GPU computing with several key advantages:

1. **Fine-grained control**: Explicit memory management and synchronization allow for optimal performance tuning
2. **Cross-platform compatibility**: Works across different GPU vendors and operating systems
3. **Graphics integration**: Seamless interoperability between compute and graphics operations
4. **Modern design**: Built for contemporary hardware architectures

However, these advantages come with increased complexity compared to CUDA or OpenCL. Vulkan requires more boilerplate code and explicit management of resources, which can make it more challenging to learn and use effectively.

Vulkan Compute is particularly well-suited for applications that:
- Need to combine graphics and compute operations
- Require cross-platform compatibility
- Benefit from fine-grained control over GPU execution
- Need to minimize driver overhead

In our next article, we'll explore DirectCompute and C++ AMP, Microsoft's GPU computing solutions that integrate with the DirectX ecosystem.

## Exercises for Practice

1. **Basic Compute**: Modify the example code to implement a different computation (e.g., vector multiplication or addition).

2. **Multiple Workgroups**: Implement a reduction algorithm that works across multiple workgroups to compute the sum of all elements in an array.

3. **Graphics Integration**: Create a simple Vulkan application that uses compute shaders to generate or modify data that is then rendered using the graphics pipeline.

4. **Performance Comparison**: Implement the same algorithm in CUDA, OpenCL, and Vulkan Compute, and compare their performance and development complexity.

## Further Resources

- [Vulkan Specification](https://www.khronos.org/registry/vulkan/)
- [Vulkan Compute Examples](https://github.com/SaschaWillems/Vulkan)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [Vulkan Cookbook](https://www.packtpub.com/product/vulkan-cookbook/9781786468154)
- [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)