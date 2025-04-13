# Part 9: Performance Optimization

## Introduction

Performance optimization is a critical aspect of graphics programming. Even with correct rendering, an application can suffer from poor frame rates, stuttering, or excessive resource usage. This lesson provides a comprehensive guide to optimizing graphics applications across DirectX 12, Vulkan, and Metal, focusing on both API-specific techniques and general principles that apply to all modern graphics hardware.

## 1. API-Specific Performance Best Practices

### DirectX 12 Optimization Techniques

#### Command List Management

```cpp
// Inefficient: Creating new command allocators and lists for every frame
ComPtr<ID3D12CommandAllocator> commandAllocator;
ComPtr<ID3D12GraphicsCommandList> commandList;
device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator));
device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList));

// Efficient: Reusing command allocators and lists with a ring buffer
const int FRAME_COUNT = 3; // Triple buffering
ComPtr<ID3D12CommandAllocator> commandAllocators[FRAME_COUNT];
ComPtr<ID3D12GraphicsCommandList> commandLists[FRAME_COUNT];

// During initialization
for (int i = 0; i < FRAME_COUNT; i++) {
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocators[i]));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocators[i].Get(), nullptr, IID_PPV_ARGS(&commandLists[i]));
    commandLists[i]->Close(); // Close initially
}

// During frame rendering
int frameIndex = currentFrameIndex % FRAME_COUNT;
commandAllocators[frameIndex]->Reset();
commandLists[frameIndex]->Reset(commandAllocators[frameIndex].Get(), nullptr);
// Record commands...
commandLists[frameIndex]->Close();
commandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&commandLists[frameIndex]);
```

#### Efficient Root Signature Design

```cpp
// Inefficient: Too many root parameters
CD3DX12_ROOT_PARAMETER rootParams[20]; // Many individual parameters

// Efficient: Grouping related parameters
CD3DX12_ROOT_PARAMETER rootParams[3];

// Root constants for frequently changing data (small, fast access)
CONST UINT NUM_ROOT_CONSTANTS = 4; // e.g., time, viewport dimensions
rootParams[0].InitAsConstants(NUM_ROOT_CONSTANTS, 0, 0, D3D12_SHADER_VISIBILITY_ALL);

// Root CBV for per-object data (medium, direct pointer)
rootParams[1].InitAsConstantBufferView(0, 1, D3D12_SHADER_VISIBILITY_ALL);

// Descriptor table for textures and samplers (many resources)
CD3DX12_DESCRIPTOR_RANGE ranges[2];
ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 16, 0); // 16 textures
ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 4, 0); // 4 samplers
rootParams[2].InitAsDescriptorTable(2, ranges, D3D12_SHADER_VISIBILITY_PIXEL);
```

#### Descriptor Management

```cpp
// Inefficient: Creating descriptors for each resource every frame

// Efficient: Using descriptor heaps and caching
class DescriptorCache {
private:
    ComPtr<ID3D12DescriptorHeap> m_heap;
    UINT m_descriptorSize;
    std::unordered_map<ID3D12Resource*, UINT> m_resourceToIndex;
    UINT m_nextIndex = 0;

public:
    void Initialize(ID3D12Device* device, UINT capacity) {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.NumDescriptors = capacity;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&m_heap));
        m_descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    D3D12_GPU_DESCRIPTOR_HANDLE GetOrCreate(ID3D12Device* device, ID3D12Resource* resource) {
        // Check if descriptor already exists
        auto it = m_resourceToIndex.find(resource);
        if (it != m_resourceToIndex.end()) {
            // Return existing descriptor
            D3D12_GPU_DESCRIPTOR_HANDLE handle = m_heap->GetGPUDescriptorHandleForHeapStart();
            handle.ptr += it->second * m_descriptorSize;
            return handle;
        }

        // Create new descriptor
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = m_heap->GetCPUDescriptorHandleForHeapStart();
        cpuHandle.ptr += m_nextIndex * m_descriptorSize;

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = resource->GetDesc().Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = resource->GetDesc().MipLevels;
        device->CreateShaderResourceView(resource, &srvDesc, cpuHandle);

        // Cache the index
        m_resourceToIndex[resource] = m_nextIndex;
        
        // Return the GPU handle
        D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = m_heap->GetGPUDescriptorHandleForHeapStart();
        gpuHandle.ptr += m_nextIndex * m_descriptorSize;
        
        m_nextIndex++;
        return gpuHandle;
    }

    ID3D12DescriptorHeap* GetHeap() { return m_heap.Get(); }
};
```

### Vulkan Optimization Techniques

#### Command Buffer Management

```cpp
// Efficient: Pre-recording static command buffers
class StaticCommandBufferCache {
private:
    VkDevice m_device;
    VkCommandPool m_commandPool;
    std::unordered_map<std::string, VkCommandBuffer> m_commandBuffers;

public:
    void Initialize(VkDevice device, VkCommandPool commandPool) {
        m_device = device;
        m_commandPool = commandPool;
    }

    VkCommandBuffer GetOrCreate(const std::string& key, std::function<void(VkCommandBuffer)> recordFunc) {
        auto it = m_commandBuffers.find(key);
        if (it != m_commandBuffers.end()) {
            return it->second;
        }

        // Allocate new command buffer
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

        // Begin recording
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // Record commands
        recordFunc(commandBuffer);

        // End recording
        vkEndCommandBuffer(commandBuffer);

        // Cache the command buffer
        m_commandBuffers[key] = commandBuffer;
        return commandBuffer;
    }
};
```

#### Pipeline Caching

```cpp
// Creating and saving a pipeline cache
VkPipelineCacheCreateInfo pipelineCacheInfo = {};
pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

// Load cached data if available
std::vector<char> cacheData = loadCacheFromDisk();
if (!cacheData.empty()) {
    pipelineCacheInfo.initialDataSize = cacheData.size();
    pipelineCacheInfo.pInitialData = cacheData.data();
}

VkPipelineCache pipelineCache;
vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &pipelineCache);

// Use the cache when creating pipelines
vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline);

// Save the cache periodically or on shutdown
size_t cacheSize;
vkGetPipelineCacheData(device, pipelineCache, &cacheSize, nullptr);
std::vector<char> newCacheData(cacheSize);
vkGetPipelineCacheData(device, pipelineCache, &cacheSize, newCacheData.data());
saveCacheToDisk(newCacheData);
```

#### Descriptor Set Optimization

```cpp
// Efficient: Grouping descriptors by update frequency
enum DescriptorSetType {
    GLOBAL,     // Updated once per app (textures, samplers)
    PER_FRAME,  // Updated once per frame (camera, time)
    PER_MATERIAL, // Updated when material changes
    PER_OBJECT  // Updated for each object
};

// Create descriptor set layouts for each type
VkDescriptorSetLayout layouts[4]; // One for each type

// Allocate descriptor sets in bulk
const uint32_t MAX_OBJECTS = 1000;
std::vector<VkDescriptorSet> perObjectSets(MAX_OBJECTS);
VkDescriptorSetAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
allocInfo.descriptorPool = descriptorPool;
allocInfo.descriptorSetCount = MAX_OBJECTS;
std::vector<VkDescriptorSetLayout> setLayouts(MAX_OBJECTS, layouts[PER_OBJECT]);
allocInfo.pSetLayouts = setLayouts.data();
vkAllocateDescriptorSets(device, &allocInfo, perObjectSets.data());

// Update descriptor sets in batches
std::vector<VkWriteDescriptorSet> writes;
writes.reserve(MAX_OBJECTS);

for (uint32_t i = 0; i < MAX_OBJECTS; i++) {
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = objectBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(ObjectUBO);
    
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = perObjectSets[i];
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;
    
    writes.push_back(write);
}

vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
```

### Metal Optimization Techniques

#### Efficient Command Encoding

```swift
// Inefficient: Creating many small encoders
let commandBuffer = commandQueue.makeCommandBuffer()!

for object in objects {
    let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
    renderEncoder.setRenderPipelineState(pipelineState)
    renderEncoder.setVertexBuffer(object.vertexBuffer, offset: 0, index: 0)
    renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: object.vertexCount)
    renderEncoder.endEncoding()
}

// Efficient: Batching commands in a single encoder
let commandBuffer = commandQueue.makeCommandBuffer()!
let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

renderEncoder.setRenderPipelineState(pipelineState)

for object in objects {
    renderEncoder.setVertexBuffer(object.vertexBuffer, offset: 0, index: 0)
    renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: object.vertexCount)
}

renderEncoder.endEncoding()
```

#### Argument Buffers for Bindless Rendering

```swift
// Define argument buffer structure in shader
struct MaterialResources {
    texture2d<float> textures[16];
    sampler samplers[4];
};

// Create argument buffer
let argumentEncoder = device.makeArgumentEncoder(arguments: [
    .texture(0, access: .read),
    .sampler(0)
])

let argumentBuffer = device.makeBuffer(length: argumentEncoder.encodedLength * 16, options: .storageModeShared)!

// Populate argument buffer with multiple textures (bindless approach)
for i in 0..<textures.count {
    let offset = argumentEncoder.encodedLength * i
    argumentEncoder.setArgumentBuffer(argumentBuffer, offset: offset)
    argumentEncoder.setTexture(textures[i], index: 0)
    argumentEncoder.setSamplerState(samplers[i % samplers.count], index: 1)
}

// Use in shader with dynamic indexing
renderEncoder.setFragmentBuffer(argumentBuffer, offset: 0, index: 1)
```

#### Optimizing Storage Modes

```swift
// Choose the right storage mode for each resource type

// For resources that are written once by CPU, read many times by GPU
let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: 1024,
    height: 1024,
    mipmapped: true)
textureDescriptor.storageMode = .private // Best GPU performance
let texture = device.makeTexture(descriptor: textureDescriptor)!

// For resources frequently updated by CPU
let dynamicBufferDescriptor = MTLBufferDescriptor()
dynamicBufferDescriptor.storageMode = .shared // Direct CPU access
let dynamicBuffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

// For resources written by GPU, read by CPU (e.g., readbacks)
let readbackBufferDescriptor = MTLBufferDescriptor()
readbackBufferDescriptor.storageMode = .managed // CPU can read after GPU writes
let readbackBuffer = device.makeBuffer(length: 1024, options: .storageModeManaged)!
```

## 2. Using Hardware Efficiently

### Batching and Instancing

#### DirectX 12 Instancing

```cpp
// Inefficient: Individual draw calls
for (int i = 0; i < objectCount; i++) {
    commandList->SetGraphicsRoot32BitConstants(0, 16, &objectTransforms[i], 0);
    commandList->DrawIndexedInstanced(indexCount, 1, 0, 0, 0);
}

// Efficient: Hardware instancing
struct InstanceData {
    XMFLOAT4X4 transform;
    XMFLOAT4 color;
};

// Create and populate instance buffer
std::vector<InstanceData> instanceData(objectCount);
for (int i = 0; i < objectCount; i++) {
    instanceData[i].transform = objectTransforms[i];
    instanceData[i].color = objectColors[i];
}

// Upload to GPU buffer
UpdateBuffer(instanceBuffer.Get(), instanceData.data(), instanceData.size() * sizeof(InstanceData));

// Set instance buffer and draw all instances in one call
commandList->IASetVertexBuffers(1, 1, &instanceBufferView);
commandList->DrawIndexedInstanced(indexCount, objectCount, 0, 0, 0);
```

#### Vulkan Indirect Drawing

```cpp
// Efficient: Multi-draw indirect for dynamic batching
struct VkDrawIndexedIndirectCommand commands[MAX_DRAW_COMMANDS];
int commandCount = 0;

// Populate commands
for (const auto& batch : renderBatches) {
    commands[commandCount].indexCount = batch.indexCount;
    commands[commandCount].instanceCount = batch.instanceCount;
    commands[commandCount].firstIndex = batch.firstIndex;
    commands[commandCount].vertexOffset = batch.vertexOffset;
    commands[commandCount].firstInstance = batch.firstInstance;
    commandCount++;
}

// Upload commands to GPU buffer
UpdateBuffer(indirectBuffer, commands, commandCount * sizeof(VkDrawIndexedIndirectCommand));

// Draw all batches with a single API call
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer, 0, commandCount, sizeof(VkDrawIndexedIndirectCommand));
```

### View Frustum Culling

```cpp
// Basic view frustum culling
struct Frustum {
    XMVECTOR planes[6]; // Left, right, top, bottom, near, far
};

Frustum CreateFrustumFromMatrix(XMMATRIX viewProj) {
    Frustum frustum;
    
    // Extract planes from view-projection matrix
    // Left plane
    frustum.planes[0] = XMVectorSet(
        viewProj.r[0].m128_f32[3] + viewProj.r[0].m128_f32[0],
        viewProj.r[1].m128_f32[3] + viewProj.r[1].m128_f32[0],
        viewProj.r[2].m128_f32[3] + viewProj.r[2].m128_f32[0],
        viewProj.r[3].m128_f32[3] + viewProj.r[3].m128_f32[0]);
    
    // Right plane
    frustum.planes[1] = XMVectorSet(
        viewProj.r[0].m128_f32[3] - viewProj.r[0].m128_f32[0],
        viewProj.r[1].m128_f32[3] - viewProj.r[1].m128_f32[0],
        viewProj.r[2].m128_f32[3] - viewProj.r[2].m128_f32[0],
        viewProj.r[3].m128_f32[3] - viewProj.r[3].m128_f32[0]);
    
    // Normalize planes
    for (int i = 0; i < 6; i++) {
        XMVECTOR length = XMVector3Length(XMVectorSet(
            XMVectorGetX(frustum.planes[i]),
            XMVectorGetY(frustum.planes[i]),
            XMVectorGetZ(frustum.planes[i]), 0));
        frustum.planes[i] = XMVectorDivide(frustum.planes[i], length);
    }
    
    return frustum;
}

bool SphereInFrustum(const Frustum& frustum, XMVECTOR center, float radius) {
    for (int i = 0; i < 6; i++) {
        XMVECTOR plane = frustum.planes[i];
        float distance = XMVectorGetX(XMVector3Dot(center, plane)) + XMVectorGetW(plane);
        
        if (distance < -radius) {
            return false; // Outside frustum
        }
    }
    
    return true; // Inside or intersecting frustum
}

// Using frustum culling during rendering
Frustum viewFrustum = CreateFrustumFromMatrix(viewProjectionMatrix);

std::vector<RenderObject> visibleObjects;
for (const auto& object : allObjects) {
    if (SphereInFrustum(viewFrustum, object.boundingSphereCenter, object.boundingSphereRadius)) {
        visibleObjects.push_back(object);
    }
}
```

### Memory Bandwidth Optimization

```cpp
// Texture compression
D3D12_RESOURCE_DESC textureDesc = {};
textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
textureDesc.Width = width;
textureDesc.Height = height;
textureDesc.DepthOrArraySize = 1;
textureDesc.MipLevels = 0; // Generate full mip chain
textureDesc.Format = DXGI_FORMAT_BC7_UNORM; // Block Compressed format
textureDesc.SampleDesc.Count = 1;
textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

// Efficient vertex layout
struct Vertex {
    XMFLOAT3 position;  // 12 bytes
    XMFLOAT3 normal;    // 12 bytes
    XMFLOAT2 texCoord;  // 8 bytes
    // Total: 32 bytes (cache-friendly)
};

// Inefficient vertex layout
struct IneffientVertex {
    XMFLOAT3 position;  // 12 bytes
    XMFLOAT4 color;     // 16 bytes
    XMFLOAT3 normal;    // 12 bytes
    XMFLOAT3 tangent;   // 12 bytes
    XMFLOAT2 texCoord0; // 8 bytes
    XMFLOAT2 texCoord1; // 8 bytes
    // Total: 68 bytes (wastes bandwidth for simple shaders)
};
```

### Shader Optimization

```hlsl
// Inefficient shader with divergent branching
float4 PSInefficient(PixelInput input) : SV_TARGET {
    float4 color = baseColor;
    
    // Highly divergent branch
    if (input.texCoord.x > 0.5) {
        for (int i = 0; i < 10; i++) {
            color += lightColors[i] * CalculateLighting(input.position, input.normal, lightPositions[i]);
        }
    } else {
        color = texture0.Sample(sampler0, input.texCoord);
    }
    
    return color;
}

// Efficient shader avoiding divergence
float4 PSEfficient(PixelInput input) : SV_TARGET {
    float4 textureColor = texture0.Sample(sampler0, input.texCoord);
    
    // Pre-calculate lighting for all pixels
    float4 lightingColor = baseColor;
    for (int i = 0; i < 10; i++) {
        lightingColor += lightColors[i] * CalculateLighting(input.position, input.normal, lightPositions[i]);
    }
    
    // Use lerp instead of branch
    float blend = step(0.5, input.texCoord.x);
    return lerp(textureColor, lightingColor, blend);
}
```

## 3. Profiling and Debugging Tools

### Using PIX for DirectX 12

```cpp
// Annotating your code for PIX
#include "pix3.h"

void RenderFrame() {
    PIXBeginEvent(commandList.Get(), PIX_COLOR(255, 0, 0), "Main Render Pass");
    
    PIXBeginEvent(commandList.Get(), PIX_COLOR(0, 255, 0), "Shadow Maps");
    RenderShadowMaps(commandList.Get());
    PIXEndEvent(commandList.Get());
    
    PIXBeginEvent(commandList.Get(), PIX_COLOR(0, 0, 255), "Geometry Pass");
    RenderGeometry(commandList.Get());
    PIXEndEvent(commandList.Get());
    
    PIXBeginEvent(commandList.Get(), PIX_COLOR(255, 255, 0), "Lighting Pass");
    RenderLighting(commandList.Get());
    PIXEndEvent(commandList.Get());
    
    PIXEndEvent(commandList.Get());
}
```

### Using Vulkan Validation Layers

```cpp
// Setting up validation layers
std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

VkInstanceCreateInfo createInfo = {};
createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

// Enable validation layers
createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
createInfo.ppEnabledLayerNames = validationLayers.data();

// Setup debug messenger
VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
debugCreateInfo.messageSeverity = 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
debugCreateInfo.messageType = 
    VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
debugCreateInfo.pfnUserCallback = debugCallback;

// Add to instance creation
createInfo.pNext = &debugCreateInfo;
```

### Using Metal Frame Capture

```swift
// Enable Metal debugging in your app
let captureManager = MTLCaptureManager.shared()
let captureDescriptor = MTLCaptureDescriptor()
captureDescriptor.captureObject = device

do {
    try captureManager.startCapture(with: captureDescriptor)
    // Render your frame
    captureManager.stopCapture()
} catch {
    print("Failed to capture: \(error)")
}

// Add debug markers
let commandBuffer = commandQueue.makeCommandBuffer()!
commandBuffer.label = "Main Frame Command Buffer"

let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
renderEncoder.label = "Main Render Pass"

renderEncoder.pushDebugGroup("Shadow Pass")
// Render shadows
renderEncoder.popDebugGroup()

renderEncoder.pushDebugGroup("Geometry Pass")
// Render geometry
renderEncoder.popDebugGroup()
```

## 4. Common Bottlenecks and Solutions

### CPU-Bound Scenarios

```cpp
// Problem: Too many draw calls
// Before: One draw call per object
for (const auto& object : objects) {
    SetObjectConstants(commandList, object);
    commandList->DrawIndexedInstanced(object.indexCount, 1, object.indexStart, object.vertexStart, 0);
}

// Solution: Multi-threaded command recording
std::vector<std::thread> threads;
std::vector<ComPtr<ID3D12GraphicsCommandList>> commandLists(threadCount);

for (int t = 0; t < threadCount; t++) {
    threads.push_back(std::thread([t, &commandLists, &objects]() {
        int objectsPerThread = objects.size() / threadCount;
        int startIdx = t * objectsPerThread;
        int endIdx = (t == threadCount - 1) ? objects.size() : (t + 1) * objectsPerThread;
        
        // Record commands for this thread's objects
        for (int i = startIdx; i < endIdx; i++) {
            SetObjectConstants(commandLists[t].Get(), objects[i]);
            commandLists[t]->DrawIndexedInstanced(objects[i].indexCount, 1, 
                                                objects[i].indexStart, 
                                                objects[i].vertexStart, 0);
        }
    }));
}

// Wait for all threads to finish
for (auto& thread : threads) {
    thread.join();
}

// Execute all command lists
std::vector<ID3D12CommandList*> cmdLists(threadCount);
for (int i = 0; i < threadCount; i++) {
    cmdLists[i] = commandLists[i].Get();
}
commandQueue->ExecuteCommandLists(threadCount, cmdLists.data());
```

### GPU Vertex-Bound Scenarios

```cpp
// Problem: Complex vertex processing
// Solution: Level of Detail (LOD) system

class MeshLOD {
private:
    std::vector<Mesh> m_lodLevels; // Different detail levels
    
public:
    void Initialize(const std::string& baseMesh, int lodCount) {
        m_lodLevels.resize(lodCount);
        
        // Load highest detail mesh
        m_lodLevels[0] = LoadMesh(baseMesh);
        
        // Generate lower detail meshes
        for (int i = 1; i < lodCount; i++) {
            float reductionFactor = static_cast<float>(i) / (lodCount - 1);
            m_lodLevels[i] = SimplifyMesh(m_lodLevels[0], reductionFactor);
        }
    }
    
    const Mesh& GetLOD(float distance, float maxDistance) {
        // Select LOD based on distance
        float normalizedDistance = std::min(distance / maxDistance, 1.0f);
        int lodIndex = static_cast<int>(normalizedDistance * (m_lodLevels.size() - 1));
        return m_lodLevels[lodIndex];
    }
};
```

### GPU Fragment-Bound Scenarios

```cpp
// Problem: High overdraw and complex fragment shaders
// Solution 1: Depth pre-pass

void RenderFrame() {
    // First pass: Render only to depth buffer with simple shader
    SetPipelineState(depthOnlyPipeline);
    for (const auto& object : visibleObjects) {
        SetObjectConstants(commandList, object);
        commandList->DrawIndexedInstanced(object.indexCount, 1, object.indexStart, object.vertexStart, 0);
    }
    
    // Second pass: Full shading (many pixels will be skipped due to depth test)
    SetPipelineState(fullShadingPipeline);
    for (const auto& object : visibleObjects) {
        SetObjectConstants(commandList, object);
        commandList->DrawIndexedInstanced(object.indexCount, 1, object.indexStart, object.vertexStart, 0);
    }
}

// Solution 2: Optimize shader complexity
// Before: Expensive lighting calculation per pixel
float4 PSExpensive(PixelInput input) : SV_TARGET {
    float4 color = baseColor;
    
    // Calculate lighting for many lights
    for (int i = 0; i < 100; i++) {
        color += CalculatePhysicallyBasedLighting(input, lights[i]);
    }
    
    return color;
}

// After: Light culling and clustering
float4 PSOptimized(PixelInput input) : SV_TARGET {
    float4 color = baseColor;
    
    // Get cluster index for this pixel
    uint3 clusterIndex = CalculateClusterIndex(input.position);
    
    // Get light list for this cluster (precomputed in compute shader)
    uint lightCount = lightGrid[clusterIndex].count;
    uint lightOffset = lightGrid[clusterIndex].offset;
    
    // Only process lights affecting this cluster
    for (uint i = 0; i < lightCount; i++) {
        uint lightIndex = lightIndices[lightOffset + i];
        color += CalculatePhysicallyBasedLighting(input, lights[lightIndex]);
    }
    
    return color;
}
```

### Memory Bandwidth-Bound Scenarios

```cpp
// Problem: Large texture data and inefficient access
// Solution: Texture streaming and virtual texturing

class TextureStreamer {
private:
    struct TextureTile {
        int mipLevel;
        int tileX, tileY;
        bool loaded;
        // ...
    };
    
    std::unordered_map<std::string, std::vector<TextureTile>> m_textureTiles;
    std::queue<TextureTile*> m_loadQueue;
    
public:
    void Update(const Camera& camera) {
        // Determine which tiles are visible based on camera position
        std::vector<TextureTile*> visibleTiles = DetermineVisibleTiles(camera);
        
        // Prioritize tiles based on distance and screen coverage
        PrioritizeTiles(visibleTiles, camera);
        
        // Queue tiles for loading
        for (auto* tile : visibleTiles) {
            if (!tile->loaded) {
                m_loadQueue.push(tile);
            }
        }
        
        // Process load queue (limited by budget)
        ProcessLoadQueue(MAX_TILES_PER_FRAME);
    }
    
    void ProcessLoadQueue(int maxTilesToLoad) {
        for (int i = 0; i < maxTilesToLoad && !m_loadQueue.empty(); i++) {
            TextureTile* tile = m_loadQueue.front();
            m_loadQueue.pop();
            
            // Load tile data from disk or generate
            LoadTileData(tile);
            tile->loaded = true;
        }
    }
};
```

## Conclusion

Performance optimization is a continuous process that requires understanding both the hardware capabilities and the specific characteristics of your application. The key takeaways from this lesson are:

1. **Know your bottlenecks**: Use profiling tools to identify whether you're CPU-bound, GPU-bound, or memory-bound before optimizing.

2. **Leverage API-specific features**: Each API (DirectX 12, Vulkan, Metal) offers unique optimization opportunities that align with its design philosophy.

3. **Minimize state changes**: Group similar draw calls, use instancing, and batch commands to reduce overhead.

4. **Optimize resource usage**: Use appropriate formats, compression, and update strategies to minimize memory bandwidth.

5. **Cull invisible geometry**: Implement frustum culling, occlusion culling, and LOD systems to avoid processing unnecessary geometry.

6. **Profile regularly**: Make optimization an integral part of your development process, not just a final step.

By applying these principles and techniques, you can create high-performance graphics applications that make efficient use of modern hardware across all three major graphics APIs.
