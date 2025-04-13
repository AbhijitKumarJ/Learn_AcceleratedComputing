# Part 7: Resource Management

## Introduction

Resource management is the backbone of any graphics application. How you create, organize, and update GPU resources directly impacts both performance and memory usage. Modern graphics APIs offer explicit control over these resources, but with different approaches and terminology. This lesson provides a comprehensive look at resource management in DirectX 12, Vulkan, and Metal, with practical examples and optimization strategies.

## 1. Textures, Buffers, and Uniform Data

### Core Resource Types

#### Textures
Textures store image data used for rendering, from simple color maps to complex data arrays. Key considerations include:

- **Formats**: Each API provides enumerations for pixel formats:
  - DirectX: `DXGI_FORMAT` (e.g., `DXGI_FORMAT_R8G8B8A8_UNORM`)  
  - Vulkan: `VkFormat` (e.g., `VK_FORMAT_R8G8B8A8_UNORM`)
  - Metal: `MTLPixelFormat` (e.g., `MTLPixelFormatRGBA8Unorm`)

- **Dimensionality**: 1D (gradients), 2D (standard images), 3D (volumetric data), Cube (environment maps)

- **Mipmapping**: Pre-calculated downscaled versions of textures that improve performance and quality

- **Sampling**: How textures are read in shaders (filtering, addressing modes)

#### Buffers
Buffers are linear memory regions for various data types:

- **Vertex Buffers**: Store vertex attributes (positions, normals, UVs)
- **Index Buffers**: Store mesh topology information
- **Constant/Uniform Buffers**: Store shader parameters
- **Storage/Unordered Access Buffers**: For read-write operations, especially in compute shaders

#### Uniform Data
Constant data passed to shaders, typically including:
- Transformation matrices (model, view, projection)
- Material properties
- Lighting parameters
- Time-based animation values

### API Comparison

#### DirectX 12
```cpp
// Creating a texture in DirectX 12
D3D12_RESOURCE_DESC textureDesc = {};
textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
textureDesc.Width = width;
textureDesc.Height = height;
textureDesc.DepthOrArraySize = 1;
textureDesc.MipLevels = 1;
textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
textureDesc.SampleDesc.Count = 1;
textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

D3D12_HEAP_PROPERTIES heapProps = {};
heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

ComPtr<ID3D12Resource> texture;
device->CreateCommittedResource(
    &heapProps,
    D3D12_HEAP_FLAG_NONE,
    &textureDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    nullptr,
    IID_PPV_ARGS(&texture));

// Creating a buffer
D3D12_RESOURCE_DESC bufferDesc = {};
bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
bufferDesc.Width = bufferSize;
bufferDesc.Height = 1;
bufferDesc.DepthOrArraySize = 1;
bufferDesc.MipLevels = 1;
bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
bufferDesc.SampleDesc.Count = 1;
bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

ComPtr<ID3D12Resource> buffer;
device->CreateCommittedResource(
    &heapProps,
    D3D12_HEAP_FLAG_NONE,
    &bufferDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&buffer));
```

#### Vulkan
```cpp
// Creating a texture in Vulkan
VkImageCreateInfo imageInfo = {};
imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
imageInfo.imageType = VK_IMAGE_TYPE_2D;
imageInfo.extent.width = width;
imageInfo.extent.height = height;
imageInfo.extent.depth = 1;
imageInfo.mipLevels = 1;
imageInfo.arrayLayers = 1;
imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

VkImage textureImage;
vkCreateImage(device, &imageInfo, nullptr, &textureImage);

// Allocate and bind memory for the image
VkMemoryRequirements memRequirements;
vkGetImageMemoryRequirements(device, textureImage, &memRequirements);

VkMemoryAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
allocInfo.allocationSize = memRequirements.size;
allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

VkDeviceMemory textureImageMemory;
vkAllocateMemory(device, &allocInfo, nullptr, &textureImageMemory);
vkBindImageMemory(device, textureImage, textureImageMemory, 0);

// Creating a buffer
VkBufferCreateInfo bufferInfo = {};
bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
bufferInfo.size = bufferSize;
bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

VkBuffer buffer;
vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
```

#### Metal
```swift
// Creating a texture in Metal
let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: width,
    height: height,
    mipmapped: true)
textureDescriptor.usage = [.shaderRead]

let texture = device.makeTexture(descriptor: textureDescriptor)!

// Creating a buffer
let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
```

### Key Differences
- **Memory Management**: 
  - Vulkan requires explicit memory allocation and binding
  - DirectX 12 uses committed resources or placed resources with heaps
  - Metal abstracts memory management but offers storage mode control

- **Resource States/Barriers**: 
  - DirectX 12 and Vulkan require explicit state transitions
  - Metal handles most transitions automatically

## 2. Resource Binding Models Compared

Resource binding is where the APIs differ most significantly in their approach.

### DirectX 12: Root Signatures and Descriptor Heaps

DirectX 12 uses a two-level binding model:

1. **Root Signature**: Defines the layout of parameters accessible to shaders
   - Root Constants: Small values embedded directly in command lists
   - Root Descriptors: GPU virtual addresses pointing to resources
   - Descriptor Tables: References to ranges within descriptor heaps

2. **Descriptor Heaps**: GPU-visible pools of descriptors
   - CBV/SRV/UAV Heaps: For constant buffer, shader resource, and unordered access views
   - Sampler Heaps: For sampler states

```cpp
// Creating a root signature
CD3DX12_ROOT_PARAMETER rootParams[2];

// Root descriptor for a constant buffer
rootParams[0].InitAsConstantBufferView(0); // b0 register

// Descriptor table for textures
CD3DX12_DESCRIPTOR_RANGE descRange;
descRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0); // t0 register
rootParams[1].InitAsDescriptorTable(1, &descRange);

CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc;
rootSigDesc.Init(2, rootParams, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

ComPtr<ID3DBlob> signature;
ComPtr<ID3DBlob> error;
D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);

device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature));

// Using the root signature
commandList->SetGraphicsRootSignature(rootSignature.Get());
commandList->SetGraphicsRootConstantBufferView(0, constantBuffer->GetGPUVirtualAddress());
commandList->SetGraphicsRootDescriptorTable(1, srvHandle);
```

### Vulkan: Descriptor Sets and Pipeline Layouts

Vulkan uses a hierarchical binding model:

1. **Descriptor Set Layout**: Defines the structure of a set of resources
2. **Pipeline Layout**: Combines multiple descriptor set layouts
3. **Descriptor Pool**: Allocates descriptor sets from
4. **Descriptor Sets**: Groups of resources bound together

```cpp
// Creating a descriptor set layout
VkDescriptorSetLayoutBinding uboBinding = {};
uboBinding.binding = 0;
uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
uboBinding.descriptorCount = 1;
uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

VkDescriptorSetLayoutBinding samplerBinding = {};
samplerBinding.binding = 1;
samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
samplerBinding.descriptorCount = 1;
samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboBinding, samplerBinding};

VkDescriptorSetLayoutCreateInfo layoutInfo = {};
layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
layoutInfo.pBindings = bindings.data();

VkDescriptorSetLayout descriptorSetLayout;
vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);

// Creating a pipeline layout
VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
pipelineLayoutInfo.setLayoutCount = 1;
pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

VkPipelineLayout pipelineLayout;
vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

// Allocating and updating descriptor sets
VkDescriptorSetAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
allocInfo.descriptorPool = descriptorPool;
allocInfo.descriptorSetCount = 1;
allocInfo.pSetLayouts = &descriptorSetLayout;

VkDescriptorSet descriptorSet;
vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

// Binding descriptor sets
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
```

### Metal: Argument Buffers and Direct Binding

Metal offers two approaches:

1. **Direct Binding**: Simple and straightforward for basic needs
2. **Argument Buffers**: More efficient for complex scenes with many resources

```swift
// Direct binding in Metal
let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
renderEncoder.setFragmentTexture(texture, index: 0)
renderEncoder.setFragmentSamplerState(sampler, index: 0)

// Using argument buffers
let argumentEncoder = device.makeArgumentEncoder(arguments: [
    .buffer(0, offset: 0, access: .readOnly),
    .texture(0, access: .read),
    .sampler(0)
])

let argumentBuffer = device.makeBuffer(length: argumentEncoder.encodedLength, options: .storageModeShared)!

argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
argumentEncoder.setBuffer(uniformBuffer, offset: 0, index: 0)
argumentEncoder.setTexture(texture, index: 1)
argumentEncoder.setSamplerState(sampler, index: 2)

renderEncoder.setFragmentBuffer(argumentBuffer, offset: 0, index: 0)
```

### Comparative Analysis

- **Performance Characteristics**:
  - DirectX 12: Root constants and descriptors are fastest but limited in number
  - Vulkan: Descriptor sets allow efficient binding of resource groups
  - Metal: Argument buffers provide the most scalable approach for complex scenes

- **Bindless Rendering**: All three APIs support "bindless" approaches for handling thousands of resources
  - DirectX 12: Large descriptor heaps with indexing
  - Vulkan: Descriptor indexing extension
  - Metal: Argument buffers with arrays

## 3. Descriptor Sets / Tables / Arguments in Depth

### Vulkan's Descriptor Sets

Descriptor sets in Vulkan group related resources that change at the same frequency:

- **Per-frame sets**: For camera matrices, time-based values
- **Per-material sets**: For textures, material properties
- **Per-object sets**: For model matrices, instance data

```cpp
// Creating multiple descriptor set layouts for different update frequencies
VkDescriptorSetLayout perFrameLayout; // For camera data, updated every frame
VkDescriptorSetLayout perMaterialLayout; // For material data, updated when material changes
VkDescriptorSetLayout perObjectLayout; // For object data, updated for each object

// Pipeline layout combining all three
std::array<VkDescriptorSetLayout, 3> layouts = {perFrameLayout, perMaterialLayout, perObjectLayout};
VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
pipelineLayoutInfo.pSetLayouts = layouts.data();

// Binding multiple descriptor sets
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 
                        0, 1, &perFrameSet, 0, nullptr);
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 
                        1, 1, &materialSet, 0, nullptr);

// For each object in a loop
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 
                        2, 1, &objectSet, 0, nullptr);
vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
```

### DirectX 12's Descriptor Tables

Descriptor tables in DirectX 12 reference ranges within descriptor heaps:

```cpp
// Creating a CBV/SRV/UAV descriptor heap
D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
heapDesc.NumDescriptors = 1000; // Size of the heap
heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&cbvSrvUavHeap));

// Creating descriptors in the heap
D3D12_CPU_DESCRIPTOR_HANDLE handle = cbvSrvUavHeap->GetCPUDescriptorHandleForHeapStart();
D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
srvDesc.Texture2D.MipLevels = 1;

device->CreateShaderResourceView(texture.Get(), &srvDesc, handle);

// Using descriptor tables
commandList->SetDescriptorHeaps(1, &cbvSrvUavHeap);
D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = cbvSrvUavHeap->GetGPUDescriptorHandleForHeapStart();
commandList->SetGraphicsRootDescriptorTable(0, gpuHandle);
```

### Metal's Argument Buffers

Argument buffers in Metal encode resource references into a buffer:

```swift
// Define argument buffer structure in shader
struct MaterialResources {
    texture2d<float> albedoMap [[id(0)]];
    texture2d<float> normalMap [[id(1)]];
    texture2d<float> metallicRoughnessMap [[id(2)]];
    sampler textureSampler [[id(3)]];
};

// Create and populate argument buffer
let argumentDescriptor = MTLArgumentDescriptor.argumentDescriptor()
argumentDescriptor.index = 0
argumentDescriptor.access = .readOnly
argumentDescriptor.dataType = .texture

// ... add more arguments ...

let argumentEncoder = device.makeArgumentEncoder(arguments: [argumentDescriptor])
let argumentBuffer = device.makeBuffer(length: argumentEncoder.encodedLength, options: .storageModeShared)

argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
argumentEncoder.setTexture(albedoTexture, index: 0)
argumentEncoder.setTexture(normalTexture, index: 1)
argumentEncoder.setTexture(metallicRoughnessTexture, index: 2)
argumentEncoder.setSamplerState(sampler, index: 3)

// Use argument buffer in render pass
renderEncoder.setFragmentBuffer(argumentBuffer, offset: 0, index: 1)
```

## 4. Efficient Resource Updates

### Mapping and Unmapping

Directly accessing GPU memory for updates:

#### DirectX 12
```cpp
// Map a buffer for CPU access
void* mappedData;
D3D12_RANGE readRange = {0, 0}; // We're not reading
buffer->Map(0, &readRange, &mappedData);
memcpy(mappedData, sourceData, dataSize);
buffer->Unmap(0, nullptr);
```

#### Vulkan
```cpp
// Map a buffer for CPU access
void* data;
vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &data);
memcpy(data, sourceData, bufferSize);

// If memory is not host coherent, flush the range
VkMappedMemoryRange memoryRange = {};
memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
memoryRange.memory = bufferMemory;
memoryRange.offset = 0;
memoryRange.size = bufferSize;
vkFlushMappedMemoryRanges(device, 1, &memoryRange);

vkUnmapMemory(device, bufferMemory);
```

#### Metal
```swift
// With storage mode shared, direct access is possible
let contents = buffer.contents()
memcpy(contents, sourceData, bufferSize)

// With storage mode managed, must explicitly synchronize
buffer.didModifyRange(0..<bufferSize)
```

### Staging Buffers

For resources that can't be directly mapped, use staging buffers:

```cpp
// DirectX 12 example
D3D12_RESOURCE_DESC uploadBufferDesc = {};
uploadBufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
uploadBufferDesc.Width = textureDataSize;
uploadBufferDesc.Height = 1;
uploadBufferDesc.DepthOrArraySize = 1;
uploadBufferDesc.MipLevels = 1;
uploadBufferDesc.Format = DXGI_FORMAT_UNKNOWN;
uploadBufferDesc.SampleDesc.Count = 1;
uploadBufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

D3D12_HEAP_PROPERTIES uploadHeapProps = {};
uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

ComPtr<ID3D12Resource> uploadBuffer;
device->CreateCommittedResource(
    &uploadHeapProps,
    D3D12_HEAP_FLAG_NONE,
    &uploadBufferDesc,
    D3D12_RESOURCE_STATE_GENERIC_READ,
    nullptr,
    IID_PPV_ARGS(&uploadBuffer));

// Copy data to upload buffer
void* mappedData;
uploadBuffer->Map(0, nullptr, &mappedData);
memcpy(mappedData, textureData, textureDataSize);
uploadBuffer->Unmap(0, nullptr);

// Copy from upload buffer to texture
D3D12_TEXTURE_COPY_LOCATION src = {};
src.pResource = uploadBuffer.Get();
src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
src.PlacedFootprint.Footprint.Width = textureWidth;
src.PlacedFootprint.Footprint.Height = textureHeight;
src.PlacedFootprint.Footprint.Depth = 1;
src.PlacedFootprint.Footprint.RowPitch = textureRowPitch;

D3D12_TEXTURE_COPY_LOCATION dst = {};
dst.pResource = texture.Get();
dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
dst.SubresourceIndex = 0;

commandList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
```

### Update Frequency Optimization

Strategies for different update patterns:

1. **Static Data (load-time only)**
   - Use default/device-local memory for best read performance
   - Upload once during initialization

2. **Per-Frame Data**
   - Use ring buffers with multiple frames in flight
   - DirectX 12: Multiple upload heaps
   - Vulkan: Multiple host-visible buffers
   - Metal: Multiple buffers with triple buffering

3. **Dynamic/Streaming Data**
   - Use persistent mapping where possible
   - Consider suballocation within larger buffers
   - Avoid frequent map/unmap cycles

```cpp
// Ring buffer example for DirectX 12
const int FRAME_COUNT = 3; // Triple buffering
ComPtr<ID3D12Resource> perFrameBuffers[FRAME_COUNT];
void* mappedPerFrameData[FRAME_COUNT];

// During initialization
for (int i = 0; i < FRAME_COUNT; i++) {
    // Create buffer
    device->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&perFrameBuffers[i]));
    
    // Map and keep mapped for the application lifetime
    perFrameBuffers[i]->Map(0, nullptr, &mappedPerFrameData[i]);
}

// During frame update
int currentFrame = frameIndex % FRAME_COUNT;
memcpy(mappedPerFrameData[currentFrame], newFrameData, frameDataSize);

// Use the current frame's buffer
commandList->SetGraphicsRootConstantBufferView(0, perFrameBuffers[currentFrame]->GetGPUVirtualAddress());
```

## Conclusion

Effective resource management is critical for performance in modern graphics applications. The key takeaways are:

1. **Choose the right resource types and formats** for your specific needs
2. **Understand the binding model** of your target API and organize resources accordingly
3. **Group resources by update frequency** to minimize binding changes
4. **Use appropriate update strategies** based on how often data changes
5. **Consider memory locality** for better cache performance

By mastering these concepts across DirectX 12, Vulkan, and Metal, you'll be able to create efficient rendering systems that make the best use of GPU resources while maintaining cross-platform compatibility where needed.
