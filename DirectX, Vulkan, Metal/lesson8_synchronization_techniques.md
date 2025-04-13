# Part 8: Synchronization Techniques

## Introduction

Synchronization is one of the most challenging aspects of modern graphics programming. With explicit APIs like DirectX 12, Vulkan, and Metal, developers are responsible for ensuring correct execution order and preventing resource access hazards. This lesson provides a comprehensive guide to synchronization primitives, their usage patterns, and common pitfalls across all three APIs.

## 1. Why Synchronization Matters

### The Asynchronous Nature of GPU Execution

Modern GPUs operate asynchronously from the CPU and internally parallelize work across multiple execution units:

- **Command Submission vs. Execution**: Commands are submitted by the CPU but executed later by the GPU
- **Parallel Execution Units**: Different operations (compute, graphics, copy) may execute simultaneously
- **Multiple Queues**: Many GPUs have dedicated queues for graphics, compute, and transfer operations

### Types of Synchronization

1. **CPU-GPU Synchronization**
   - CPU waiting for GPU work to complete
   - GPU waiting for CPU to provide resources or commands

2. **GPU-GPU Synchronization**
   - Between different command buffers/lists
   - Between different queues (graphics, compute, transfer)
   - Within a single command buffer/list (pipeline stages)

### Resource Hazards

Without proper synchronization, resources can be accessed incorrectly, leading to race conditions:

- **Write-After-Read (WAR)**: Writing to a resource that was previously read
- **Read-After-Write (RAW)**: Reading from a resource that was previously written
- **Write-After-Write (WAW)**: Multiple writes to the same resource

Example scenario without synchronization:
```
Operation 1: Render to texture T
Operation 2: Use texture T as input for a compute shader
Operation 3: Copy texture T to another texture
```

Without synchronization, Operation 2 might read from texture T before Operation 1 finishes writing to it, or Operation 3 might start copying while Operation 2 is still reading.

### Consequences of Poor Synchronization

- **Visual Artifacts**: Incomplete rendering, flickering, or corrupted textures
- **Incorrect Computation Results**: Partial or stale data used in calculations
- **GPU Hangs**: Deadlocks or timeouts when the GPU waits indefinitely
- **Driver/Application Crashes**: Access violations or validation errors
- **Performance Degradation**: Unnecessary stalls or serialization

## 2. Fences, Semaphores, and Barriers

### CPU-GPU Synchronization

#### DirectX 12 Fences

Fences in DirectX 12 are used primarily for CPU-GPU synchronization:

```cpp
// Creating a fence
ComPtr<ID3D12Fence> fence;
UINT64 fenceValue = 0;
device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

// Signal the fence from the GPU (typically at the end of a frame)
commandQueue->Signal(fence.Get(), ++fenceValue);

// Wait for the GPU to complete work on the CPU
if (fence->GetCompletedValue() < fenceValue) {
    HANDLE eventHandle = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    fence->SetEventOnCompletion(fenceValue, eventHandle);
    WaitForSingleObject(eventHandle, INFINITE);
    CloseHandle(eventHandle);
}
```

#### Vulkan Fences

Vulkan fences serve a similar purpose:

```cpp
// Creating a fence
VkFenceCreateInfo fenceInfo = {};
fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled for first frame

VkFence fence;
vkCreateFence(device, &fenceInfo, nullptr, &fence);

// Submit work and associate with fence
vkQueueSubmit(queue, 1, &submitInfo, fence);

// Wait for the GPU to complete work
vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
vkResetFences(device, 1, &fence); // Reset for reuse
```

#### Metal Events and Completion Handlers

Metal uses events and completion handlers for CPU-GPU synchronization:

```swift
// Using a completion handler
commandBuffer.addCompletedHandler { [weak self] _ in
    // This block is called when the GPU finishes executing the command buffer
    self?.frameCompleted()
}

// Using MTLEvent
let event = device.makeEvent()!
commandBuffer.encodeSignalEvent(event, value: 1)

// Wait for the event on the CPU
while event.signaledValue < 1 {
    // Spin or sleep
    Thread.sleep(forTimeInterval: 0.001)
}
```

### GPU-GPU Synchronization Between Command Buffers

#### Vulkan Semaphores

Semaphores in Vulkan synchronize work between different queue submissions:

```cpp
// Creating semaphores
VkSemaphoreCreateInfo semaphoreInfo = {};
semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

VkSemaphore imageAvailableSemaphore;
VkSemaphore renderFinishedSemaphore;
vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore);
vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore);

// Acquire next swapchain image, signaling imageAvailableSemaphore when ready
vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

// Wait for imageAvailableSemaphore before rendering, signal renderFinishedSemaphore when done
VkSubmitInfo submitInfo = {};
submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
submitInfo.waitSemaphoreCount = 1;
submitInfo.pWaitSemaphores = waitSemaphores;
submitInfo.pWaitDstStageMask = waitStages;
submitInfo.commandBufferCount = 1;
submitInfo.pCommandBuffers = &commandBuffer;

VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
submitInfo.signalSemaphoreCount = 1;
submitInfo.pSignalSemaphores = signalSemaphores;

vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

// Wait for renderFinishedSemaphore before presenting
VkPresentInfoKHR presentInfo = {};
presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
presentInfo.waitSemaphoreCount = 1;
presentInfo.pWaitSemaphores = signalSemaphores;

presentInfo.swapchainCount = 1;
presentInfo.pSwapchains = &swapchain;
presentInfo.pImageIndices = &imageIndex;

vkQueuePresentKHR(presentQueue, &presentInfo);
```

#### DirectX 12 Command Queue Synchronization

DirectX 12 uses fences for synchronization between queues:

```cpp
// Signal a fence on one queue
commandQueue1->Signal(fence.Get(), fenceValue);

// Wait for the fence on another queue
commandQueue2->Wait(fence.Get(), fenceValue);
```

#### Metal Command Buffer Dependencies

Metal provides several ways to synchronize between command buffers:

```swift
// Using events
let event = device.makeEvent()!

// Signal the event in one command buffer
commandBuffer1.encodeSignalEvent(event, value: 1)

// Wait for the event in another command buffer
commandBuffer2.encodeWaitForEvent(event, value: 1)

// Or using dependencies
commandBuffer2.addCompletedHandler { _ in
    // This will only execute after commandBuffer1 completes
    commandBuffer3.commit()
}
commandBuffer1.commit()
commandBuffer2.commit()
```

### Intra-Command Buffer Synchronization

#### DirectX 12 Resource Barriers

Barriers in DirectX 12 manage resource state transitions and execution dependencies:

```cpp
// Transition a texture from render target to shader resource
D3D12_RESOURCE_BARRIER barrier = {};
barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
barrier.Transition.pResource = texture.Get();
barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

commandList->ResourceBarrier(1, &barrier);
```

#### Vulkan Pipeline Barriers

Vulkan's pipeline barriers are more explicit about pipeline stages and memory access:

```cpp
// Transition a texture from color attachment to shader read
VkImageMemoryBarrier barrier = {};
barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.image = image;
barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
barrier.subresourceRange.baseMipLevel = 0;
barrier.subresourceRange.levelCount = 1;
barrier.subresourceRange.baseArrayLayer = 0;
barrier.subresourceRange.layerCount = 1;
barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // Source stage
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,         // Destination stage
    0,                                             // Dependency flags
    0, nullptr,                                    // Memory barriers
    0, nullptr,                                    // Buffer memory barriers
    1, &barrier                                    // Image memory barriers
);
```

#### Metal Memory Barriers and Resource Usage

Metal handles many transitions automatically but provides explicit barriers when needed:

```swift
// Texture barrier within a render pass
let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!

// First draw that writes to the texture
renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

// Texture barrier to ensure first draw completes before second draw reads
renderEncoder.memoryBarrier(resources: [texture], after: .fragment, before: .fragment)

// Second draw that reads from the texture
renderEncoder.drawPrimitives(type: .triangle, vertexStart: 3, vertexCount: 3)

// For barriers between different encoders
let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
blitEncoder.synchronizeResource(texture)
blitEncoder.endEncoding()
```

## 3. Pipeline Stages and Resource Transitions

### Understanding Pipeline Stages

The graphics pipeline consists of multiple stages that execute in sequence for a given draw call, but may overlap between different draw calls:

- **Early Stages**: Vertex input, vertex shader, tessellation, geometry shader
- **Middle Stages**: Rasterization, fragment/pixel shader
- **Late Stages**: Color blending, output merger

Compute and transfer operations have their own stages.

### Resource State Transitions

Resources must be in the correct state for each operation:

#### DirectX 12 Resource States

Common states include:
- `D3D12_RESOURCE_STATE_COMMON`
- `D3D12_RESOURCE_STATE_RENDER_TARGET`
- `D3D12_RESOURCE_STATE_DEPTH_WRITE`
- `D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE`
- `D3D12_RESOURCE_STATE_UNORDERED_ACCESS`
- `D3D12_RESOURCE_STATE_COPY_DEST`
- `D3D12_RESOURCE_STATE_COPY_SOURCE`

```cpp
// Split barrier example (allows overlapping work)
D3D12_RESOURCE_BARRIER beginBarrier = {};
beginBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
beginBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
beginBarrier.Transition.pResource = resource.Get();
beginBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
beginBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
beginBarrier.Transition.Subresource = 0;

commandList->ResourceBarrier(1, &beginBarrier);

// Do some work here that doesn't depend on the resource

D3D12_RESOURCE_BARRIER endBarrier = beginBarrier;
endBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
commandList->ResourceBarrier(1, &endBarrier);
```

#### Vulkan Image Layouts and Access Masks

Vulkan separates the concepts of image layout and memory access:

**Common Image Layouts:**
- `VK_IMAGE_LAYOUT_UNDEFINED`
- `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`
- `VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
- `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`
- `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL`
- `VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL`
- `VK_IMAGE_LAYOUT_GENERAL`

**Access Masks:**
- `VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT`
- `VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT`
- `VK_ACCESS_SHADER_READ_BIT`
- `VK_ACCESS_TRANSFER_READ_BIT`
- `VK_ACCESS_TRANSFER_WRITE_BIT`

```cpp
// Optimizing barriers with subresource transitions
VkImageMemoryBarrier barrier = {};
barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
barrier.image = image;

// Only transition specific mip levels
barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
barrier.subresourceRange.baseMipLevel = 2;  // Start at mip level 2
barrier.subresourceRange.levelCount = 3;    // Transition 3 mip levels
barrier.subresourceRange.baseArrayLayer = 0;
barrier.subresourceRange.layerCount = 1;

barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    0, nullptr,
    0, nullptr,
    1, &barrier
);
```

#### Metal Resource Usage

Metal handles most transitions implicitly but provides explicit control when needed:

```swift
// Tracking resource usage across encoders
let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
renderEncoder.setFragmentTexture(texture, index: 0)
renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
renderEncoder.endEncoding()

// Ensure texture writes are visible before reading in compute
let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
blitEncoder.synchronizeResource(texture)
blitEncoder.endEncoding()

// Now use the texture in compute
let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
computeEncoder.setTexture(texture, index: 0)
// ...
computeEncoder.endEncoding()
```

### Execution Dependencies

Synchronization isn't just about resource state—it's also about ensuring operations complete in the right order:

```cpp
// Vulkan execution dependency without resource transition
vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,    // Source stage
    VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,     // Destination stage
    0,                                        // Dependency flags
    0, nullptr,                               // Memory barriers
    0, nullptr,                               // Buffer memory barriers
    0, nullptr                                // Image memory barriers
);
```

## 4. Common Synchronization Pitfalls

### Over-Synchronization

Excessive synchronization reduces parallelism and hurts performance:

```cpp
// BAD: Unnecessary full pipeline barrier
vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,      // Too broad
    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,      // Too broad
    0,
    0, nullptr,
    0, nullptr,
    1, &barrier
);

// GOOD: Specific pipeline stages
vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,  // Only wait for color attachment writes
    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,          // Only block fragment shader reads
    0,
    0, nullptr,
    0, nullptr,
    1, &barrier
);
```

### Under-Synchronization

Missing synchronization leads to race conditions:

```cpp
// BAD: Missing barrier between render target write and texture read
renderToTexture(commandList, texture.Get());
// Missing barrier here!
sampleFromTexture(commandList, texture.Get());

// GOOD: Proper barrier
renderToTexture(commandList, texture.Get());

D3D12_RESOURCE_BARRIER barrier = {};
barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
barrier.Transition.pResource = texture.Get();
barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

commandList->ResourceBarrier(1, &barrier);

sampleFromTexture(commandList, texture.Get());
```

### Incorrect Barrier Placement

Barriers must be placed between the operations they're synchronizing:

```cpp
// BAD: Barrier in wrong place
commandList->ResourceBarrier(1, &barrier);  // Too early!
renderToTexture(commandList, texture.Get());
sampleFromTexture(commandList, texture.Get());

// GOOD: Barrier between dependent operations
renderToTexture(commandList, texture.Get());
commandList->ResourceBarrier(1, &barrier);  // Correct placement
sampleFromTexture(commandList, texture.Get());
```

### CPU Busy-Waiting

Inefficient CPU waiting wastes power and blocks other work:

```cpp
// BAD: Busy-waiting
while (fence->GetCompletedValue() < fenceValue) {
    // Spin, wasting CPU cycles
}

// GOOD: Efficient waiting with event
if (fence->GetCompletedValue() < fenceValue) {
    HANDLE eventHandle = CreateEventEx(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    fence->SetEventOnCompletion(fenceValue, eventHandle);
    WaitForSingleObject(eventHandle, INFINITE);  // CPU can sleep
    CloseHandle(eventHandle);
}
```

### Deadlocks

Circular dependencies can cause the GPU to hang:

```cpp
// BAD: Potential deadlock
commandQueue1->Wait(fence1.Get(), value1);  // Queue 1 waits for Queue 2
commandQueue2->Wait(fence2.Get(), value2);  // Queue 2 waits for Queue 1

// GOOD: Avoid circular dependencies
// Design your synchronization to have a clear direction of dependencies
```

### Debugging Tips

1. **Use Validation Layers/Debug Layers**
   - DirectX 12: Enable the debug layer with `D3D12GetDebugInterface`
   - Vulkan: Enable validation layers
   - Metal: Enable the validation layer in Xcode

2. **GPU Timing**
   - Insert timestamp queries around suspected problematic barriers
   - Look for unexpectedly long gaps between operations

3. **Graphics Debuggers**
   - PIX for Windows (DirectX)
   - RenderDoc (Vulkan, DirectX)
   - Xcode GPU Frame Debugger (Metal)

4. **Simplify and Isolate**
   - Remove barriers one by one to identify which ones are necessary
   - Test with simplified rendering paths

## Conclusion

Mastering synchronization is essential for correct and efficient graphics applications. The key principles apply across all modern APIs, though the specific mechanisms differ:

- **Understand the hardware**: Know how the GPU processes commands and manages resources
- **Be specific**: Use the most precise synchronization scope possible
- **Balance correctness and performance**: Ensure correct execution while minimizing stalls
- **Test thoroughly**: Synchronization bugs may only appear on specific hardware or under load

By applying these principles and understanding the synchronization primitives in DirectX 12, Vulkan, and Metal, you can create robust graphics applications that make efficient use of the GPU while avoiding race conditions and other synchronization hazards.
