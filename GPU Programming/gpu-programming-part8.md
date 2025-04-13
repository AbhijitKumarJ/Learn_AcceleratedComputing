# DirectCompute and C++ AMP

*Welcome to the eighth installment of our GPU programming series! In this article, we'll explore Microsoft's GPU computing solutions: DirectCompute and C++ AMP. These technologies provide Windows-specific alternatives to CUDA and OpenCL, with tight integration into the DirectX ecosystem.*

## Introduction to Microsoft's GPU Computing Options

Microsoft has developed several technologies for GPU computing on Windows platforms. The two most significant are DirectCompute and C++ AMP (Accelerated Massive Parallelism). While these technologies are Windows-specific, they offer advantages for developers working within the Microsoft ecosystem, particularly those already using DirectX for graphics.

### DirectCompute

DirectCompute is a compute shader technology that's part of the DirectX API. It was introduced with DirectX 11 and has evolved with subsequent DirectX versions. DirectCompute allows developers to harness the GPU for general-purpose computing tasks while maintaining compatibility with DirectX graphics pipelines.

### C++ AMP

C++ AMP is a higher-level programming model that extends C++ with constructs for data-parallel computing. It was designed to simplify GPU programming by providing a more accessible abstraction layer over the hardware. C++ AMP code can be more concise and readable than equivalent DirectCompute, CUDA, or OpenCL code.

## DirectCompute: Integrating with DirectX

DirectCompute is tightly integrated with the DirectX graphics pipeline, making it an excellent choice for applications that combine graphics and compute workloads.

### Setting Up DirectCompute

To use DirectCompute, you need to set up a DirectX environment:

```cpp
// Create Direct3D 11 device and context
D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
ID3D11Device* device;
ID3D11DeviceContext* context;

HRESULT hr = D3D11CreateDevice(
    nullptr,                    // Default adapter
    D3D_DRIVER_TYPE_HARDWARE,  // Hardware acceleration
    nullptr,                    // No software rasterizer
    D3D11_CREATE_DEVICE_DEBUG,  // Debug flag
    featureLevels,              // Feature levels
    ARRAYSIZE(featureLevels),   // Number of feature levels
    D3D11_SDK_VERSION,          // SDK version
    &device,                    // Output device
    nullptr,                    // Output feature level
    &context                    // Output context
);

if (FAILED(hr)) {
    // Handle device creation error
}
```

### Creating Compute Shaders

DirectCompute shaders are written in HLSL (High-Level Shading Language), the same language used for DirectX graphics shaders. Here's a simple compute shader that doubles each element in a buffer:

```hlsl
// ComputeShader.hlsl
RWBuffer<float> OutputBuffer : register(u0);
Buffer<float> InputBuffer : register(t0);

[numthreads(256, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    uint index = DTid.x;
    OutputBuffer[index] = InputBuffer[index] * 2.0f;
}
```

To compile and use this shader:

```cpp
// Load and compile the compute shader
ID3DBlob* shaderBlob = nullptr;
ID3DBlob* errorBlob = nullptr;

HRESULT hr = D3DCompileFromFile(
    L"ComputeShader.hlsl",  // Shader file
    nullptr,                // Defines
    nullptr,                // Include handler
    "CSMain",               // Entry point
    "cs_5_0",               // Target
    0,                      // Flags1
    0,                      // Flags2
    &shaderBlob,            // Output blob
    &errorBlob              // Error blob
);

if (FAILED(hr)) {
    // Handle compilation error
    if (errorBlob) {
        OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        errorBlob->Release();
    }
    return false;
}

// Create the compute shader
ID3D11ComputeShader* computeShader = nullptr;
hr = device->CreateComputeShader(
    shaderBlob->GetBufferPointer(),
    shaderBlob->GetBufferSize(),
    nullptr,
    &computeShader
);

shaderBlob->Release();

if (FAILED(hr)) {
    // Handle shader creation error
    return false;
}
```

### Creating Buffers

To work with data in DirectCompute, you need to create buffer resources:

```cpp
// Create input buffer
D3D11_BUFFER_DESC inputDesc = {};
inputDesc.ByteWidth = sizeof(float) * numElements;
inputDesc.Usage = D3D11_USAGE_DEFAULT;
inputDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
inputDesc.CPUAccessFlags = 0;
inputDesc.StructureByteStride = sizeof(float);

D3D11_SUBRESOURCE_DATA inputData = {};
inputData.pSysMem = sourceData;

ID3D11Buffer* inputBuffer = nullptr;
hr = device->CreateBuffer(&inputDesc, &inputData, &inputBuffer);

// Create input buffer view
D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
srvDesc.Buffer.FirstElement = 0;
srvDesc.Buffer.NumElements = numElements;

ID3D11ShaderResourceView* inputView = nullptr;
hr = device->CreateShaderResourceView(inputBuffer, &srvDesc, &inputView);

// Create output buffer
D3D11_BUFFER_DESC outputDesc = {};
outputDesc.ByteWidth = sizeof(float) * numElements;
outputDesc.Usage = D3D11_USAGE_DEFAULT;
outputDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
outputDesc.CPUAccessFlags = 0;
outputDesc.StructureByteStride = sizeof(float);

ID3D11Buffer* outputBuffer = nullptr;
hr = device->CreateBuffer(&outputDesc, nullptr, &outputBuffer);

// Create output buffer view
D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
uavDesc.Buffer.FirstElement = 0;
uavDesc.Buffer.NumElements = numElements;

ID3D11UnorderedAccessView* outputView = nullptr;
hr = device->CreateUnorderedAccessView(outputBuffer, &uavDesc, &outputView);
```

### Dispatching Compute Work

To execute the compute shader:

```cpp
// Set shader and resources
context->CSSetShader(computeShader, nullptr, 0);
context->CSSetShaderResources(0, 1, &inputView);
context->CSSetUnorderedAccessViews(0, 1, &outputView, nullptr);

// Dispatch compute work
// For numElements = 1024 and numthreads(256,1,1), we need 1024/256 = 4 thread groups
context->Dispatch(numElements / 256, 1, 1);

// Unset resources to avoid resource conflicts
ID3D11UnorderedAccessView* nullUAV = nullptr;
ID3D11ShaderResourceView* nullSRV = nullptr;
context->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
context->CSSetShaderResources(0, 1, &nullSRV);
```

### Reading Results

To read back the results from the GPU:

```cpp
// Create a staging buffer for reading back data
D3D11_BUFFER_DESC stagingDesc = {};
stagingDesc.ByteWidth = sizeof(float) * numElements;
stagingDesc.Usage = D3D11_USAGE_STAGING;
stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
stagingDesc.StructureByteStride = sizeof(float);

ID3D11Buffer* stagingBuffer = nullptr;
hr = device->CreateBuffer(&stagingDesc, nullptr, &stagingBuffer);

// Copy output to staging buffer
context->CopyResource(stagingBuffer, outputBuffer);

// Map staging buffer to read data
D3D11_MAPPED_SUBRESOURCE mappedResource;
hr = context->Map(stagingBuffer, 0, D3D11_MAP_READ, 0, &mappedResource);

if (SUCCEEDED(hr)) {
    // Copy data from mapped resource
    memcpy(resultData, mappedResource.pData, sizeof(float) * numElements);
    context->Unmap(stagingBuffer, 0);
}

// Release staging buffer
stagingBuffer->Release();
```

## C++ AMP: A Higher-Level Approach

C++ AMP (Accelerated Massive Parallelism) provides a more accessible approach to GPU programming by extending C++ with constructs for data-parallel computing.

### Setting Up C++ AMP

To use C++ AMP, include the necessary headers and set up an accelerator:

```cpp
#include <amp.h>
#include <iostream>

using namespace concurrency;

int main() {
    // List all accelerators
    std::vector<accelerator> accs = accelerator::get_all();
    std::wcout << "Available accelerators:" << std::endl;
    for (auto& acc : accs) {
        std::wcout << acc.description << std::endl;
    }
    
    // Select default accelerator
    accelerator default_acc;
    std::wcout << "Using: " << default_acc.description << std::endl;
    
    // Check if the accelerator supports double precision
    if (default_acc.supports_double_precision) {
        std::wcout << "Double precision supported" << std::endl;
    }
    
    // Rest of the code...
    return 0;
}
```

### C++ AMP Programming Model

C++ AMP introduces several key abstractions:

- **array**: A container for data that can be manipulated on the accelerator
- **array_view**: A view of data that can be shared between CPU and accelerator
- **parallel_for_each**: A function to express parallel computation
- **extent**: Defines the shape and size of the computation domain
- **index**: Represents a position within the computation domain

### Vector Addition Example

Here's a simple vector addition example using C++ AMP:

```cpp
#include <amp.h>
#include <iostream>
#include <vector>

using namespace concurrency;

int main() {
    const int size = 1024;
    
    // Initialize data on CPU
    std::vector<float> a(size);
    std::vector<float> b(size);
    std::vector<float> c(size);
    
    for (int i = 0; i < size; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    // Create array_views
    array_view<const float, 1> a_view(size, a);
    array_view<const float, 1> b_view(size, b);
    array_view<float, 1> c_view(size, c);
    
    // Clear previous content
    c_view.discard_data();
    
    // Perform computation on GPU
    parallel_for_each(
        c_view.extent,
        [=](index<1> idx) restrict(amp) {
            c_view[idx] = a_view[idx] + b_view[idx];
        }
    );
    
    // Synchronize to get results back to CPU
    c_view.synchronize();
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < size; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cout << "Error at index " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl;
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "Vector addition completed successfully!" << std::endl;
    }
    
    return 0;
}
```

### Tiled Matrix Multiplication

C++ AMP supports tiling, which allows for efficient use of shared memory. Here's a tiled matrix multiplication example:

```cpp
#include <amp.h>
#include <iostream>
#include <vector>

using namespace concurrency;

// Matrix multiplication using tiling
void matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int m, int n, int k) {
    array_view<const float, 2> a_view(m, k, a);
    array_view<const float, 2> b_view(k, n, b);
    array_view<float, 2> c_view(m, n, c);
    
    c_view.discard_data();
    
    // Define tile size
    const int tile_size = 16;
    
    parallel_for_each(
        c_view.extent.tile<tile_size, tile_size>(),
        [=](tiled_index<tile_size, tile_size> t_idx) restrict(amp) {
            int row = t_idx.global[0];
            int col = t_idx.global[1];
            
            float sum = 0.0f;
            
            // Use tile_static for shared memory within a tile
            tile_static float a_tile[tile_size][tile_size];
            tile_static float b_tile[tile_size][tile_size];
            
            for (int i = 0; i < k; i += tile_size) {
                // Load tiles collaboratively
                if (i + t_idx.local[1] < k && row < m) {
                    a_tile[t_idx.local[0]][t_idx.local[1]] = a_view(row, i + t_idx.local[1]);
                } else {
                    a_tile[t_idx.local[0]][t_idx.local[1]] = 0.0f;
                }
                
                if (i + t_idx.local[0] < k && col < n) {
                    b_tile[t_idx.local[0]][t_idx.local[1]] = b_view(i + t_idx.local[0], col);
                } else {
                    b_tile[t_idx.local[0]][t_idx.local[1]] = 0.0f;
                }
                
                // Synchronize to make sure tiles are loaded
                t_idx.barrier.wait();
                
                // Compute partial sum for this tile
                for (int j = 0; j < tile_size; j++) {
                    sum += a_tile[t_idx.local[0]][j] * b_tile[j][t_idx.local[1]];
                }
                
                // Synchronize before loading new tiles
                t_idx.barrier.wait();
            }
            
            // Write result
            if (row < m && col < n) {
                c_view(row, col) = sum;
            }
        }
    );
    
    c_view.synchronize();
}

int main() {
    const int m = 128; // Rows in A
    const int n = 128; // Columns in B
    const int k = 128; // Columns in A / Rows in B
    
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n);
    
    // Initialize matrices
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = static_cast<float>(i + j);
        }
    }
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = static_cast<float>(i - j);
        }
    }
    
    // Perform matrix multiplication
    matrix_multiply(a, b, c, m, n, k);
    
    std::cout << "Matrix multiplication completed!" << std::endl;
    
    return 0;
}
```

### Restrictions in C++ AMP

C++ AMP imposes certain restrictions on code that runs on the GPU:

1. **restrict(amp)**: Code that runs on the GPU must be marked with `restrict(amp)`
2. **Limited C++ features**: Not all C++ features are available in GPU code
3. **No pointers**: Raw pointers are not allowed in GPU code
4. **No virtual functions**: Virtual function calls are not supported
5. **Limited standard library**: Only a subset of the C++ standard library is available

```cpp
// Example of restrictions
void cpu_function() {
    // This can use all C++ features
    std::vector<int> vec;
    vec.push_back(10);
}

void gpu_function(index<1> idx) restrict(amp) {
    // Restricted C++ subset
    // std::vector not allowed
    // No raw pointers
    // No virtual functions
    int value = idx[0] * 2;
}
```

## When to Use DirectCompute or C++ AMP

Here are some considerations for choosing between DirectCompute, C++ AMP, and other GPU computing options:

### Use DirectCompute When:

- You're already using DirectX for graphics and need tight integration
- You need fine-grained control over GPU resources
- You're developing Windows-only applications
- You need to use specific DirectX features
- Performance is critical and you need low-level control

### Use C++ AMP When:

- You want a higher-level, more accessible programming model
- You prefer a more C++-like approach to GPU programming
- You're developing Windows applications but don't need DirectX integration
- You want to minimize boilerplate code
- You're transitioning from CPU to GPU code and want a gentler learning curve

### Use CUDA or OpenCL Instead When:

- You need cross-platform compatibility (OpenCL)
- You're targeting NVIDIA GPUs specifically and want the best performance (CUDA)
- You need access to specialized libraries (e.g., cuDNN, cuBLAS)
- You're developing scientific or HPC applications

## Comparison with CUDA and OpenCL

Here's how DirectCompute and C++ AMP compare to CUDA and OpenCL:

| Feature | DirectCompute | C++ AMP | CUDA | OpenCL |
|---------|---------------|---------|------|--------|
| **Platform Support** | Windows only | Windows only | NVIDIA GPUs | Cross-platform |
| **Language** | HLSL | C++ with extensions | C/C++ with extensions | C-based |
| **Abstraction Level** | Low-level | Higher-level | Low to mid-level | Low-level |
| **Graphics Integration** | Tight DirectX integration | Limited | Separate from graphics | Separate from graphics |
| **Ecosystem** | DirectX | Visual Studio | Rich NVIDIA libraries | Vendor-specific libraries |
| **Learning Curve** | Steep | Moderate | Steep | Steep |
| **Boilerplate Code** | Significant | Minimal | Moderate | Significant |

## Complete DirectCompute Example

Here's a complete example of a DirectCompute application that doubles each element in a buffer:

```cpp
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// HLSL Compute Shader
const char* shaderSource = R"(
    RWBuffer<float> OutputBuffer : register(u0);
    Buffer<float> InputBuffer : register(t0);

    [numthreads(256, 1, 1)]
    void CSMain(uint3 DTid : SV_DispatchThreadID)
    {
        uint index = DTid.x;
        OutputBuffer[index] = InputBuffer[index] * 2.0f;
    }
)";

int main() {
    const int numElements = 1024;
    
    // Initialize input data
    std::vector<float> inputData(numElements);
    for (int i = 0; i < numElements; i++) {
        inputData[i] = static_cast<float>(i);
    }
    
    // Create D3D11 device and context
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL featureLevel;
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,  // Hardware acceleration
        nullptr,                    // No software rasterizer
        0,                          // Flags
        nullptr,                    // Feature levels
        0,                          // Number of feature levels
        D3D11_SDK_VERSION,          // SDK version
        &device,                    // Output device
        &featureLevel,              // Output feature level
        &context                    // Output context
    );
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device" << std::endl;
        return 1;
    }
    
    // Compile compute shader
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    
    hr = D3DCompile(
        shaderSource,            // Shader source
        strlen(shaderSource),    // Source size
        "CSMain",               // Source name
        nullptr,                // Defines
        nullptr,                // Include handler
        "CSMain",               // Entry point
        "cs_5_0",               // Target
        0,                      // Flags1
        0,                      // Flags2
        &shaderBlob,            // Output blob
        &errorBlob              // Error blob
    );
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return 1;
    }
    
    // Create compute shader
    ID3D11ComputeShader* computeShader = nullptr;
    hr = device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &computeShader
    );
    
    shaderBlob->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader" << std::endl;
        return 1;
    }
    
    // Create input buffer
    D3D11_BUFFER_DESC inputDesc = {};
    inputDesc.ByteWidth = sizeof(float) * numElements;
    inputDesc.Usage = D3D11_USAGE_DEFAULT;
    inputDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    inputDesc.CPUAccessFlags = 0;
    inputDesc.StructureByteStride = sizeof(float);
    
    D3D11_SUBRESOURCE_DATA inputData = {};
    inputData.pSysMem = inputData.data();
    
    ID3D11Buffer* inputBuffer = nullptr;
    hr = device->CreateBuffer(&inputDesc, &inputData, &inputBuffer);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create input buffer" << std::endl;
        return 1;
    }
    
    // Create input buffer view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = numElements;
    
    ID3D11ShaderResourceView* inputView = nullptr;
    hr = device->CreateShaderResourceView(inputBuffer, &srvDesc, &inputView);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create input view" << std::endl;
        return 1;
    }
    
    // Create output buffer
    D3D11_BUFFER_DESC outputDesc = {};
    outputDesc.ByteWidth = sizeof(float) * numElements;
    outputDesc.Usage = D3D11_USAGE_DEFAULT;
    outputDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    outputDesc.CPUAccessFlags = 0;
    outputDesc.StructureByteStride = sizeof(float);
    
    ID3D11Buffer* outputBuffer = nullptr;
    hr = device->CreateBuffer(&outputDesc, nullptr, &outputBuffer);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create output buffer" << std::endl;
        return 1;
    }
    
    // Create output buffer view
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = numElements;
    
    ID3D11UnorderedAccessView* outputView = nullptr;
    hr = device->CreateUnorderedAccessView(outputBuffer, &uavDesc, &outputView);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create output view" << std::endl;
        return 1;
    }
    
    // Execute compute shader
    context->CSSetShader(computeShader, nullptr, 0);
    context->CSSetShaderResources(0, 1, &inputView);
    context->CSSetUnorderedAccessViews(0, 1, &outputView, nullptr);
    
    context->Dispatch(numElements / 256, 1, 1);
    
    // Unset resources
    ID3D11UnorderedAccessView* nullUAV = nullptr;
    ID3D11ShaderResourceView* nullSRV = nullptr;
    context->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
    context->CSSetShaderResources(0, 1, &nullSRV);
    
    // Create staging buffer for reading results
    D3D11_BUFFER_DESC stagingDesc = {};
    stagingDesc.ByteWidth = sizeof(float) * numElements;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.StructureByteStride = sizeof(float);
    
    ID3D11Buffer* stagingBuffer = nullptr;
    hr = device->CreateBuffer(&stagingDesc, nullptr, &stagingBuffer);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging buffer" << std::endl;
        return 1;
    }
    
    // Copy output to staging buffer
    context->CopyResource(stagingBuffer, outputBuffer);
    
    // Read results
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = context->Map(stagingBuffer, 0, D3D11_MAP_READ, 0, &mappedResource);
    
    if (SUCCEEDED(hr)) {
        float* resultData = static_cast<float*>(mappedResource.pData);
        
        // Verify results
        bool correct = true;
        for (int i = 0; i < numElements; i++) {
            if (resultData[i] != inputData[i] * 2.0f) {
                std::cout << "Error at index " << i << ": " << resultData[i] 
                          << " != " << (inputData[i] * 2.0f) << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "Computation completed successfully!" << std::endl;
        }
        
        context->Unmap(stagingBuffer, 0);
    }
    
    // Cleanup
    stagingBuffer->Release();
    outputView->Release();
    outputBuffer->Release();
    inputView->Release();
    inputBuffer->Release();
    computeShader->Release();
    context->Release();
    device->Release();
    
    return 0;
}
```

## Complete C++ AMP Example

Here's a complete example of a C++ AMP application that performs matrix multiplication:

```cpp
#include <amp.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace concurrency;
using namespace std::chrono;

// CPU matrix multiplication for comparison
void cpu_matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, 
                        std::vector<float>& c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// C++ AMP matrix multiplication
void amp_matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, 
                        std::vector<float>& c, int m, int n, int k) {
    array_view<const float, 2> a_view(m, k, a);
    array_view<const float, 2> b_view(k, n, b);
    array_view<float, 2> c_view(m, n, c);
    
    c_view.discard_data();
    
    parallel_for_each(
        c_view.extent,
        [=](index<2> idx) restrict(amp) {
            int row = idx[0];
            int col = idx[1];
            
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += a_view(row, i) * b_view(i, col);
            }
            
            c_view[idx] = sum;
        }
    );
    
    c_view.synchronize();
}

// C++ AMP tiled matrix multiplication
void amp_tiled_matrix_multiply(const std::vector<float>& a, const std::vector<float>& b, 
                              std::vector<float>& c, int m, int n, int k) {
    array_view<const float, 2> a_view(m, k, a);
    array_view<const float, 2> b_view(k, n, b);
    array_view<float, 2> c_view(m, n, c);
    
    c_view.discard_data();
    
    const int tile_size = 16;
    
    parallel_for_each(
        c_view.extent.tile<tile_size, tile_size>(),
        [=](tiled_index<tile_size, tile_size> t_idx) restrict(amp) {
            int row = t_idx.global[0];
            int col = t_idx.global[1];
            
            float sum = 0.0f;
            
            tile_static float a_tile[tile_size][tile_size];
            tile_static float b_tile[tile_size][tile_size];
            
            for (int i = 0; i < k; i += tile_size) {
                // Load tiles collaboratively
                if (i + t_idx.local[1] < k && row < m) {
                    a_tile[t_idx.local[0]][t_idx.local[1]] = a_view(row, i + t_idx.local[1]);
                } else {
                    a_tile[t_idx.local[0]][t_idx.local[1]] = 0.0f;
                }
                
                if (i + t_idx.local[0] < k && col < n) {
                    b_tile[t_idx.local[0]][t_idx.local[1]] = b_view(i + t_idx.local[0], col);
                } else {
                    b_tile[t_idx.local[0]][t_idx.local[1]] = 0.0f;
                }
                
                t_idx.barrier.wait();
                
                for (int j = 0; j < tile_size; j++) {
                    sum += a_tile[t_idx.local[0]][j] * b_tile[j][t_idx.local[1]];
                }
                
                t_idx.barrier.wait();
            }
            
            if (row < m && col < n) {
                c_view(row, col) = sum;
            }
        }
    );
    
    c_view.synchronize();
}

int main() {
    // Check available accelerators
    std::vector<accelerator> accs = accelerator::get_all();
    std::wcout << "Available accelerators:" << std::endl;
    for (auto& acc : accs) {
        std::wcout << acc.description << std::endl;
    }
    
    // Select default accelerator
    accelerator default_acc;
    std::wcout << "Using: " << default_acc.description << std::endl;
    
    // Matrix dimensions
    const int m = 1024; // Rows in A
    const int n = 1024; // Columns in B
    const int k = 1024; // Columns in A / Rows in B
    
    // Initialize matrices
    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c_cpu(m * n);
    std::vector<float> c_amp(m * n);
    std::vector<float> c_amp_tiled(m * n);
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = static_cast<float>((i + j) % 10);
        }
    }
    
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = static_cast<float>((i - j) % 10);
        }
    }
    
    // CPU matrix multiplication
    auto cpu_start = high_resolution_clock::now();
    cpu_matrix_multiply(a, b, c_cpu, m, n, k);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();
    
    std::cout << "CPU matrix multiplication: " << cpu_duration << " ms" << std::endl;
    
    // C++ AMP matrix multiplication
    auto amp_start = high_resolution_clock::now();
    amp_matrix_multiply(a, b, c_amp, m, n, k);
    auto amp_end = high_resolution_clock::now();
    auto amp_duration = duration_cast<milliseconds>(amp_end - amp_start).count();
    
    std::cout << "C++ AMP matrix multiplication: " << amp_duration << " ms" << std::endl;
    
    // C++ AMP tiled matrix multiplication
    auto amp_tiled_start = high_resolution_clock::now();
    amp_tiled_matrix_multiply(a, b, c_amp_tiled, m, n, k);
    auto amp_tiled_end = high_resolution_clock::now();
    auto amp_tiled_duration = duration_cast<milliseconds>(amp_tiled_end - amp_tiled_start).count();
    
    std::cout << "C++ AMP tiled matrix multiplication: " << amp_tiled_duration << " ms" << std::endl;
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        if (std::abs(c_cpu[i] - c_amp[i]) > 0.001f) {
            std::cout << "Error at index " << i << ": CPU = " << c_cpu[i] 
                      << ", AMP = " << c_amp[i] << std::endl;
            correct = false;
            break;
        }
        
        if (std::abs(c_cpu[i] - c_amp_tiled[i]) > 0.001f) {
            std::cout << "Error at index " << i << ": CPU = " << c_cpu[i] 
                      << ", AMP tiled = " << c_amp_tiled[i] << std::endl;
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "All computations produced correct results!" << std::endl;
    }
    
    // Performance comparison
    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "CPU: " << cpu_duration << " ms (baseline)" << std::endl;
    std::cout << "C++ AMP: " << amp_duration << " ms (" 
              << static_cast<float>(cpu_duration) / amp_duration << "x speedup)" << std::endl;
    std::cout << "C++ AMP tiled: " << amp_tiled_duration << " ms (" 
              << static_cast<float>(cpu_duration) / amp_tiled_duration << "x speedup)" << std::endl;
    
    return 0;
}
```

## Conclusion

DirectCompute and C++ AMP provide Windows-specific alternatives for GPU computing, each with its own strengths:

- **DirectCompute** offers low-level control and tight integration with DirectX, making it ideal for graphics applications that need compute capabilities.

- **C++ AMP** provides a higher-level, more accessible programming model that extends C++ with parallel constructs, making it easier to transition from CPU to GPU programming.

While these technologies are limited to Windows platforms, they offer advantages for developers already working within the Microsoft ecosystem. DirectCompute is particularly valuable for applications that combine graphics and compute workloads, while C++ AMP offers a gentler learning curve for developers new to GPU programming.

In our next article, we'll explore GPU performance optimization techniques, focusing on profiling, memory access patterns, and other strategies to maximize GPU performance across different frameworks.

## Exercises for Practice

1. **DirectCompute Image Processing**: Implement a simple image processing filter (e.g., Gaussian blur or edge detection) using DirectCompute.

2. **C++ AMP Vector Operations**: Create a library of vector operations (addition, subtraction, dot product, cross product) using C++ AMP.

3. **Performance Comparison**: Implement the same algorithm (e.g., matrix multiplication) using DirectCompute, C++ AMP, and CPU code, and compare their performance.

4. **DirectX Integration**: Create a DirectX application that uses DirectCompute to generate or modify data that is then rendered using the graphics pipeline.

## Further Resources

- [DirectCompute Programming Guide](https://docs.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader)
- [C++ AMP Documentation](https://docs.microsoft.com/en-us/cpp/parallel/amp/cpp-amp-overview?view=msvc-160)
- [Programming Windows Store Apps with C++ AMP](https://www.microsoft.com/en-us/download/details.aspx?id=41638)
- [DirectX Graphics Samples](https://github.com/microsoft/DirectX-Graphics-Samples)
- [C++ AMP Samples](https://github.com/microsoft/cpp-docs/tree/master/docs/parallel/amp/code-samples)