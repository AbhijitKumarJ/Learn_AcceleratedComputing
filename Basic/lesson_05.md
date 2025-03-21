# Lesson 5: AMD's GPU Computing with ROCm

Welcome to the fifth lesson in our "Accelerating the Future" series. In this lesson, we'll explore AMD's approach to GPU computing through their ROCm platform, providing an alternative to NVIDIA's CUDA ecosystem that we covered in the previous lesson.

## Introduction to AMD's GPU Architecture

AMD's Graphics Processing Units (GPUs) have evolved significantly over the years, with their current RDNA (for gaming) and CDNA (for computing) architectures representing their latest innovations.

### RDNA Architecture (Radeon DNA)
- Designed primarily for gaming and graphics workloads
- Features Compute Units (CUs) as the basic building blocks
- Optimized for high performance in real-time rendering
- Found in Radeon RX series consumer graphics cards

### CDNA Architecture (Compute DNA)
- Specifically designed for high-performance computing and data centers
- Optimized for scientific computing, AI, and machine learning
- Features enhanced double-precision floating-point performance
- Includes Matrix Cores for accelerated matrix operations
- Found in AMD Instinct accelerators

At the heart of AMD's GPU architecture are **Compute Units (CUs)**, which are roughly equivalent to NVIDIA's Streaming Multiprocessors (SMs). Each CU contains:
- Multiple SIMD (Single Instruction, Multiple Data) units
- Vector and scalar processing elements
- Local data share (equivalent to NVIDIA's shared memory)
- Texture units and other specialized hardware

AMD GPUs organize threads into **wavefronts** (similar to NVIDIA's warps), which are groups of threads that execute together. In AMD GPUs, a wavefront typically consists of 64 threads (compared to 32 in NVIDIA's architecture).

## What is ROCm and how it compares to CUDA

**ROCm** (Radeon Open Compute) is AMD's open-source platform for GPU computing. Launched in 2016, ROCm represents AMD's commitment to open-source software in the high-performance computing space.

Key components of the ROCm platform include:
- **ROCr**: The runtime system that manages GPU execution
- **HIP**: A C++ runtime API and programming language for GPU computing
- **ROCm Libraries**: Optimized libraries for common operations (math, deep learning, etc.)
- **ROCm Compilers**: Tools to compile code for AMD GPUs
- **Profiling and Debugging Tools**: For performance analysis and troubleshooting

### ROCm vs. CUDA: Key Differences

| Feature | ROCm | CUDA |
|---------|------|------|
| **License** | Open-source | Proprietary |
| **Hardware Support** | AMD GPUs (with limited support for some NVIDIA GPUs) | NVIDIA GPUs only |
| **Maturity** | Newer, still evolving | Well-established |
| **Ecosystem** | Growing, but smaller | Extensive |
| **Programming Model** | HIP (with compatibility layers for CUDA) | CUDA C/C++ |
| **Library Support** | Growing set of libraries | Comprehensive libraries |
| **Community** | Smaller but growing | Large and active |
| **Documentation** | Improving but less comprehensive | Extensive |

The open-source nature of ROCm is one of its most significant advantages, allowing for community contributions and adaptations for specific needs. However, CUDA still maintains an edge in terms of ecosystem maturity and available resources.

## The HIP Programming Model: Writing Portable GPU Code

**HIP** (Heterogeneous-Computing Interface for Portability) is a C++ runtime API and programming language that allows developers to write code that can run on both AMD and NVIDIA GPUs. HIP is designed to be very similar to CUDA, making it easier for developers familiar with CUDA to transition to AMD hardware.

Key features of HIP include:
- CUDA-like syntax and programming model
- Support for both AMD and NVIDIA GPUs
- Tools to automatically convert CUDA code to HIP
- Minimal performance overhead compared to native implementations

### Basic HIP Program Structure

A typical HIP program follows a structure similar to CUDA:

```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Kernel definition
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    // Copy data from host to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vectorAdd, 
                       dim3(blocksPerGrid), 
                       dim3(threadsPerBlock), 
                       0, 0, 
                       d_A, d_B, d_C, N);
    
    // Copy result back to host
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    // Verify result
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }
    
    std::cout << "Vector addition " 
              << (success ? "successful" : "failed") << std::endl;
    
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return 0;
}
```

If you compare this to a CUDA program, you'll notice many similarities. The main differences are:
- The inclusion of `hip/hip_runtime.h` instead of CUDA headers
- The use of `hipMalloc`, `hipMemcpy`, and `hipFree` instead of their CUDA counterparts
- The kernel launch syntax using `hipLaunchKernelGGL` (though HIP also supports the `<<<>>>` syntax with some compilers)

## Converting CUDA Code to HIP: Basic Principles

AMD provides tools to help convert CUDA code to HIP, making it easier to port existing applications to run on AMD hardware. The primary tool is `hipify`, which comes in two variants:

1. **hipify-perl**: A Perl script that performs source-to-source translation
2. **hipify-clang**: A more robust tool based on Clang that provides better code analysis and transformation

The basic conversion process involves:

1. **API Mapping**: CUDA functions are mapped to their HIP equivalents
   - `cudaMalloc` → `hipMalloc`
   - `cudaMemcpy` → `hipMemcpy`
   - `cudaFree` → `hipFree`

2. **Header Replacement**:
   - `cuda.h` → `hip/hip_runtime.h`
   - `cuda_runtime.h` → `hip/hip_runtime.h`

3. **Kernel Launch Syntax**:
   - CUDA: `myKernel<<<grid, block>>>(args);`
   - HIP: `hipLaunchKernelGGL(myKernel, grid, block, 0, 0, args);`
     (though HIP also supports the `<<<>>>` syntax in some cases)

4. **Type Renaming**:
   - `cudaStream_t` → `hipStream_t`
   - `cudaEvent_t` → `hipEvent_t`

5. **Device Functions**:
   - `__device__` functions work the same way in HIP
   - Some built-in CUDA functions have HIP equivalents with different names

Here's a simple example of converting a CUDA function to HIP:

**CUDA Version:**
```cpp
#include <cuda_runtime.h>

__global__ void addKernel(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void addWithCuda(int *a, int *b, int *c, int size) {
    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;
    
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    
    addKernel<<<1, size>>>(dev_a, dev_b, dev_c);
    
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
```

**HIP Version:**
```cpp
#include <hip/hip_runtime.h>

__global__ void addKernel(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void addWithHIP(int *a, int *b, int *c, int size) {
    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;
    
    hipMalloc((void**)&dev_a, size * sizeof(int));
    hipMalloc((void**)&dev_b, size * sizeof(int));
    hipMalloc((void**)&dev_c, size * sizeof(int));
    
    hipMemcpy(dev_a, a, size * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(dev_b, b, size * sizeof(int), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(addKernel, dim3(1), dim3(size), 0, 0, dev_a, dev_b, dev_c);
    
    hipMemcpy(c, dev_c, size * sizeof(int), hipMemcpyDeviceToHost);
    
    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);
}
```

## Simple example of a ROCm/HIP program

Let's look at a more complete example of a HIP program that performs matrix multiplication:

```cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

// Matrix dimensions
#define WIDTH 1024

// Kernel function for matrix multiplication
__global__ void matrixMul(float* A, float* B, float* C, int width) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // Perform dot product for this element
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        
        // Store the result
        C[row * width + col] = sum;
    }
}

int main() {
    // Matrix size in bytes
    size_t bytes = WIDTH * WIDTH * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[WIDTH * WIDTH];
    float *h_B = new float[WIDTH * WIDTH];
    float *h_C = new float[WIDTH * WIDTH];
    
    // Initialize matrices with random values
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);
    
    // Copy data from host to device
    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
    
    // Set up execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (WIDTH + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel
    hipLaunchKernelGGL(matrixMul,
                       blocksPerGrid,
                       threadsPerBlock,
                       0, 0,
                       d_A, d_B, d_C, WIDTH);
    
    // Check for errors
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        std::cerr << "Kernel launch failed: " << hipGetErrorString(error) << std::endl;
        return -1;
    }
    
    // Copy result back to host
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);
    
    // Verify result (checking just a few elements)
    std::cout << "Verification (first few elements):" << std::endl;
    for (int i = 0; i < 5; i++) {
        float sum = 0.0f;
        for (int k = 0; k < WIDTH; k++) {
            sum += h_A[i * WIDTH + k] * h_B[k * WIDTH + i];
        }
        std::cout << "C[" << i << "," << i << "]: GPU = " << h_C[i * WIDTH + i]
                  << ", CPU = " << sum << std::endl;
    }
    
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return 0;
}
```

This program:
1. Allocates memory for three matrices (A, B, and C)
2. Initializes A and B with random values
3. Copies A and B to the GPU
4. Performs matrix multiplication on the GPU
5. Copies the result back to the CPU
6. Verifies a few elements of the result
7. Frees all allocated memory

## AMD's approach to open-source GPU computing

AMD has taken a distinctly different approach from NVIDIA by embracing open-source software for GPU computing. This strategy has several key aspects:

### 1. Open-Source Software Stack
The entire ROCm platform is open-source, allowing developers to:
- Examine the source code
- Contribute improvements
- Customize the platform for specific needs
- Integrate ROCm into other open-source projects

### 2. Linux Kernel Integration
AMD has worked to integrate their GPU drivers directly into the Linux kernel, rather than relying solely on proprietary drivers. This approach:
- Improves compatibility with Linux distributions
- Reduces maintenance burden for system administrators
- Aligns with the open-source philosophy of many HPC environments

### 3. Community Collaboration
AMD actively collaborates with:
- Open-source software communities
- Academic institutions
- Industry partners
- Standards organizations

### 4. Compatibility Layers
AMD provides tools to make transitioning from CUDA easier:
- HIP for code portability
- Compatibility libraries that implement CUDA API functionality
- Conversion tools like hipify

### 5. Support for Industry Standards
AMD supports open standards like:
- OpenCL
- OpenMP
- DirectX Compute
- Vulkan Compute

This open approach has both advantages and challenges:

**Advantages:**
- Greater transparency
- Community-driven improvements
- Better integration with open-source ecosystems
- No vendor lock-in
- Potential for broader hardware support

**Challenges:**
- Smaller ecosystem compared to CUDA
- Fewer learning resources
- Less mature tooling
- Performance optimization can be more challenging

## When to choose AMD GPUs for compute workloads

Choosing between AMD and NVIDIA GPUs for compute workloads depends on several factors:

### Scenarios favoring AMD GPUs:

1. **Open-source requirements**
   - Projects that require or prefer open-source solutions
   - Organizations with open-source policies
   - Academic environments focused on transparency

2. **Cost considerations**
   - AMD GPUs often provide better performance per dollar
   - Lower total cost of ownership in some scenarios
   - No licensing costs for development tools

3. **Specific performance advantages**
   - Some AMD GPUs excel at certain workloads
   - Higher memory bandwidth in some models
   - Better double-precision performance in some cases

4. **Avoiding vendor lock-in**
   - Organizations wanting hardware flexibility
   - Projects that may need to run on different hardware
   - Long-term sustainability concerns

5. **Linux integration**
   - Better kernel integration
   - Smoother experience with open-source Linux distributions
   - Easier deployment in some HPC environments

### Considerations before choosing AMD:

1. **Existing codebase**
   - Extensive CUDA codebases may require significant porting effort
   - Some CUDA features have no direct HIP equivalent
   - Performance tuning may need to be redone

2. **Library and framework support**
   - Check if the libraries you need are available for ROCm
   - Some specialized CUDA libraries may not have ROCm equivalents
   - Framework support (like deep learning frameworks) may vary

3. **Development tools**
   - Profiling and debugging tools are less mature
   - IDE integration may be more limited
   - Documentation and examples are less comprehensive

4. **Community and support**
   - Smaller community means fewer online resources
   - Fewer third-party tutorials and examples
   - Commercial support options may be more limited

## Resources for learning more about ROCm

### Official Documentation
- [ROCm Documentation](https://rocmdocs.amd.com/en/latest/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
- [AMD GPU ISA Documentation](https://developer.amd.com/resources/developer-guides-manuals/)

### Tutorials and Examples
- [ROCm GitHub Repository](https://github.com/RadeonOpenCompute/ROCm)
- [HIP Samples](https://github.com/ROCm-Developer-Tools/HIP-Examples)
- [AMD ROCm Learning Center](https://developer.amd.com/resources/rocm-learning-center/)

### Community Resources
- [ROCm GitHub Issues](https://github.com/RadeonOpenCompute/ROCm/issues)
- [AMD ROCm Forum](https://community.amd.com/t5/ROCm/bd-p/rocm)
- [Stack Overflow ROCm Tag](https://stackoverflow.com/questions/tagged/rocm)

### Books and Publications
- "Heterogeneous Computing with OpenCL 2.0" (while focused on OpenCL, covers concepts relevant to ROCm)
- "Professional CUDA C Programming" (useful for understanding concepts that transfer to HIP)
- AMD Research Publications: [AMD Research](https://www.amd.com/en/corporate/research.html)

### Online Courses
- Coursera and edX courses on parallel programming (while not ROCm-specific, they teach relevant concepts)
- AMD Developer YouTube channel: [AMD Developer](https://www.youtube.com/c/AMDDeveloper)

## Key terminology definitions

- **ROCm (Radeon Open Compute)**: AMD's open-source platform for GPU computing
- **HIP (Heterogeneous-Computing Interface for Portability)**: A C++ runtime API for writing portable GPU code
- **Compute Unit (CU)**: The basic building block of AMD GPUs, similar to NVIDIA's SM
- **Wavefront**: A group of 64 threads that execute together on AMD GPUs (equivalent to NVIDIA's warp)
- **CDNA (Compute DNA)**: AMD's architecture for compute-focused GPUs
- **RDNA (Radeon DNA)**: AMD's architecture for graphics-focused GPUs
- **hipify**: Tools for converting CUDA code to HIP
- **Work-group**: A group of threads that can cooperate (equivalent to CUDA's thread block)
- **Work-item**: An individual thread of execution (equivalent to CUDA's thread)
- **Local Data Share (LDS)**: Fast memory shared within a work-group (equivalent to CUDA's shared memory)
- **AMD Instinct**: AMD's line of data center GPUs designed for HPC and AI workloads

## Try it yourself: Converting a CUDA program to HIP

Try converting this simple CUDA vector addition program to HIP:

```cpp
// CUDA Vector Addition
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Initialize the host arrays
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device arrays
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy the host data to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Addition CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // Copy the device result to the host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(1);
        }
    }
    printf("Test PASSED\n");

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

**Solution:**
```cpp
// HIP Vector Addition
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Initialize the host arrays
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host data
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device arrays
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    hipMalloc((void **)&d_A, size);
    hipMalloc((void **)&d_B, size);
    hipMalloc((void **)&d_C, size);

    // Copy the host data to the device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Launch the Vector Addition HIP Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    hipLaunchKernelGGL(vectorAdd, 
                       dim3(blocksPerGrid), 
                       dim3(threadsPerBlock), 
                       0, 0, 
                       d_A, d_B, d_C, numElements);

    // Copy the device result to the host
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(1);
        }
    }
    printf("Test PASSED\n");

    // Free device global memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

## Quick recap and preview of next lesson

In this lesson, we've covered:
- AMD's GPU architecture and how it compares to NVIDIA's
- The ROCm platform and its components
- The HIP programming model for portable GPU code
- How to convert CUDA code to HIP
- A practical example of a HIP program
- AMD's open-source approach to GPU computing
- When to choose AMD GPUs for compute workloads
- Resources for learning more about ROCm

In the next lesson, we'll explore NVIDIA's Tensor Cores, which are specialized hardware units designed to accelerate matrix operations for deep learning. We'll learn what Tensor Cores are, how they work, and the impact they've had on AI training and inference performance.

---

*Remember: The field of GPU computing is constantly evolving, with both AMD and NVIDIA regularly releasing new hardware and software. While the concepts covered in this lesson will remain relevant, specific details about performance, features, and compatibility may change over time.*