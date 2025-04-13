# GPU Performance Optimization Techniques

*Welcome to the ninth installment of our GPU programming series! In this article, we'll dive into performance optimization techniques for GPU computing. Even with the massive parallelism offered by GPUs, achieving optimal performance requires careful attention to how you structure your code, manage memory, and utilize hardware resources.*

## Introduction to GPU Performance Optimization

GPU performance optimization is both an art and a science. While GPUs offer tremendous computational power, extracting maximum performance requires understanding the underlying hardware architecture and the specific characteristics of your workload. In this article, we'll explore key techniques for optimizing GPU code, focusing on profiling, memory access patterns, occupancy, and other critical factors.

## Profiling GPU Code

Before optimizing, you need to identify performance bottlenecks. GPU profiling tools help you understand where your application spends time and resources.

### NVIDIA Nsight and Visual Profiler

For CUDA applications, NVIDIA provides powerful profiling tools:

- **Nsight Systems**: For system-wide performance analysis
- **Nsight Compute**: For detailed kernel analysis
- **NVIDIA Visual Profiler**: The classic profiling tool (being phased out in favor of Nsight)

```bash
# Running Nsight Systems from command line
nsys profile --stats=true ./my_application

# Running Nsight Compute from command line
ncu --metrics all ./my_application
```

Key metrics to look for in NVIDIA profilers:

- **SM Occupancy**: How efficiently you're using the streaming multiprocessors
- **Memory Throughput**: Global, shared, and texture memory bandwidth utilization
- **Instruction Throughput**: Arithmetic, load/store, and control flow instruction rates
- **Warp Execution Efficiency**: Percentage of active threads in warps
- **Stall Reasons**: What's causing warps to stall (memory dependencies, synchronization, etc.)

![NVIDIA Nsight Profiler](https://via.placeholder.com/800x400?text=NVIDIA+Nsight+Profiler)

### AMD Radeon GPU Profiler

For AMD GPUs, the Radeon GPU Profiler (RGP) provides detailed performance analysis:

```bash
# Capturing a profile with RGP
rgpProfiler --application ./my_application
```

### Intel Graphics Performance Analyzers

For Intel GPUs, Intel Graphics Performance Analyzers (GPA) offer profiling capabilities:

```bash
# Using Intel GPA Framework to profile an application
gpa-counter-report ./my_application
```

### Vendor-Neutral Profiling

For cross-platform profiling, consider tools like:

- **Remotery**: A real-time CPU/GPU profiler
- **Tracy Profiler**: A frame profiler for games and real-time applications
- **Chrome Tracing**: Can be used with custom instrumentation

### Custom Timing Measurements

You can also implement custom timing in your code:

```cpp
// CUDA example with custom timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// Launch kernel
myKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel execution time: %f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

```cpp
// OpenCL example with custom timing
cl_event event;
cl_ulong start_time, end_time;

clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &event);
clWaitForEvents(1, &event);

clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);

double milliseconds = (end_time - start_time) * 1.0e-6;
printf("Kernel execution time: %f ms\n", milliseconds);

clReleaseEvent(event);
```

## Memory Coalescing and Bank Conflicts

Memory access patterns have a profound impact on GPU performance. Two critical concepts are memory coalescing and bank conflicts.

### Global Memory Coalescing

Memory coalescing refers to combining multiple memory accesses into fewer transactions. When threads in a warp access contiguous memory locations, the hardware can coalesce these into a single transaction.

#### Non-Coalesced vs. Coalesced Access

```cuda
// Non-coalesced access (poor performance)
__global__ void nonCoalescedKernel(float* input, float* output, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    if (x < width && y < height) {
        // Column-major access pattern in row-major stored array
        // Adjacent threads access memory with stride 'width'
        output[x * height + y] = input[x * height + y] * 2.0f;
    }
}

// Coalesced access (good performance)
__global__ void coalescedKernel(float* input, float* output, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    if (x < width && y < height) {
        // Row-major access pattern in row-major stored array
        // Adjacent threads access adjacent memory
        output[y * width + x] = input[y * width + x] * 2.0f;
    }
}
```

![Memory Coalescing Diagram](https://via.placeholder.com/800x300?text=Memory+Coalescing+Diagram)

#### Optimizing Struct of Arrays vs. Array of Structs

```cuda
// Array of Structs (AoS) - Often leads to non-coalesced access
struct Particle {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
};

__global__ void updateParticlesAoS(Particle* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles[idx].x += particles[idx].vx;
        particles[idx].y += particles[idx].vy;
        particles[idx].z += particles[idx].vz;
    }
}

// Struct of Arrays (SoA) - Better for coalescing
struct ParticleSystem {
    float *x, *y, *z;    // Positions
    float *vx, *vy, *vz; // Velocities
};

__global__ void updateParticlesSoA(ParticleSystem particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        particles.x[idx] += particles.vx[idx];
        particles.y[idx] += particles.vy[idx];
        particles.z[idx] += particles.vz[idx];
    }
}
```

### Shared Memory Bank Conflicts

Shared memory is divided into banks that can be accessed simultaneously. A bank conflict occurs when multiple threads in a warp access different addresses in the same bank, causing serialization.

```cuda
// Kernel with bank conflicts
__global__ void bankConflictKernel(float* input, float* output, int n) {
    __shared__ float sharedData[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    __syncthreads();
    
    // This access pattern may cause bank conflicts
    // If shared memory has 32 banks, threads will access the same bank
    // when the stride is 32 (or a multiple of 32)
    float value = sharedData[threadIdx.x * 32 % 256];
    
    if (idx < n) {
        output[idx] = value;
    }
}

// Kernel without bank conflicts
__global__ void noBankConflictKernel(float* input, float* output, int n) {
    __shared__ float sharedData[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < n) {
        sharedData[threadIdx.x] = input[idx];
    }
    __syncthreads();
    
    // This access pattern avoids bank conflicts
    // Using a prime number stride ensures different banks are accessed
    float value = sharedData[threadIdx.x * 17 % 256];
    
    if (idx < n) {
        output[idx] = value;
    }
}
```

![Shared Memory Bank Conflicts](https://via.placeholder.com/800x300?text=Shared+Memory+Bank+Conflicts)

#### Padding to Avoid Bank Conflicts

```cuda
// Using padding to avoid bank conflicts
__global__ void paddedSharedMemKernel(float* input, float* output, int width, int height) {
    // Add padding to avoid bank conflicts
    // For a 32-bank system, we add 1 element of padding per row
    __shared__ float sharedData[32][33]; // 33 instead of 32 for padding
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    if (x < width && y < height) {
        // Load data into shared memory
        sharedData[ty][tx] = input[y * width + x];
    }
    __syncthreads();
    
    // Process data (e.g., 2D stencil)
    if (tx > 0 && tx < 31 && ty > 0 && ty < 31 && x < width && y < height) {
        float result = 0.2f * (sharedData[ty][tx] + 
                               sharedData[ty-1][tx] + 
                               sharedData[ty+1][tx] + 
                               sharedData[ty][tx-1] + 
                               sharedData[ty][tx+1]);
        output[y * width + x] = result;
    }
}
```

## Occupancy and Latency Hiding

Occupancy refers to the ratio of active warps to the maximum number of warps supported by a streaming multiprocessor (SM). Higher occupancy can help hide memory and instruction latency.

### Factors Affecting Occupancy

1. **Register Usage**: Each thread uses registers, and excessive register usage can limit occupancy
2. **Shared Memory Usage**: Shared memory is allocated per block, limiting the number of blocks per SM
3. **Block Size**: The number of threads per block affects how many blocks can run concurrently
4. **Hardware Limits**: Maximum warps per SM, maximum blocks per SM, etc.

### Calculating Theoretical Occupancy

NVIDIA provides the CUDA Occupancy Calculator to help determine theoretical occupancy:

```python
# Example using CUDA Python tools
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Compile a simple kernel
mod = SourceModule("""
__global__ void my_kernel(float *dest, float *a, float *b) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    dest[i] = a[i] + b[i];
}
""")

kernel = mod.get_function("my_kernel")

# Get kernel attributes
attrs = kernel.get_attributes()
print(f"Registers per thread: {attrs['num_regs']}")
print(f"Shared memory per block: {attrs['shared_size_bytes']} bytes")

# Calculate occupancy
block_size = 256
theoretical_occupancy = cuda.occupancy.max_active_blocks_per_multiprocessor(kernel, block_size, 0)
max_blocks_per_sm = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR)
max_threads_per_sm = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)

occupancy = (theoretical_occupancy * block_size) / max_threads_per_sm
print(f"Theoretical occupancy: {occupancy * 100:.2f}%")
```

### Optimizing Register Usage

You can control register usage with compiler flags or pragmas:

```cuda
// Limit registers per thread to 32
#pragma nv_attr(maxrregcount=32)
__global__ void limitedRegisterKernel(float* input, float* output, int n) {
    // Kernel code
}
```

Compile with register limits:

```bash
nvcc -maxrregcount=32 my_kernel.cu -o my_program
```

### Optimizing Block Size

Choosing the right block size is crucial for occupancy:

```cuda
// Helper function to find optimal block size
int findOptimalBlockSize(void (*kernel)(float*, float*, int), int sharedMemBytes) {
    int blockSizes[] = {128, 256, 512, 1024};
    int maxOccupancy = 0;
    int optimalBlockSize = 128;
    
    for (int i = 0; i < 4; i++) {
        int blockSize = blockSizes[i];
        int numBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, sharedMemBytes);
        
        int occupancy = numBlocks * blockSize;
        if (occupancy > maxOccupancy) {
            maxOccupancy = occupancy;
            optimalBlockSize = blockSize;
        }
    }
    
    return optimalBlockSize;
}
```

### Dynamic Parallelism

CUDA dynamic parallelism allows kernels to launch other kernels, which can help optimize workloads with irregular parallelism:

```cuda
__global__ void childKernel(float* data, int idx, int range) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < range) {
        data[idx + i] *= 2.0f;
    }
}

__global__ void parentKernel(float* data, int* indices, int* ranges, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        if (ranges[i] > 32) { // Only launch child kernel for larger ranges
            int blocks = (ranges[i] + 255) / 256;
            childKernel<<<blocks, 256>>>(data, indices[i], ranges[i]);
        } else {
            // Process small ranges directly
            for (int j = 0; j < ranges[i]; j++) {
                data[indices[i] + j] *= 2.0f;
            }
        }
    }
}
```

## Shared Memory Optimization Patterns

Shared memory is a powerful tool for optimization, but it must be used effectively.

### Tiling for Stencil Operations

Tiling involves loading a block of data into shared memory to reduce global memory accesses:

```cuda
__global__ void stencil1D(float* input, float* output, int n) {
    __shared__ float tile[BLOCK_SIZE + 2]; // +2 for halo regions
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tileIdx = threadIdx.x + 1; // +1 to account for halo
    
    // Load main data
    if (idx < n) {
        tile[tileIdx] = input[idx];
    }
    
    // Load halo regions
    if (threadIdx.x == 0 && idx > 0) {
        tile[0] = input[idx - 1];
    }
    if (threadIdx.x == blockDim.x - 1 && idx < n - 1) {
        tile[tileIdx + 1] = input[idx + 1];
    }
    
    __syncthreads();
    
    // Compute stencil
    if (idx < n) {
        output[idx] = 0.3f * tile[tileIdx - 1] + 0.4f * tile[tileIdx] + 0.3f * tile[tileIdx + 1];
    }
}
```

### Shared Memory for Reduction Operations

Shared memory is ideal for reduction operations like sum, min, max, etc.:

```cuda
__global__ void reduceSum(float* input, float* output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Optimized Reduction with Warp Shuffle

Modern GPUs support warp shuffle operations that allow threads within a warp to exchange values directly:

```cuda
__global__ void reduceSum_shuffle(float* input, float* output, int n) {
    __shared__ float sdata[32]; // One element per warp
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % 32; // Lane index within warp
    int warpId = tid / 32; // Warp index
    
    // Each thread loads one element
    float sum = (idx < n) ? input[idx] : 0;
    
    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes result to shared memory
    if (lane == 0) {
        sdata[warpId] = sum;
    }
    
    __syncthreads();
    
    // Final reduction (only first warp)
    if (warpId == 0) {
        sum = (tid < blockDim.x / 32) ? sdata[lane] : 0;
        
        // Warp-level reduction again
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread writes final result
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}
```

## Advanced Optimization Techniques

### Stream Processing for Overlapping Operations

CUDA streams allow overlapping kernel execution with data transfers:

```cpp
// Create streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Divide work between streams
int halfSize = n / 2;
size_t halfBytes = halfSize * sizeof(float);

// Stream 1: first half of data
cudaMemcpyAsync(d_input, h_input, halfBytes, cudaMemcpyHostToDevice, stream1);
myKernel<<<blocks/2, threads, 0, stream1>>>(d_input, d_output, halfSize);
cudaMemcpyAsync(h_output, d_output, halfBytes, cudaMemcpyDeviceToHost, stream1);

// Stream 2: second half of data
cudaMemcpyAsync(d_input + halfSize, h_input + halfSize, halfBytes, cudaMemcpyHostToDevice, stream2);
myKernel<<<blocks/2, threads, 0, stream2>>>(d_input + halfSize, d_output + halfSize, halfSize);
cudaMemcpyAsync(h_output + halfSize, d_output + halfSize, halfBytes, cudaMemcpyDeviceToHost, stream2);

// Synchronize all streams
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// Cleanup
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### Persistent Threads

The persistent threads pattern keeps GPU threads active for the entire application lifetime:

```cuda
__global__ void persistentKernel(float* data, int* taskQueue, int* taskCount, int maxTasks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (true) {
        // Atomically get next task
        int taskId = atomicAdd(taskCount, 1);
        
        if (taskId >= maxTasks) {
            break; // No more tasks
        }
        
        // Process task
        int taskType = taskQueue[taskId * 2];
        int taskParam = taskQueue[taskId * 2 + 1];
        
        switch (taskType) {
            case 0: // Addition
                data[taskParam] += 1.0f;
                break;
            case 1: // Multiplication
                data[taskParam] *= 2.0f;
                break;
            // Other task types...
        }
    }
}
```

### Loop Unrolling

Unrolling loops can reduce instruction overhead:

```cuda
// Without loop unrolling
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// With manual loop unrolling (4x)
__global__ void vectorAdd_unrolled(float* a, float* b, float* c, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    
    if (idx + 3 < n) {
        c[idx] = a[idx] + b[idx];
        c[idx+1] = a[idx+1] + b[idx+1];
        c[idx+2] = a[idx+2] + b[idx+2];
        c[idx+3] = a[idx+3] + b[idx+3];
    } else {
        // Handle boundary cases
        for (int i = 0; i < 4 && idx + i < n; i++) {
            c[idx+i] = a[idx+i] + b[idx+i];
        }
    }
}
```

You can also use compiler pragmas for automatic unrolling:

```cuda
__global__ void vectorAdd_pragma(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    #pragma unroll 4
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}
```

### Instruction-Level Optimization

Using intrinsic functions can improve performance for specific operations:

```cuda
__global__ void fastMathKernel(float* input, float* output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Use fast math intrinsics
        float x = input[idx];
        output[idx] = __fmul_rn(x, __fadd_rn(x, 1.0f)); // x * (x + 1)
    }
}
```

Compile with fast math options:

```bash
nvcc --use_fast_math my_kernel.cu -o my_program
```

## Case Study: Optimizing Matrix Multiplication

Let's apply these optimization techniques to matrix multiplication:

### Naive Implementation

```cuda
__global__ void matrixMul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Shared Memory Tiling

```cuda
__global__ void matrixMul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles collaboratively
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Further Optimizations

```cuda
// Using vectorized loads and register blocking
__global__ void matrixMul_optimized(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    const int VECTOR_SIZE = 4; // Process 4 elements at once
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Register blocking: each thread computes a 2x2 block of outputs
    float sum[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles using vectorized loads where possible
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            // Use float4 for coalesced loads if alignment permits
            if ((t * TILE_SIZE + threadIdx.x) % VECTOR_SIZE == 0 && threadIdx.x < TILE_SIZE - VECTOR_SIZE + 1) {
                float4 data = *reinterpret_cast<float4*>(&A[row * K + t * TILE_SIZE + threadIdx.x]);
                As[threadIdx.y][threadIdx.x] = data.x;
                As[threadIdx.y][threadIdx.x + 1] = data.y;
                As[threadIdx.y][threadIdx.x + 2] = data.z;
                As[threadIdx.y][threadIdx.x + 3] = data.w;
            } else {
                As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
            }
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Similar vectorized loading for B
        // ...
        
        __syncthreads();
        
        // Compute partial sums with register blocking
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Each thread computes a 2x2 block of outputs
            if (row < M && row + 1 < M && col < N && col + 1 < N) {
                float a0 = As[threadIdx.y][k];
                float a1 = As[threadIdx.y + 1][k];
                float b0 = Bs[k][threadIdx.x];
                float b1 = Bs[k][threadIdx.x + 1];
                
                sum[0][0] += a0 * b0;
                sum[0][1] += a0 * b1;
                sum[1][0] += a1 * b0;
                sum[1][1] += a1 * b1;
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    if (row < M && col < N) {
        C[row * N + col] = sum[0][0];
        if (col + 1 < N) C[row * N + col + 1] = sum[0][1];
        if (row + 1 < M) C[(row + 1) * N + col] = sum[1][0];
        if (row + 1 < M && col + 1 < N) C[(row + 1) * N + col + 1] = sum[1][1];
    }
}
```

### Performance Comparison

```cpp
// Benchmark different implementations
void benchmark() {
    const int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    
    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Benchmark naive implementation
    dim3 threads(16, 16);
    dim3 grid((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    
    cudaEventRecord(start);
    matrixMul_naive<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive implementation: %f ms\n", milliseconds);
    
    // Benchmark tiled implementation
    cudaEventRecord(start);
    matrixMul_tiled<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Tiled implementation: %f ms\n", milliseconds);
    
    // Benchmark optimized implementation
    cudaEventRecord(start);
    matrixMul_optimized<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Optimized implementation: %f ms\n", milliseconds);
    
    // Compare with cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuBLAS implementation: %f ms\n", milliseconds);
    
    cublasDestroy(handle);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

## Conclusion

Optimizing GPU code is a complex but rewarding process. The key takeaways from this article include:

1. **Profile First**: Always profile your code to identify bottlenecks before optimizing
2. **Memory Access Patterns**: Ensure coalesced global memory access and avoid shared memory bank conflicts
3. **Occupancy**: Balance register usage, shared memory, and block size to maximize occupancy
4. **Shared Memory**: Use shared memory to reduce global memory accesses and enable collaboration between threads
5. **Advanced Techniques**: Consider streams, persistent threads, and instruction-level optimizations for further performance gains

Remember that optimization is often application-specific. What works well for one algorithm may not be optimal for another. Always measure performance before and after optimizations to ensure you're making meaningful improvements.

In our next article, we'll explore GPU algorithms and patterns, focusing on common parallel primitives and how to implement them efficiently on GPUs.

## Exercises for Practice

1. **Profiling Exercise**: Profile a simple CUDA or OpenCL application using vendor tools and identify performance bottlenecks.

2. **Memory Coalescing**: Implement and compare coalesced vs. non-coalesced memory access patterns for a simple vector operation.

3. **Shared Memory Optimization**: Optimize a 2D stencil computation using shared memory tiling.

4. **Occupancy Analysis**: Analyze how different block sizes and register usage affect occupancy and performance for a compute-bound kernel.

5. **Stream Processing**: Implement a pipeline using CUDA streams to overlap computation with data transfers.

## Further Resources

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA Performance Analysis Tools](https://developer.nvidia.com/performance-analysis-tools)
- [OpenCL Optimization Guide](https://www.khronos.org/registry/OpenCL/specs/opencl-1.1-extensions.pdf)
- [AMD OpenCL Programming Guide](https://developer.amd.com/wordpress/media/2013/12/AMD_OpenCL_Programming_User_Guide.pdf)
- [GPU Gems 3: Chapter 31 - Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)