# Lesson 4: Introduction to CUDA Programming

Welcome to the fourth lesson in our "Accelerating the Future" series. In this lesson, we'll dive into CUDA programming, which is the gateway to unleashing the power of NVIDIA GPUs for general-purpose computing.

## What is CUDA and why it revolutionized GPU computing

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. Introduced in 2006, CUDA represented a revolutionary shift in how we use GPUs.

Before CUDA, programming GPUs for non-graphics tasks was extremely challenging. Developers had to disguise their computational problems as graphics problems, using graphics APIs in unintended ways. This approach, known as GPGPU (General-Purpose computing on Graphics Processing Units), was cumbersome and limited.

CUDA changed everything by providing:
- A direct way to access the GPU's computational resources
- A programming model that extends popular languages like C/C++
- Tools and libraries specifically designed for general-purpose computing

This innovation opened up GPU acceleration to fields beyond graphics, including scientific research, finance, artificial intelligence, and data analysis.

## The CUDA programming model explained simply

The CUDA programming model is based on a few key concepts:

1. **Heterogeneous Programming**: Your application runs on two different processors - the CPU (host) and the GPU (device).

2. **Parallel Execution**: Instead of running a single thread of execution, CUDA programs launch thousands or millions of threads that execute the same function on different data.

3. **Kernel Functions**: These are special functions that execute on the GPU in parallel across many threads.

4. **Memory Hierarchy**: CUDA provides different types of memory with various performance characteristics and visibility scopes.

Think of CUDA as a way to identify parts of your program that can be executed in parallel, package them into kernel functions, and then launch these functions on the GPU with many threads.

## Understanding the host (CPU) and device (GPU) relationship

In CUDA programming, the CPU and GPU work together but have distinct roles:

**The CPU (Host):**
- Controls the overall program flow
- Allocates memory on both the CPU and GPU
- Transfers data between CPU and GPU memory
- Launches kernel functions on the GPU
- Processes results after GPU computation

**The GPU (Device):**
- Executes kernel functions with massive parallelism
- Performs the computationally intensive work
- Has its own memory space separate from the CPU
- Excels at data-parallel tasks where the same operation is applied to many data elements

This relationship creates a programming pattern where you:
1. Prepare data on the CPU
2. Transfer data to the GPU
3. Process data in parallel on the GPU
4. Transfer results back to the CPU
5. Continue with CPU processing or display results

## Your first CUDA program: Hello World example with explanation

Let's look at a simple CUDA program that prints "Hello, World from GPU!":

```cpp
#include <stdio.h>

// This is a kernel function that will run on the GPU
__global__ void helloFromGPU() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Print from the CPU
    printf("Hello, World from CPU!\n");
    
    // Launch the kernel with 1 block of 1 thread
    helloFromGPU<<<1, 1>>>();
    
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();
    
    return 0;
}
```

Let's break this down:

- `__global__` is a CUDA keyword that declares a function as a kernel that runs on the GPU but can be called from the CPU.
- `helloFromGPU<<<1, 1>>>()` launches the kernel with a specific configuration:
  - The first number (1) specifies one block
  - The second number (1) specifies one thread per block
- `cudaDeviceSynchronize()` makes the CPU wait until the GPU has completed its work.

When you compile and run this program, you'll see both messages printed, demonstrating that code is executing on both the CPU and GPU.

## Basic memory management: Host to device transfers

One of the most important aspects of CUDA programming is managing memory transfers between the CPU and GPU. Here's a simple example that adds two arrays:

```cpp
#include <stdio.h>

// Kernel function to add two arrays
__global__ void addArrays(int *a, int *b, int *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int size = 10;
    size_t bytes = size * sizeof(int);
    
    // Host arrays (CPU memory)
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    
    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Device arrays (GPU memory)
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 block of 256 threads
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addArrays<<<numBlocks, blockSize>>>(d_a, d_b, d_c, size);
    
    // Copy results back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < size; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

Key memory operations:
- `cudaMalloc()`: Allocates memory on the GPU
- `cudaMemcpy()`: Transfers data between CPU and GPU memory
- `cudaFree()`: Releases GPU memory

The memory transfer pattern is:
1. Allocate memory on both CPU and GPU
2. Initialize data on the CPU
3. Copy data from CPU to GPU
4. Perform computation on the GPU
5. Copy results from GPU back to CPU
6. Free memory on both CPU and GPU

## Thinking in parallel: How to structure problems for GPU computation

To effectively use GPUs, you need to think differently about your problems. Here are some guidelines:

1. **Identify data parallelism**: Look for operations that can be performed independently on many data elements.

2. **Minimize data dependencies**: Operations that depend on results from other operations are harder to parallelize.

3. **Focus on throughput over latency**: GPUs excel when processing large amounts of data, even if individual operations might take longer.

4. **Batch similar operations**: Group similar tasks together rather than switching between different types of operations.

5. **Consider memory access patterns**: GPUs perform best when threads access memory in specific patterns (coalesced access).

Example problems well-suited for GPU acceleration:
- Matrix operations
- Image processing
- Simulations with local interactions
- Machine learning training and inference
- Financial modeling with independent calculations

## CUDA threads, blocks, and grids visualized

CUDA organizes parallel execution in a hierarchical structure:

```
Grid
├── Block (0,0)
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   ├── Thread (1,0)
│   └── Thread (1,1)
├── Block (0,1)
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   ├── Thread (1,0)
│   └── Thread (1,1)
└── Block (1,0)
    ├── Thread (0,0)
    ├── Thread (0,1)
    ├── Thread (1,0)
    └── Thread (1,1)
```

- **Thread**: The smallest execution unit that runs a single instance of the kernel.
- **Block**: A group of threads that can cooperate and share a fast memory space called shared memory.
- **Grid**: A collection of blocks that execute the same kernel function.

When you launch a kernel with `myKernel<<<numBlocks, threadsPerBlock>>>()`, you're configuring this hierarchy.

Threads within a block can:
- Synchronize with each other using `__syncthreads()`
- Share data through shared memory
- Cooperate on subtasks

Blocks are scheduled independently, allowing the GPU to scale across different hardware capabilities.

## Common beginner mistakes and how to avoid them

1. **Not checking for errors**
   - Always check return values from CUDA API calls
   - Use `cudaGetLastError()` after kernel launches
   - Enable error checking in development builds

2. **Ignoring memory transfer costs**
   - Memory transfers between CPU and GPU are slow
   - Try to minimize transfers by keeping data on the GPU
   - Consider using pinned memory for faster transfers

3. **Launching too few threads**
   - GPUs need thousands of threads to achieve good performance
   - Aim for at least enough threads to keep all CUDA cores busy

4. **Ignoring warp divergence**
   - Threads execute in groups of 32 (warps)
   - If threads in a warp take different code paths, performance suffers
   - Try to ensure threads in the same warp follow the same execution path

5. **Not considering memory coalescing**
   - When threads access adjacent memory locations, performance improves
   - Uncoalesced memory access can severely limit performance

6. **Using too much shared memory or registers**
   - Limited resources per block can reduce occupancy
   - Monitor resource usage with profiling tools

7. **Synchronizing too frequently**
   - Thread synchronization has overhead
   - Minimize calls to `__syncthreads()`

## Key terminology definitions

- **Kernel**: A function that runs on the GPU and is executed by many threads in parallel
- **Host**: The CPU and its memory
- **Device**: The GPU and its memory
- **Thread**: The smallest execution unit in CUDA
- **Block**: A group of threads that can cooperate with each other
- **Grid**: A collection of blocks executing the same kernel
- **Warp**: A group of 32 threads that execute together in SIMT fashion
- **SIMT**: Single Instruction, Multiple Thread - the execution model of CUDA
- **Shared memory**: Fast memory shared between threads in the same block
- **Global memory**: The main GPU memory accessible by all threads
- **Coalescing**: When threads access memory in a pattern that can be combined into fewer transactions

## Try it yourself: A simple matrix addition exercise

Try implementing a CUDA kernel that adds two matrices. Here's a skeleton to get you started:

```cpp
#include <stdio.h>

// Matrix addition kernel
__global__ void matrixAdd(float *A, float *B, float *C, int width, int height) {
    // TODO: Calculate row and column index from threadIdx and blockIdx
    // TODO: Check if indices are within bounds
    // TODO: Perform the addition and store in C
}

int main() {
    // Matrix dimensions
    int width = 1024;
    int height = 1024;
    size_t bytes = width * height * sizeof(float);
    
    // Allocate host memory
    // TODO: Allocate and initialize matrices A, B, and C on the host
    
    // Allocate device memory
    // TODO: Allocate matrices on the device
    
    // Copy data to device
    // TODO: Copy A and B to the device
    
    // Launch kernel
    // TODO: Configure grid and block dimensions
    // TODO: Launch the matrixAdd kernel
    
    // Copy result back to host
    // TODO: Copy C from device to host
    
    // Verify results
    // TODO: Check that C = A + B
    
    // Free memory
    // TODO: Free host and device memory
    
    return 0;
}
```

## Further reading resources

**Beginner level:**
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA by Example: An Introduction to General-Purpose GPU Programming](https://developer.nvidia.com/cuda-example)

**Intermediate level:**
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Professional CUDA C Programming](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

**Advanced level:**
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Programming Massively Parallel Processors](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0124159923)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/tag/cuda/)

## Quick recap and preview of next lesson

In this lesson, we've covered:
- What CUDA is and why it revolutionized GPU computing
- The basic CUDA programming model
- The relationship between the CPU (host) and GPU (device)
- How to write a simple CUDA program
- Memory management and data transfers
- How to think about problems in a parallel way
- CUDA's thread hierarchy
- Common pitfalls and how to avoid them

In the next lesson, we'll explore AMD's approach to GPU computing with ROCm and HIP. We'll learn how AMD's solution compares to NVIDIA's CUDA, and how to write code that can run on both platforms.

---

*Remember: CUDA programming is a skill that develops with practice. Start with simple examples, use debugging tools, and gradually tackle more complex problems as you become comfortable with the parallel programming paradigm.*