# Getting Started with CUDA Programming

*Welcome to the fourth installment of our GPU programming series! After exploring the theoretical foundations of parallel computing and GPU architecture, we're now ready to write actual GPU code. In this article, we'll set up a CUDA development environment and create our first CUDA programs.*

## Setting up the CUDA Development Environment

### System Requirements

Before diving into CUDA programming, ensure your system meets these requirements:

1. **NVIDIA GPU**: You need an NVIDIA GPU that supports CUDA. Most NVIDIA GPUs released in the last decade support CUDA, but with varying capabilities (different "Compute Capability" versions).

2. **Operating System**:
   - Windows 7/10/11
   - Linux (Ubuntu, CentOS, RHEL recommended)
   - macOS (support limited to older versions due to Apple's transition away from NVIDIA)

3. **Disk Space**: At least 4GB for the CUDA Toolkit and additional space for development tools.

4. **C/C++ Knowledge**: CUDA is an extension of C/C++, so familiarity with these languages is essential.

### Installing the CUDA Toolkit

The CUDA Toolkit includes the compiler, libraries, debugging tools, and runtime components needed for CUDA development.

#### Windows Installation

1. Download the CUDA Toolkit installer from [NVIDIA's CUDA Downloads page](https://developer.nvidia.com/cuda-downloads).
2. Run the installer and follow the prompts.
3. Choose either a custom or express installation (express is recommended for beginners).
4. After installation, verify by opening a command prompt and typing:
   ```
   nvcc --version
   ```

#### Linux Installation

For Ubuntu:

```bash
# Add NVIDIA package repository
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Update package list
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda

# Set up environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

For other Linux distributions, refer to NVIDIA's installation guide for specific instructions.

#### macOS Installation

For macOS (if using a compatible system):

1. Download the CUDA Toolkit installer for macOS from NVIDIA's website.
2. Open the .dmg file and follow the installation instructions.
3. Add the following to your .bash_profile or .zshrc:
   ```bash
   export PATH=/Developer/NVIDIA/CUDA-xx.x/bin:$PATH
   export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-xx.x/lib:$DYLD_LIBRARY_PATH
   ```
   (Replace xx.x with your CUDA version)

### Setting Up an Integrated Development Environment (IDE)

While you can use any text editor with the CUDA command-line tools, an IDE can significantly improve productivity.

#### Visual Studio (Windows)

1. Install Visual Studio (Community edition is free).
2. During CUDA Toolkit installation, select the option to integrate with Visual Studio.
3. In Visual Studio, you can create CUDA projects via: File → New → Project → NVIDIA → CUDA.

#### Visual Studio Code (Cross-platform)

1. Install Visual Studio Code.
2. Add the "C/C++" and "CUDA C++" extensions.
3. Configure your workspace settings for CUDA compilation.

#### Eclipse with Nsight (Cross-platform)

1. Install Eclipse IDE for C/C++ Developers.
2. Install the NVIDIA Nsight Eclipse Plugins via the Eclipse Marketplace.

### Verifying Your Installation

Create a simple CUDA program to verify your installation:

1. Create a file named `cuda_check.cu` with the following content:

```cuda
#include <stdio.h>

__global__ void checkGPU() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Launch kernel with 5 threads
    checkGPU<<<1, 5>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}
```

2. Compile and run the program:

```bash
nvcc cuda_check.cu -o cuda_check
./cuda_check  # or cuda_check.exe on Windows
```

If everything is set up correctly, you should see information about your GPU and messages from five GPU threads.

## CUDA Programming Model

Now that we have our environment set up, let's understand the fundamental concepts of the CUDA programming model.

### Kernels: GPU Functions

In CUDA, functions that run on the GPU are called **kernels**. Kernels are defined using the `__global__` specifier and are launched from CPU code using a special syntax:

```cuda
// Kernel definition
__global__ void myKernel(int *data) {
    // GPU code here
}

// Kernel launch from CPU code
myKernel<<<numBlocks, threadsPerBlock>>>(data);
```

The `<<<...>>>` syntax is unique to CUDA and specifies the execution configuration - how many thread blocks and how many threads per block should be used to execute the kernel.

### Threads, Blocks, and Grids

CUDA organizes parallel execution in a hierarchical structure:

1. **Thread**: The basic unit of execution. Each thread has a unique ID accessible via `threadIdx`.

2. **Block**: A group of threads that can cooperate via shared memory and synchronization. Blocks have a unique ID accessible via `blockIdx`.

3. **Grid**: A collection of blocks executing the same kernel. The grid dimensions are specified at kernel launch.

This hierarchy maps directly to the GPU hardware we discussed in the previous article:
- Threads within a block execute on the same Streaming Multiprocessor (SM)
- Different blocks may execute on different SMs, enabling scalability

![CUDA Thread Hierarchy](https://via.placeholder.com/800x400?text=CUDA+Thread+Hierarchy)

### Thread Indexing

CUDA provides built-in variables to identify threads:

- `threadIdx`: 3D vector (x, y, z) identifying the thread within its block
- `blockIdx`: 3D vector identifying the block within the grid
- `blockDim`: 3D vector specifying the dimensions of each block
- `gridDim`: 3D vector specifying the dimensions of the grid

For a 1D array of data, a common pattern to calculate the global thread index is:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D data (like images), we might use:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;
```

### Memory Types in CUDA

CUDA provides access to various memory types, each with different scope, lifetime, and performance characteristics:

1. **Global Memory**: Accessible by all threads, high capacity but high latency
2. **Shared Memory**: Shared between threads in a block, low latency but limited size
3. **Local Memory**: Private to each thread, spills to global memory
4. **Registers**: Fastest memory, private to each thread, very limited quantity
5. **Constant Memory**: Read-only memory accessible by all threads, cached
6. **Texture Memory**: Optimized for 2D spatial locality, cached

Choosing the right memory type for your data is crucial for performance.

## Writing Your First CUDA Program

Let's create a simple CUDA program that adds two vectors. This is the "Hello, World" of parallel computing.

### Vector Addition Example

```cuda
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Vector size
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Threads per block (typically a multiple of 32)
    int threadsPerBlock = 256;
    
    // Blocks per grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < n; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            break;
        }
    }
    
    printf("Vector addition completed successfully!\n");
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

Let's break down this example:

1. We define a kernel `vectorAdd` that adds corresponding elements from arrays `a` and `b` and stores the result in `c`.

2. In `main()`, we:
   - Allocate memory on both the host (CPU) and device (GPU)
   - Initialize the host arrays with random values
   - Copy the input data from host to device
   - Calculate the execution configuration (blocks and threads)
   - Launch the kernel
   - Copy the result back from device to host
   - Verify the result
   - Free allocated memory

3. The execution configuration is calculated to ensure we have enough threads to process all elements:
   - We use 256 threads per block (typically a multiple of 32 for efficiency)
   - We calculate the number of blocks needed to cover all elements

### Compiling and Running the Example

Save the code to a file named `vector_add.cu` and compile it with:

```bash
nvcc vector_add.cu -o vector_add
```

Then run the executable:

```bash
./vector_add  # or vector_add.exe on Windows
```

You should see the message "Vector addition completed successfully!" if everything works correctly.

## Understanding CUDA Memory Management

Memory management is one of the most critical aspects of CUDA programming. Let's explore the basic operations:

### Memory Allocation

- **Host memory** is allocated using standard C functions like `malloc()` or C++ operators like `new`.
- **Device memory** is allocated using `cudaMalloc()`:
  ```cuda
  float *d_array;
  cudaMalloc(&d_array, size_in_bytes);
  ```

### Memory Transfer

Data must be explicitly transferred between host and device using `cudaMemcpy()`:

```cuda
// Host to device transfer
cudaMemcpy(d_array, h_array, size_in_bytes, cudaMemcpyHostToDevice);

// Device to host transfer
cudaMemcpy(h_array, d_array, size_in_bytes, cudaMemcpyDeviceToHost);
```

### Memory Deallocation

- **Host memory** is freed using `free()` or `delete`.
- **Device memory** is freed using `cudaFree()`:
  ```cuda
  cudaFree(d_array);
  ```

### Unified Memory (Modern CUDA)

Modern CUDA provides a simplified memory model called Unified Memory, which automatically migrates data between host and device:

```cuda
// Allocate unified memory
float *unified_array;
cudaMallocManaged(&unified_array, size_in_bytes);

// Access from host or device without explicit transfers
// ...

// Free unified memory
cudaFree(unified_array);
```

Unified Memory simplifies code but may not always provide the best performance for all applications.

## Error Handling in CUDA

CUDA functions return error codes that should be checked for robust applications. A common pattern is to create an error-checking macro:

```cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_array, bytes));
CUDA_CHECK(cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice));
```

For kernel launches, which are asynchronous, use:

```cuda
kernel<<<blocks, threads>>>(args);
CUDA_CHECK(cudaGetLastError()); // Check for launch errors
CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish and check for errors
```

## A More Complex Example: Matrix Multiplication

Let's implement a slightly more complex example: matrix multiplication. This will demonstrate 2D indexing and the use of shared memory for optimization.

```cuda
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

// Naive matrix multiplication kernel
__global__ void matrixMulNaive(float *A, float *B, float *C, int width) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Optimized matrix multiplication using shared memory
__global__ void matrixMulShared(float *A, float *B, float *C, int width) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Shared memory for the sub-matrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Loop over all sub-matrices needed to compute the block
    for (int m = 0; m < (width + BLOCK_SIZE - 1) / BLOCK_SIZE; m++) {
        // Load sub-matrices into shared memory
        int aRow = blockRow * BLOCK_SIZE + row;
        int aCol = m * BLOCK_SIZE + col;
        int bRow = m * BLOCK_SIZE + row;
        int bCol = blockCol * BLOCK_SIZE + col;
        
        // Check bounds
        if (aRow < width && aCol < width) {
            As[row][col] = A[aRow * width + aCol];
        } else {
            As[row][col] = 0.0f;
        }
        
        if (bRow < width && bCol < width) {
            Bs[row][col] = B[bRow * width + bCol];
        } else {
            Bs[row][col] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Multiply sub-matrices
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[row][k] * Bs[k][col];
        }
        
        // Synchronize to ensure computation is done before loading new sub-matrices
        __syncthreads();
    }
    
    // Write result
    int cRow = blockRow * BLOCK_SIZE + row;
    int cCol = blockCol * BLOCK_SIZE + col;
    if (cRow < width && cCol < width) {
        C[cRow * width + cCol] = sum;
    }
}

int main() {
    int width = 1024; // Matrix dimensions (width x width)
    size_t bytes = width * width * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes); // For verification
    
    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                       (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // Launch naive kernel and measure time
    cudaEventRecord(start);
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive kernel execution time: %.2f ms\n", milliseconds);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Compute reference result on CPU for verification
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += h_A[i * width + k] * h_B[k * width + j];
            }
            h_C_ref[i * width + j] = sum;
        }
    }
    
    // Verify naive kernel result
    bool correct = true;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Naive kernel result verified!\n");
    }
    
    // Launch optimized kernel and measure time
    cudaEventRecord(start);
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Shared memory kernel execution time: %.2f ms\n", milliseconds);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify optimized kernel result
    correct = true;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("Shared memory kernel result verified!\n");
    }
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
```

This example demonstrates:

1. **2D thread indexing** for matrix operations
2. **Shared memory usage** to reduce global memory accesses
3. **Performance measurement** using CUDA events
4. **Result verification** by comparing with CPU computation

The shared memory version should show a significant performance improvement over the naive version, especially for larger matrices.

## Compiling and Running CUDA Applications

### Basic Compilation

The NVIDIA CUDA Compiler (NVCC) is used to compile CUDA code:

```bash
nvcc my_program.cu -o my_program
```

### Common Compilation Flags

- **Architecture specification**: `-arch=sm_XX` (where XX is the compute capability)
  ```bash
  nvcc -arch=sm_75 my_program.cu -o my_program  # For Turing GPUs
  ```

- **Optimization level**: `-O0` (no optimization) to `-O3` (maximum optimization)
  ```bash
  nvcc -O3 my_program.cu -o my_program
  ```

- **Debugging**: `-g -G` enables host and device debugging
  ```bash
  nvcc -g -G my_program.cu -o my_program
  ```

- **Generating PTX code**: `-ptx` generates human-readable intermediate code
  ```bash
  nvcc -ptx my_program.cu
  ```

### Linking with Libraries

CUDA comes with powerful libraries that you can link against:

```bash
# Link with cuBLAS (CUDA Basic Linear Algebra Subroutines)
nvcc my_program.cu -o my_program -lcublas

# Link with cuFFT (CUDA Fast Fourier Transform)
nvcc my_program.cu -o my_program -lcufft
```

## Conclusion and Next Steps

In this article, we've set up a CUDA development environment and written our first CUDA programs. We've covered the basic concepts of the CUDA programming model, including kernels, thread hierarchy, and memory management.

Here's what you should now be able to do:

1. Set up a CUDA development environment
2. Understand the CUDA programming model
3. Write basic CUDA kernels
4. Manage memory transfers between CPU and GPU
5. Compile and run CUDA applications

In the next article, we'll dive deeper into CUDA memory management, exploring advanced techniques for optimizing memory access patterns and utilizing different memory types effectively.

### Exercises for Practice

To reinforce your learning, try these exercises:

1. Modify the vector addition example to perform vector subtraction or multiplication.
2. Implement a CUDA kernel that computes the element-wise square root of a vector.
3. Write a kernel that transposes a matrix (swaps rows and columns).
4. Experiment with different block sizes in the matrix multiplication example and observe the performance impact.
5. Implement a parallel reduction to find the sum or maximum value in an array.

---

*Ready to optimize your GPU code? Join us for the next article in our series: "CUDA Memory Management" where we'll explore advanced memory techniques for maximum performance.*