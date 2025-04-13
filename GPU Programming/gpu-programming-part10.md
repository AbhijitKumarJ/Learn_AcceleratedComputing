# GPU Algorithms and Patterns

*Welcome to the tenth installment of our GPU programming series! In this article, we'll explore common GPU algorithms and parallel patterns that form the building blocks of efficient GPU applications. Understanding these patterns will help you design and implement high-performance parallel solutions for a wide range of problems.*

## Introduction to GPU Algorithms and Patterns

GPU programming requires a different approach to algorithm design compared to traditional CPU programming. While CPUs excel at sequential processing with complex control flow, GPUs thrive on data-parallel workloads where the same operation is applied to many data elements simultaneously. In this article, we'll explore fundamental parallel patterns that are particularly well-suited for GPU implementation.

## Parallel Reduction and Scan Operations

Reduction and scan operations are fundamental building blocks for many parallel algorithms. They transform arrays of data into summary values or derived arrays through associative operations like sum, min, max, or logical operations.

### Parallel Reduction

Reduction combines all elements in an array using an associative operation (like addition) to produce a single result. For example, summing all elements in an array.

#### Basic Parallel Reduction

```cuda
__global__ void reduce_sum(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    // Load shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

#### Optimized Reduction

The basic reduction has several inefficiencies. Here's an optimized version:

```cuda
__global__ void reduce_sum_optimized(float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    // Each thread loads multiple elements
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load first round of data
    float sum = 0;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction with sequential addressing
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no __syncthreads needed within a warp)
    if (tid < 32) {
        // Volatile prevents compiler optimization
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

#### Modern Reduction with Warp Shuffle

On newer GPUs, warp shuffle instructions allow threads within a warp to exchange values directly:

```cuda
__global__ void reduce_sum_shuffle(float* input, float* output, int n) {
    // Each thread loads one element
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % warpSize;
    unsigned int warpId = tid / warpSize;
    
    // Load data
    float sum = (i < n) ? input[i] : 0;
    
    // Warp-level reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes result to shared memory
    __shared__ float warpSums[32]; // Max 32 warps per block
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();
    
    // Final reduction (first warp only)
    if (warpId == 0) {
        sum = (lane < blockDim.x / warpSize) ? warpSums[lane] : 0;
        
        // Warp-level reduction again
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread writes result
        if (lane == 0) output[blockIdx.x] = sum;
    }
}
```

### Parallel Scan (Prefix Sum)

Scan operations compute all partial reductions of an array. For example, an exclusive prefix sum computes the sum of all previous elements for each position.

#### Naive Scan

```cuda
__global__ void scan_naive(float* input, float* output, int n) {
    // Simple but inefficient approach
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < i; j++) {
            sum += input[j];
        }
        output[i] = sum; // Exclusive scan
    }
}
```

#### Work-Efficient Scan (Hillis-Steele)

```cuda
__global__ void scan_hillis_steele(float* input, float* output, int n) {
    __shared__ float temp[2 * BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    
    // Load input into shared memory
    if (block_start + tid < n) {
        temp[tid] = input[block_start + tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Hillis-Steele scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * 2 - 1;
        if (index >= stride) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (block_start + tid < n) {
        output[block_start + tid] = (tid > 0) ? temp[tid - 1] : 0; // Exclusive scan
    }
}
```

#### Blelloch Scan (Work-Efficient)

```cuda
__global__ void scan_blelloch(float* input, float* output, int n) {
    __shared__ float temp[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    
    // Load input into shared memory
    if (block_start + tid < n) {
        temp[tid] = input[block_start + tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Reduction phase (up-sweep)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear the last element
    if (tid == 0) temp[blockDim.x - 1] = 0;
    __syncthreads();
    
    // Distribution phase (down-sweep)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            float temp_prev = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += temp_prev;
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (block_start + tid < n) {
        output[block_start + tid] = temp[tid];
    }
}
```

## Sorting on the GPU

Sorting is a fundamental operation in computing, and GPUs can accelerate sorting algorithms significantly through parallelization.

### Bitonic Sort

Bitonic sort is well-suited for GPU implementation due to its regular structure and parallelism:

```cuda
__global__ void bitonic_sort_step(float* values, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j; // Calculate bitonic partner
    
    // Only threads with ixj > i are active to avoid double swapping
    if (ixj > i) {
        float x_i = values[i];
        float x_ixj = values[ixj];
        
        // Sort in ascending or descending order based on bit k
        bool ascending = ((i & k) == 0);
        if ((ascending && x_i > x_ixj) || (!ascending && x_i < x_ixj)) {
            values[i] = x_ixj;
            values[ixj] = x_i;
        }
    }
}

// Host function to launch the kernel
void bitonic_sort(float* d_values, int n) {
    // Bitonic sort requires power-of-2 sized arrays
    int num_threads = 256;
    int num_blocks = (n + num_threads - 1) / num_threads;
    
    // Outer loop for bitonic stages
    for (int k = 2; k <= n; k *= 2) {
        // Inner loop for bitonic sub-stages
        for (int j = k/2; j > 0; j /= 2) {
            bitonic_sort_step<<<num_blocks, num_threads>>>(d_values, j, k);
        }
    }
}
```

### Radix Sort

Radix sort is another efficient sorting algorithm for GPUs, especially for integer keys:

```cuda
// Simplified radix sort kernel (for 32-bit integers)
__global__ void radix_sort_step(unsigned int* keys_in, unsigned int* keys_out, 
                               int n, int bit) {
    __shared__ unsigned int count[2]; // Count of 0s and 1s
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Initialize counts
    if (tid < 2) count[tid] = 0;
    __syncthreads();
    
    // Count bits
    int bit_value = 0;
    if (gid < n) {
        bit_value = (keys_in[gid] >> bit) & 1;
        atomicAdd(&count[bit_value], 1);
    }
    __syncthreads();
    
    // Compute scan for positioning
    if (tid == 0) {
        count[1] += count[0]; // Exclusive scan
    }
    __syncthreads();
    
    // Reorder elements
    if (gid < n) {
        int pos;
        if (bit_value == 0) {
            pos = atomicAdd(&count[0], 1);
        } else {
            pos = atomicAdd(&count[1], 1) - count[0];
        }
        keys_out[pos] = keys_in[gid];
    }
}
```

### Merge Sort

Merge sort can also be parallelized on GPUs:

```cuda
__global__ void merge_kernel(float* A, float* B, float* C, int n, int m) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n + m) return;
    
    // Binary search to find position
    int a_idx = min(tid, n);
    int b_idx = tid - a_idx;
    
    int a_start = max(0, tid - m);
    int a_end = min(n, tid);
    
    while (a_start < a_end) {
        int a_mid = (a_start + a_end) / 2;
        int b_mid = tid - a_mid - 1;
        
        if (b_mid >= 0 && b_mid < m && A[a_mid] > B[b_mid]) {
            a_end = a_mid;
        } else {
            a_start = a_mid + 1;
        }
    }
    
    a_idx = a_start;
    b_idx = tid - a_idx;
    
    // Determine the value to write
    if (b_idx >= m || (a_idx < n && A[a_idx] <= B[b_idx])) {
        C[tid] = A[a_idx];
    } else {
        C[tid] = B[b_idx];
    }
}
```

## Graph Algorithms

Graph algorithms are challenging to parallelize efficiently due to irregular memory access patterns, but GPUs can still provide significant speedups for many graph operations.

### Breadth-First Search (BFS)

```cuda
__global__ void bfs_kernel(int* adjacency_list, int* adjacency_offsets, 
                          int* frontier, int* new_frontier, int* visited, 
                          int* frontier_size, int* new_frontier_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= *frontier_size) return;
    
    int node = frontier[tid];
    int start = adjacency_offsets[node];
    int end = adjacency_offsets[node + 1];
    
    // Process all neighbors
    for (int edge = start; edge < end; edge++) {
        int neighbor = adjacency_list[edge];
        if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
            int idx = atomicAdd(new_frontier_size, 1);
            new_frontier[idx] = neighbor;
        }
    }
}
```

### Single-Source Shortest Path (SSSP)

Bellman-Ford algorithm for SSSP can be parallelized on GPUs:

```cuda
__global__ void bellman_ford_kernel(int* adjacency_list, int* adjacency_offsets, 
                                   int* weights, int* distances, bool* changed,
                                   int num_nodes) {
    int node = threadIdx.x + blockIdx.x * blockDim.x;
    if (node >= num_nodes) return;
    
    int start = adjacency_offsets[node];
    int end = adjacency_offsets[node + 1];
    
    for (int edge = start; edge < end; edge++) {
        int neighbor = adjacency_list[edge];
        int weight = weights[edge];
        int new_dist = distances[node] + weight;
        
        if (new_dist < distances[neighbor]) {
            atomicMin(&distances[neighbor], new_dist);
            *changed = true;
        }
    }
}
```

### Connected Components

Finding connected components using label propagation:

```cuda
__global__ void connected_components_kernel(int* adjacency_list, int* adjacency_offsets,
                                           int* labels, bool* changed, int num_nodes) {
    int node = threadIdx.x + blockIdx.x * blockDim.x;
    if (node >= num_nodes) return;
    
    int start = adjacency_offsets[node];
    int end = adjacency_offsets[node + 1];
    int my_label = labels[node];
    
    // Find minimum label among neighbors
    for (int edge = start; edge < end; edge++) {
        int neighbor = adjacency_list[edge];
        int neighbor_label = labels[neighbor];
        
        if (neighbor_label < my_label) {
            my_label = neighbor_label;
            *changed = true;
        }
    }
    
    // Update label if needed
    if (my_label < labels[node]) {
        labels[node] = my_label;
    }
}
```

## Matrix Operations and Stencil Computations

Matrix operations and stencil computations are naturally suited for GPU acceleration due to their regular structure and data parallelism.

### Matrix-Matrix Multiplication

```cuda
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Output element coordinates
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 2D Stencil Computation

```cuda
__global__ void stencil_2d(float* input, float* output, int width, int height) {
    // Shared memory for input tile plus halo
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + 1; // +1 for halo
    int ty = threadIdx.y + 1; // +1 for halo
    
    // Load main tile data
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    } else {
        tile[ty][tx] = 0.0f;
    }
    
    // Load halo regions
    if (threadIdx.x == 0 && x > 0) {
        tile[ty][0] = input[y * width + (x - 1)];
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1) {
        tile[ty][tx + 1] = input[y * width + (x + 1)];
    }
    if (threadIdx.y == 0 && y > 0) {
        tile[0][tx] = input[(y - 1) * width + x];
    }
    if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1) {
        tile[ty + 1][tx] = input[(y + 1) * width + x];
    }
    
    __syncthreads();
    
    // Apply stencil
    if (x < width && y < height) {
        float result = 0.2f * tile[ty][tx] +
                      0.2f * tile[ty-1][tx] +
                      0.2f * tile[ty+1][tx] +
                      0.2f * tile[ty][tx-1] +
                      0.2f * tile[ty][tx+1];
        output[y * width + x] = result;
    }
}
```

### Sparse Matrix-Vector Multiplication

```cuda
__global__ void spmv_csr(int* row_offsets, int* column_indices, float* values,
                        float* x, float* y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[column_indices[i]];
        }
        
        y[row] = dot;
    }
}
```

## Implementing Efficient GPU Algorithms

When implementing these algorithms, consider the following optimization strategies:

### Memory Access Patterns

- Ensure coalesced memory access for global memory operations
- Use shared memory for data reuse within thread blocks
- Consider using texture memory for read-only data with spatial locality

### Work Distribution

- Balance work across threads and blocks
- Avoid thread divergence within warps
- Consider dynamic work distribution for irregular workloads

### Algorithmic Considerations

- Choose algorithms that map well to GPU architecture
- Prefer regular computation patterns over irregular ones
- Consider hybrid CPU-GPU approaches for complex algorithms

## Case Study: Parallel Histogram

Histogram computation is a common operation that requires careful handling on GPUs due to potential atomic operation conflicts:

```cuda
__global__ void histogram_naive(unsigned char* data, int* histogram, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < n) {
        atomicAdd(&histogram[data[tid]], 1);
    }
}

__global__ void histogram_shared(unsigned char* data, int* histogram, int n) {
    __shared__ int temp_histogram[256];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    
    // Initialize shared memory
    if (local_tid < 256) {
        temp_histogram[local_tid] = 0;
    }
    __syncthreads();
    
    // Build local histogram
    if (tid < n) {
        atomicAdd(&temp_histogram[data[tid]], 1);
    }
    __syncthreads();
    
    // Merge with global histogram
    if (local_tid < 256) {
        atomicAdd(&histogram[local_tid], temp_histogram[local_tid]);
    }
}
```

## Conclusion

Efficient GPU algorithms and patterns are essential for harnessing the full power of parallel computing. In this article, we've explored fundamental parallel operations including reduction, scan, sorting, graph algorithms, and matrix computations. By understanding these patterns and their GPU implementations, you can develop high-performance parallel solutions for a wide range of computational problems.

In our next article, we'll explore multi-GPU programming, focusing on techniques for distributing work across multiple GPUs and managing inter-GPU communication.

## Exercises for Practice

1. **Reduction Implementation**: Implement and compare the performance of the three reduction algorithms (basic, optimized, and shuffle-based) for summing a large array of floating-point numbers.

2. **Parallel Scan**: Implement an exclusive prefix sum algorithm and test it on arrays of different sizes.

3. **Sorting Challenge**: Implement a parallel sorting algorithm of your choice and compare its performance against a CPU-based sort for various input sizes.

4. **Graph Algorithm**: Implement a parallel BFS algorithm for a large graph and analyze its performance characteristics.

5. **Stencil Computation**: Implement a 2D stencil computation (like a simple blur filter) using shared memory tiling and measure the performance improvement over a naive implementation.

## Further Resources

- [NVIDIA CUDA Documentation on Reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Efficient Parallel Scan Algorithms](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [GPU Gems 2: Chapter 46 - Improved GPU Sorting](https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting)
- [Efficient Sparse Matrix-Vector Multiplication on CUDA](https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf)
- [Graph Algorithms on GPUs](https://www.cs.utexas.edu/~pingali/CS377P/2020sp/lectures/GraphAlgorithmsOnGPUs.pdf)