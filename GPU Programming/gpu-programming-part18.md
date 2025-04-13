# Unified Memory and Heterogeneous Computing

*Welcome to the eighteenth installment of our GPU programming series! In this article, we'll explore unified memory and heterogeneous computing, focusing on CPU-GPU memory sharing models, heterogeneous task scheduling, system-wide coherency, and programming models for heterogeneous systems.*

## Introduction to Unified Memory

Traditionally, GPU programming required explicit memory management with manual data transfers between the CPU (host) and GPU (device). This approach, while offering fine-grained control, introduces complexity and potential for errors. Unified Memory represents a significant advancement by providing a single memory space accessible by both CPUs and GPUs.

Unified Memory creates a pool of managed memory that is shared between the CPU and GPU, automatically migrating data between host and device as needed. This simplifies programming and enables applications to use memory sizes exceeding GPU memory capacity.

## CPU-GPU Memory Sharing Models

Several memory sharing models have evolved to address the challenges of heterogeneous computing:

### 1. Explicit Memory Management

The traditional approach requires programmers to explicitly manage data transfers:

```cuda
// Traditional explicit memory management in CUDA
void explicit_memory_example(float* host_data, int size) {
    // Allocate device memory
    float* device_data;
    cudaMalloc(&device_data, size * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(device_data, host_data, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    vector_add<<<(size + 255) / 256, 256>>>(device_data, size);
    
    // Copy results back to host
    cudaMemcpy(host_data, device_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(device_data);
}
```

### 2. CUDA Unified Memory

CUDA's Unified Memory provides a single memory space accessible by both CPU and GPU:

```cuda
// CUDA Unified Memory example
void unified_memory_example(int size) {
    // Allocate unified memory accessible by CPU and GPU
    float* unified_data;
    cudaMallocManaged(&unified_data, size * sizeof(float));
    
    // Initialize data from CPU
    for (int i = 0; i < size; i++) {
        unified_data[i] = static_cast<float>(i);
    }
    
    // Launch kernel - no explicit transfers needed
    vector_add<<<(size + 255) / 256, 256>>>(unified_data, size);
    
    // Ensure GPU work is complete before CPU accesses data
    cudaDeviceSynchronize();
    
    // CPU can now access the results directly
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += unified_data[i];
    }
    
    // Free unified memory
    cudaFree(unified_data);
}
```

### 3. Zero-Copy Memory

Zero-copy memory allows the GPU to directly access host memory without copying:

```cuda
// Zero-copy memory example
void zero_copy_example(int size) {
    // Allocate page-locked host memory accessible by GPU
    float* zero_copy_data;
    cudaHostAlloc(&zero_copy_data, size * sizeof(float), cudaHostAllocMapped);
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        zero_copy_data[i] = static_cast<float>(i);
    }
    
    // Get device pointer to host memory
    float* device_ptr;
    cudaHostGetDevicePointer(&device_ptr, zero_copy_data, 0);
    
    // Launch kernel using device pointer
    vector_add<<<(size + 255) / 256, 256>>>(device_ptr, size);
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Free host memory
    cudaFreeHost(zero_copy_data);
}
```

### 4. Heterogeneous Memory Management (HMM)

HMM is a Linux kernel feature that enables fine-grained page migration between CPU and GPU:

```cpp
// Simplified HMM concept (actual implementation is at OS level)
void hmm_concept_example(int size) {
    // Allocate HMM memory (conceptual)
    float* hmm_data = allocate_hmm_memory(size * sizeof(float));
    
    // CPU accesses memory - pages reside in CPU memory
    for (int i = 0; i < size; i++) {
        hmm_data[i] = static_cast<float>(i);
    }
    
    // GPU kernel accesses memory - pages migrate to GPU as needed
    launch_gpu_kernel(hmm_data, size);
    
    // CPU accesses results - pages migrate back to CPU as needed
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += hmm_data[i];
    }
    
    // Free HMM memory
    free_hmm_memory(hmm_data);
}
```

## Heterogeneous Task Scheduling

Efficient utilization of heterogeneous systems requires intelligent task scheduling across different compute resources.

### Task-Based Programming Models

Task-based models allow work to be distributed across CPUs and GPUs based on resource availability and suitability:

```cpp
// Conceptual task-based heterogeneous scheduling
void heterogeneous_task_example() {
    // Create a task graph
    TaskGraph graph;
    
    // Add CPU and GPU tasks with dependencies
    Task* preprocessing = graph.add_task(CPU_TASK, preprocess_function);
    Task* gpu_compute1 = graph.add_task(GPU_TASK, gpu_kernel1);
    Task* gpu_compute2 = graph.add_task(GPU_TASK, gpu_kernel2);
    Task* cpu_compute = graph.add_task(CPU_TASK, cpu_function);
    Task* postprocessing = graph.add_task(CPU_TASK, postprocess_function);
    
    // Define dependencies
    graph.add_dependency(preprocessing, gpu_compute1);
    graph.add_dependency(preprocessing, gpu_compute2);
    graph.add_dependency(gpu_compute1, postprocessing);
    graph.add_dependency(gpu_compute2, postprocessing);
    graph.add_dependency(cpu_compute, postprocessing);
    
    // Execute task graph with runtime scheduler
    graph.execute();
}
```

### Dynamic Work Distribution

Dynamic work distribution adapts to system conditions and workload characteristics:

```cpp
// Dynamic work distribution example
void dynamic_work_distribution(float* data, int size) {
    // Determine workload split based on system conditions
    float cpu_capability = measure_cpu_capability();
    float gpu_capability = measure_gpu_capability();
    
    float total_capability = cpu_capability + gpu_capability;
    int cpu_elements = static_cast<int>(size * (cpu_capability / total_capability));
    int gpu_elements = size - cpu_elements;
    
    // Launch CPU work asynchronously
    std::thread cpu_thread([&]() {
        process_on_cpu(data, cpu_elements);
    });
    
    // Launch GPU work
    process_on_gpu(data + cpu_elements, gpu_elements);
    
    // Wait for CPU work to complete
    cpu_thread.join();
}
```

## System-Wide Coherency

Maintaining coherent memory views across heterogeneous components is crucial for correctness.

### Memory Coherence Protocols

Modern heterogeneous systems implement various coherence protocols:

1. **Directory-based coherence**: Tracks memory ownership and state across components
2. **Snooping protocols**: Components monitor memory transactions to maintain coherence
3. **Hybrid approaches**: Combine directory and snooping for different memory regions

### Coherence Domains

Coherence domains define regions where memory consistency is maintained:

```cpp
// Conceptual coherence domain example
void coherence_domain_example() {
    // Create coherence domains
    CoherenceDomain system_coherent;
    CoherenceDomain device_coherent;
    
    // Allocate memory in system coherent domain (visible to all devices)
    void* system_memory = system_coherent.allocate(SIZE);
    
    // Allocate memory in device coherent domain (visible to specific devices)
    void* device_memory = device_coherent.allocate(SIZE);
    
    // System coherent operations
    cpu_function(system_memory);
    gpu_function(system_memory);  // Automatically coherent
    
    // Device coherent operations require explicit synchronization
    cpu_function(device_memory);
    explicit_synchronize();  // Ensure coherence
    gpu_function(device_memory);
}
```

### Synchronization Primitives

Synchronization primitives ensure proper ordering of memory operations:

```cuda
// CUDA memory synchronization example
__global__ void kernel_with_sync(int* data, int* flag) {
    // Perform computation
    data[threadIdx.x] = compute_value();
    
    // Ensure all threads complete computation
    __syncthreads();
    
    // Thread 0 signals completion
    if (threadIdx.x == 0) {
        // Ensure all memory writes are visible
        __threadfence_system();
        *flag = 1;  // Signal completion
    }
}

void host_function() {
    int* d_data;
    int* d_flag;
    cudaMalloc(&d_data, SIZE);
    cudaMallocManaged(&d_flag, sizeof(int));
    *d_flag = 0;
    
    // Launch kernel
    kernel_with_sync<<<1, 256>>>(d_data, d_flag);
    
    // Wait for kernel completion signal
    while (*d_flag == 0) {
        // CPU busy-wait with memory fence
        std::atomic_thread_fence(std::memory_order_acquire);
    }
    
    // Now safe to access d_data
    process_results(d_data);
}
```

## Programming Models for Heterogeneous Systems

Several programming models have emerged to simplify heterogeneous computing.

### CUDA Graphs

CUDA Graphs allow defining and optimizing complex task graphs:

```cuda
// CUDA Graphs example
void cuda_graphs_example(float* input, float* output, int size) {
    // Create stream for graph capture
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Allocate device memory
    float *d_input, *d_intermediate, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_intermediate, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    // Begin graph capture
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    // Record operations in graph
    cudaMemcpyAsync(d_input, input, size * sizeof(float), 
                   cudaMemcpyHostToDevice, stream);
    
    kernel1<<<blocks, threads, 0, stream>>>(d_input, d_intermediate, size);
    kernel2<<<blocks, threads, 0, stream>>>(d_intermediate, d_output, size);
    
    cudaMemcpyAsync(output, d_output, size * sizeof(float), 
                   cudaMemcpyDeviceToHost, stream);
    
    // End capture
    cudaStreamEndCapture(stream, &graph);
    
    // Create executable graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    
    // Execute graph multiple times
    for (int i = 0; i < ITERATIONS; i++) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }
    
    // Clean up
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_input);
    cudaFree(d_intermediate);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}
```

### OpenCL 2.0+ SVM (Shared Virtual Memory)

OpenCL's SVM provides different levels of memory sharing:

```cpp
// OpenCL SVM example
void opencl_svm_example() {
    // Set up OpenCL context, queue, etc.
    cl_context context = create_context();
    cl_command_queue queue = create_command_queue(context);
    cl_program program = create_program(context);
    cl_kernel kernel = create_kernel(program, "svm_kernel");
    
    // Allocate fine-grained SVM buffer
    cl_svm_mem_flags flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
    float* svm_ptr = (float*)clSVMAlloc(context, flags, SIZE, 0);
    
    // Initialize data directly through CPU
    for (int i = 0; i < SIZE/sizeof(float); i++) {
        svm_ptr[i] = static_cast<float>(i);
    }
    
    // Set kernel argument using SVM pointer
    clSetKernelArgSVMPointer(kernel, 0, svm_ptr);
    
    // Execute kernel
    size_t global_size = SIZE/sizeof(float);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    
    // No need for explicit transfers - CPU can access data after synchronization
    clFinish(queue);
    
    // Process results directly
    float sum = 0.0f;
    for (int i = 0; i < SIZE/sizeof(float); i++) {
        sum += svm_ptr[i];
    }
    
    // Free SVM memory
    clSVMFree(context, svm_ptr);
}
```

### SYCL

SYCL provides a single-source heterogeneous programming model based on C++:

```cpp
// SYCL example
#include <CL/sycl.hpp>

void sycl_example() {
    constexpr size_t size = 1024;
    std::vector<float> input(size, 1.0f);
    std::vector<float> output(size, 0.0f);
    
    // Create SYCL queue
    sycl::queue queue(sycl::default_selector{});
    
    // Create buffers
    sycl::buffer<float, 1> input_buffer(input.data(), sycl::range<1>(size));
    sycl::buffer<float, 1> output_buffer(output.data(), sycl::range<1>(size));
    
    // Submit work to queue
    queue.submit([&](sycl::handler& cgh) {
        // Accessors define data access mode
        auto input_accessor = input_buffer.get_access<sycl::access::mode::read>(cgh);
        auto output_accessor = output_buffer.get_access<sycl::access::mode::write>(cgh);
        
        // Define kernel
        cgh.parallel_for<class vector_add>(sycl::range<1>(size), 
            [=](sycl::id<1> idx) {
                output_accessor[idx] = input_accessor[idx] * 2.0f;
            });
    });
    
    // Buffer destructors ensure data is synchronized back to host
}
```

### Heterogeneous-Compute Interface for Portability (HIP)

HIP allows code to run on both NVIDIA and AMD GPUs:

```cpp
// HIP example
#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void hip_example(int size) {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(size * sizeof(float));
    h_b = (float*)malloc(size * sizeof(float));
    h_c = (float*)malloc(size * sizeof(float));
    
    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory
    hipMalloc(&d_a, size * sizeof(float));
    hipMalloc(&d_b, size * sizeof(float));
    hipMalloc(&d_c, size * sizeof(float));
    
    // Copy data to device
    hipMemcpy(d_a, h_a, size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size * sizeof(float), hipMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock), 
                      0, 0, d_a, d_b, d_c, size);
    
    // Copy result back to host
    hipMemcpy(h_c, d_c, size * sizeof(float), hipMemcpyDeviceToHost);
    
    // Clean up
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}
```

## Practical Example: Heterogeneous Image Processing Pipeline

Let's implement a practical example of a heterogeneous image processing pipeline that leverages both CPU and GPU:

```cpp
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <chrono>

// Image structure
struct Image {
    int width;
    int height;
    unsigned char* data;  // RGB data
};

// CPU function for image loading and preprocessing
Image load_and_preprocess_image(const char* filename) {
    // Load image from file (simplified)
    Image img = load_image(filename);
    
    // Perform CPU-efficient preprocessing
    for (int i = 0; i < img.width * img.height * 3; i += 3) {
        // Convert to grayscale (simplified)
        unsigned char gray = (img.data[i] + img.data[i+1] + img.data[i+2]) / 3;
        img.data[i] = img.data[i+1] = img.data[i+2] = gray;
    }
    
    return img;
}

// GPU kernel for image filtering
__global__ void apply_filter_kernel(unsigned char* input, unsigned char* output, 
                                  int width, int height, float* filter, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r = 0.0f, g = 0.0f, b = 0.0f;
        
        // Apply convolution filter
        int half_filter = filter_size / 2;
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                int image_x = min(max(x + fx, 0), width - 1);
                int image_y = min(max(y + fy, 0), height - 1);
                int filter_idx = (fy + half_filter) * filter_size + (fx + half_filter);
                int image_idx = (image_y * width + image_x) * 3;
                
                float filter_val = filter[filter_idx];
                r += input[image_idx] * filter_val;
                g += input[image_idx + 1] * filter_val;
                b += input[image_idx + 2] * filter_val;
            }
        }
        
        output[idx] = static_cast<unsigned char>(min(max(r, 0.0f), 255.0f));
        output[idx + 1] = static_cast<unsigned char>(min(max(g, 0.0f), 255.0f));
        output[idx + 2] = static_cast<unsigned char>(min(max(b, 0.0f), 255.0f));
    }
}

// CPU function for post-processing and saving
void postprocess_and_save(Image img, const char* filename) {
    // Enhance contrast (simplified)
    for (int i = 0; i < img.width * img.height * 3; i++) {
        img.data[i] = static_cast<unsigned char>(min(img.data[i] * 1.2, 255.0));
    }
    
    // Save image (simplified)
    save_image(img, filename);
}

// Main heterogeneous pipeline
void process_image_pipeline(const char* input_file, const char* output_file) {
    // Define a 5x5 Gaussian blur filter
    const int filter_size = 5;
    float h_filter[filter_size * filter_size] = {
        1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f,
        4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
        6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f,
        4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f,
        1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f
    };
    
    // Step 1: CPU - Load and preprocess image
    Image input_image = load_and_preprocess_image(input_file);
    int width = input_image.width;
    int height = input_image.height;
    
    // Step 2: Prepare for GPU processing
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_filter;
    
    // Allocate device memory
    cudaMalloc(&d_input, width * height * 3);
    cudaMalloc(&d_output, width * height * 3);
    cudaMalloc(&d_filter, filter_size * filter_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, input_image.data, width * height * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Step 3: GPU - Apply filter
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                  (height + block_size.y - 1) / block_size.y);
    
    apply_filter_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height, d_filter, filter_size);
    
    // Allocate memory for output image
    Image output_image;
    output_image.width = width;
    output_image.height = height;
    output_image.data = new unsigned char[width * height * 3];
    
    // Copy result back to host
    cudaMemcpy(output_image.data, d_output, width * height * 3, cudaMemcpyDeviceToHost);
    
    // Step 4: CPU - Post-process and save
    postprocess_and_save(output_image, output_file);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    delete[] input_image.data;
    delete[] output_image.data;
}
```

## Conclusion

Unified memory and heterogeneous computing represent a significant evolution in GPU programming, simplifying development while enabling more complex applications. By providing seamless memory sharing between CPUs and GPUs, these technologies allow developers to focus on algorithms rather than memory management details.

As heterogeneous systems continue to evolve, programming models are becoming more sophisticated, offering higher-level abstractions that automatically optimize task scheduling and data movement. This trend will accelerate as new hardware architectures emerge with tighter integration between different compute resources.

In the next article, we'll explore GPU computing in the cloud, discussing major cloud GPU offerings, remote development workflows, cost optimization strategies, and container-based GPU applications.

## Further Resources

1. [NVIDIA Unified Memory Programming](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
2. [Heterogeneous Computing with OpenCL](https://www.sciencedirect.com/book/9780124058941/heterogeneous-computing-with-opencl)
3. [SYCL Programming Model](https://www.khronos.org/sycl/)
4. [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
5. [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)