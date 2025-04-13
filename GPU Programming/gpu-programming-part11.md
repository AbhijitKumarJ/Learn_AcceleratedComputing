# Multi-GPU Programming

*Welcome to the eleventh installment of our GPU programming series! In this article, we'll explore multi-GPU programming techniques that allow you to harness the power of multiple GPUs to solve larger problems and achieve even greater performance gains.*

## Introduction to Multi-GPU Computing

As computational demands continue to grow, a single GPU may not provide sufficient processing power for the most demanding applications. Multi-GPU systems offer a solution by distributing work across multiple graphics processors, potentially providing linear or near-linear speedups for well-designed applications.

In this article, we'll explore strategies for distributing work across multiple GPUs, communication techniques between GPUs, scaling considerations, and approaches for handling heterogeneous systems with different GPU models.

## Distributing Work Across Multiple GPUs

Effectively distributing work across multiple GPUs requires careful consideration of the problem structure, data dependencies, and communication patterns.

### Domain Decomposition

Domain decomposition involves dividing the problem space into subdomains, with each GPU responsible for processing a specific region.

#### Spatial Decomposition

For problems with spatial locality (like image processing, simulations, or matrix operations), spatial decomposition is often effective:

```cpp
// Example: Distributing image processing across multiple GPUs
void process_image_multi_gpu(unsigned char* h_image, int width, int height, int num_gpus) {
    // Calculate rows per GPU (assuming row-wise decomposition)
    int rows_per_gpu = (height + num_gpus - 1) / num_gpus;
    
    // Allocate host memory for results
    unsigned char* h_result = new unsigned char[width * height];
    
    // Allocate device memory and streams for each GPU
    unsigned char** d_images = new unsigned char*[num_gpus];
    unsigned char** d_results = new unsigned char*[num_gpus];
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    
    for (int i = 0; i < num_gpus; i++) {
        // Set device
        cudaSetDevice(i);
        
        // Calculate this GPU's portion
        int start_row = i * rows_per_gpu;
        int end_row = min((i + 1) * rows_per_gpu, height);
        int rows = end_row - start_row;
        
        // Create stream
        cudaStreamCreate(&streams[i]);
        
        // Allocate device memory
        cudaMalloc(&d_images[i], width * rows * sizeof(unsigned char));
        cudaMalloc(&d_results[i], width * rows * sizeof(unsigned char));
        
        // Copy this GPU's portion of the image
        cudaMemcpyAsync(d_images[i], h_image + start_row * width,
                       width * rows * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        
        process_kernel<<<grid, block, 0, streams[i]>>>(d_images[i], d_results[i], width, rows);
        
        // Copy results back
        cudaMemcpyAsync(h_result + start_row * width, d_results[i],
                       width * rows * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaFree(d_images[i]);
        cudaFree(d_results[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    delete[] d_images;
    delete[] d_results;
    delete[] streams;
}
```

#### Task Decomposition

For problems with independent tasks, each GPU can process a subset of tasks:

```cpp
// Example: Processing multiple independent tasks
void process_tasks_multi_gpu(Task* tasks, int num_tasks, int num_gpus) {
    // Calculate tasks per GPU
    int tasks_per_gpu = (num_tasks + num_gpus - 1) / num_gpus;
    
    // Allocate device memory and streams for each GPU
    Task** d_tasks = new Task*[num_gpus];
    Result** d_results = new Result*[num_gpus];
    Result** h_results = new Result*[num_gpus];
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    
    for (int i = 0; i < num_gpus; i++) {
        // Set device
        cudaSetDevice(i);
        
        // Calculate this GPU's portion
        int start_task = i * tasks_per_gpu;
        int end_task = min((i + 1) * tasks_per_gpu, num_tasks);
        int num_gpu_tasks = end_task - start_task;
        
        if (num_gpu_tasks <= 0) continue;
        
        // Create stream
        cudaStreamCreate(&streams[i]);
        
        // Allocate device memory
        cudaMalloc(&d_tasks[i], num_gpu_tasks * sizeof(Task));
        cudaMalloc(&d_results[i], num_gpu_tasks * sizeof(Result));
        
        // Allocate host memory for results
        h_results[i] = new Result[num_gpu_tasks];
        
        // Copy this GPU's tasks
        cudaMemcpyAsync(d_tasks[i], tasks + start_task,
                       num_gpu_tasks * sizeof(Task),
                       cudaMemcpyHostToDevice, streams[i]);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (num_gpu_tasks + block_size - 1) / block_size;
        
        process_task_kernel<<<grid_size, block_size, 0, streams[i]>>>
            (d_tasks[i], d_results[i], num_gpu_tasks);
        
        // Copy results back
        cudaMemcpyAsync(h_results[i], d_results[i],
                       num_gpu_tasks * sizeof(Result),
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // Merge results
    for (int i = 0; i < num_gpus; i++) {
        int start_task = i * tasks_per_gpu;
        int end_task = min((i + 1) * tasks_per_gpu, num_tasks);
        int num_gpu_tasks = end_task - start_task;
        
        if (num_gpu_tasks <= 0) continue;
        
        // Process results as needed
        for (int j = 0; j < num_gpu_tasks; j++) {
            // Do something with h_results[i][j]
        }
    }
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaFree(d_tasks[i]);
        cudaFree(d_results[i]);
        cudaStreamDestroy(streams[i]);
        delete[] h_results[i];
    }
    
    delete[] d_tasks;
    delete[] d_results;
    delete[] h_results;
    delete[] streams;
}
```

### Load Balancing

Effective load balancing ensures that all GPUs are utilized efficiently, especially when dealing with heterogeneous workloads or GPUs with different capabilities.

#### Static Load Balancing

```cpp
// Example: Static load balancing based on GPU compute capability
void static_load_balancing(Task* tasks, int num_tasks, int num_gpus) {
    // Get compute capabilities of all GPUs
    float* compute_powers = new float[num_gpus];
    float total_power = 0.0f;
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        // Simplified compute power estimation
        compute_powers[i] = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        total_power += compute_powers[i];
    }
    
    // Normalize compute powers
    for (int i = 0; i < num_gpus; i++) {
        compute_powers[i] /= total_power;
    }
    
    // Distribute tasks based on compute power
    int* tasks_per_gpu = new int[num_gpus];
    int* start_indices = new int[num_gpus];
    
    int assigned_tasks = 0;
    for (int i = 0; i < num_gpus; i++) {
        tasks_per_gpu[i] = (i < num_gpus - 1) ?
            (int)(compute_powers[i] * num_tasks) :
            (num_tasks - assigned_tasks);
        
        start_indices[i] = assigned_tasks;
        assigned_tasks += tasks_per_gpu[i];
    }
    
    // Process tasks on each GPU
    // ...
    
    delete[] compute_powers;
    delete[] tasks_per_gpu;
    delete[] start_indices;
}
```

#### Dynamic Load Balancing

```cpp
// Example: Dynamic load balancing with task queue
void dynamic_load_balancing(Task* tasks, int num_tasks, int num_gpus) {
    // Create a task queue
    std::queue<int> task_queue;
    for (int i = 0; i < num_tasks; i++) {
        task_queue.push(i);
    }
    
    // Create mutex for queue access
    std::mutex queue_mutex;
    
    // Create worker threads for each GPU
    std::vector<std::thread> workers;
    std::atomic<int> completed_tasks(0);
    
    for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
        workers.push_back(std::thread([&, gpu_id]() {
            cudaSetDevice(gpu_id);
            
            // Allocate GPU resources
            Task* d_task;
            Result* d_result;
            Result h_result;
            
            cudaMalloc(&d_task, sizeof(Task));
            cudaMalloc(&d_result, sizeof(Result));
            
            while (true) {
                // Get next task
                int task_id;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (task_queue.empty()) break;
                    
                    task_id = task_queue.front();
                    task_queue.pop();
                }
                
                // Process task
                cudaMemcpy(d_task, &tasks[task_id], sizeof(Task), cudaMemcpyHostToDevice);
                process_single_task<<<1, 256>>>(d_task, d_result);
                cudaMemcpy(&h_result, d_result, sizeof(Result), cudaMemcpyDeviceToHost);
                
                // Store result
                // ...
                
                completed_tasks++;
            }
            
            // Free GPU resources
            cudaFree(d_task);
            cudaFree(d_result);
        }));
    }
    
    // Wait for all workers to finish
    for (auto& worker : workers) {
        worker.join();
    }
}
```

## Inter-GPU Communication Strategies

Efficient communication between GPUs is crucial for problems that require data exchange during computation.

### Peer-to-Peer (P2P) Memory Access

Modern NVIDIA GPUs support direct memory access between devices on the same PCIe root complex:

```cpp
// Example: Setting up peer-to-peer access
void setup_p2p_access(int num_gpus) {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) continue;
            
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            
            if (can_access) {
                cudaDeviceEnablePeerAccess(j, 0);
                printf("GPU %d can access GPU %d memory\n", i, j);
            }
        }
    }
}

// Example: Using peer-to-peer memory copy
void p2p_memory_copy(float* d_src, float* d_dst, size_t size, int src_gpu, int dst_gpu) {
    cudaSetDevice(src_gpu);
    cudaMemcpyPeer(d_dst, dst_gpu, d_src, src_gpu, size);
}
```

### Unified Virtual Addressing (UVA)

UVA provides a single virtual address space across all GPUs and the CPU:

```cpp
// Example: Using UVA for multi-GPU access
void use_uva(int num_gpus) {
    // Allocate memory on each GPU
    float** d_data = new float*[num_gpus];
    size_t size = 1024 * sizeof(float);
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], size);
        
        // Initialize data
        float value = (float)i;
        cudaMemset(d_data[i], value, size);
    }
    
    // Launch kernel that accesses memory from different GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        
        // Access data from next GPU (circular)
        int next_gpu = (i + 1) % num_gpus;
        
        dim3 block(256);
        dim3 grid((1024 + block.x - 1) / block.x);
        
        access_remote_memory<<<grid, block>>>(d_data[i], d_data[next_gpu], 1024);
    }
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaFree(d_data[i]);
    }
    
    delete[] d_data;
}

__global__ void access_remote_memory(float* local_data, float* remote_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Read from remote memory and write to local memory
        local_data[idx] += remote_data[idx];
    }
}
```

### NCCL (NVIDIA Collective Communications Library)

NCCL provides optimized collective communication primitives for multi-GPU systems:

```cpp
// Example: Using NCCL for all-reduce operation
void nccl_all_reduce(float** d_data, int num_gpus, int elements_per_gpu) {
    // Initialize NCCL
    ncclComm_t* comms = new ncclComm_t[num_gpus];
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    
    // Set up devices and streams
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }
    
    // Initialize NCCL communicators
    ncclCommInitAll(comms, num_gpus, nullptr);
    
    // Perform all-reduce operation
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        ncclAllReduce(d_data[i], d_data[i], elements_per_gpu, ncclFloat, ncclSum,
                     comms[i], streams[i]);
    }
    
    // Synchronize
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }
    
    // Clean up
    for (int i = 0; i < num_gpus; i++) {
        ncclCommDestroy(comms[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    delete[] comms;
    delete[] streams;
}
```

### Host Memory as Intermediary

When direct GPU-to-GPU communication is not possible, host memory can serve as an intermediary:

```cpp
// Example: Using host memory for GPU-to-GPU transfer
void host_memory_transfer(float* d_src, float* d_dst, size_t size, int src_gpu, int dst_gpu) {
    // Allocate pinned host memory
    float* h_buffer;
    cudaMallocHost(&h_buffer, size);
    
    // Copy from source GPU to host
    cudaSetDevice(src_gpu);
    cudaMemcpy(h_buffer, d_src, size, cudaMemcpyDeviceToHost);
    
    // Copy from host to destination GPU
    cudaSetDevice(dst_gpu);
    cudaMemcpy(d_dst, h_buffer, size, cudaMemcpyHostToDevice);
    
    // Free host memory
    cudaFreeHost(h_buffer);
}
```

## Scaling Considerations

Scaling applications across multiple GPUs introduces several considerations that affect performance and design decisions.

### Communication Overhead

As the number of GPUs increases, communication overhead can become a bottleneck:

```cpp
// Example: Analyzing communication overhead
void analyze_communication_overhead(int num_gpus, size_t data_size) {
    // Measure P2P transfer time
    float** d_data = new float*[num_gpus];
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate memory on each GPU
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_data[i], data_size);
    }
    
    // Measure transfer time between each pair of GPUs
    for (int src = 0; src < num_gpus; src++) {
        for (int dst = 0; dst < num_gpus; dst++) {
            if (src == dst) continue;
            
            cudaSetDevice(src);
            cudaEventRecord(start);
            
            cudaMemcpyPeer(d_data[dst], dst, d_data[src], src, data_size);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            float bandwidth = data_size / (milliseconds * 1.0e-3) / 1.0e9; // GB/s
            printf("Transfer from GPU %d to GPU %d: %.2f GB/s\n", src, dst, bandwidth);
        }
    }
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaFree(d_data[i]);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] d_data;
}
```

### Computation-to-Communication Ratio

The ratio of computation to communication is crucial for scaling efficiency:

```cpp
// Example: Analyzing computation-to-communication ratio
void analyze_comp_comm_ratio(int num_gpus, int problem_size) {
    // Estimate computation time per GPU
    float comp_time = estimate_computation_time(problem_size / num_gpus);
    
    // Estimate communication time between GPUs
    float comm_time = estimate_communication_time(problem_size / num_gpus, num_gpus);
    
    // Calculate ratio
    float ratio = comp_time / comm_time;
    
    printf("Computation time: %.2f ms\n", comp_time);
    printf("Communication time: %.2f ms\n", comm_time);
    printf("Computation-to-communication ratio: %.2f\n", ratio);
    
    if (ratio > 10.0f) {
        printf("High ratio: Good scaling expected\n");
    } else if (ratio > 1.0f) {
        printf("Medium ratio: Moderate scaling expected\n");
    } else {
        printf("Low ratio: Poor scaling expected, communication bound\n");
    }
}
```

### Amdahl's Law in Multi-GPU Context

Amdahl's Law helps predict the theoretical speedup with multiple GPUs:

```cpp
// Example: Applying Amdahl's Law to multi-GPU scaling
float predict_speedup(float serial_fraction, int num_gpus) {
    return 1.0f / (serial_fraction + (1.0f - serial_fraction) / num_gpus);
}

void analyze_scaling(float serial_fraction) {
    printf("Predicted speedups with serial fraction %.2f:\n", serial_fraction);
    for (int gpus = 1; gpus <= 16; gpus *= 2) {
        float speedup = predict_speedup(serial_fraction, gpus);
        float efficiency = speedup / gpus * 100.0f;
        printf("%d GPUs: %.2fx speedup (%.2f%% efficiency)\n", 
               gpus, speedup, efficiency);
    }
}
```

## Heterogeneous Systems with Different GPU Models

Many systems contain GPUs with different capabilities, requiring special handling for optimal performance.

### Device Capability Detection

```cpp
// Example: Detecting and categorizing GPU capabilities
void detect_gpu_capabilities(std::vector<GPUInfo>& gpu_info) {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        GPUInfo info;
        info.id = i;
        info.name = prop.name;
        info.compute_capability = prop.major * 10 + prop.minor;
        info.memory_size = prop.totalGlobalMem;
        info.num_sms = prop.multiProcessorCount;
        info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        info.compute_power = info.num_sms * info.max_threads_per_sm;
        
        gpu_info.push_back(info);
        
        printf("GPU %d: %s (CC %d.%d, %d SMs, %.1f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount, prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    }
}
```

### Workload Distribution Based on Capabilities

```cpp
// Example: Distributing workload based on GPU capabilities
void distribute_workload(const std::vector<GPUInfo>& gpu_info, int total_work) {
    // Calculate total compute power
    float total_power = 0.0f;
    for (const auto& gpu : gpu_info) {
        total_power += gpu.compute_power;
    }
    
    // Distribute work proportionally
    int assigned_work = 0;
    for (size_t i = 0; i < gpu_info.size(); i++) {
        float ratio = gpu_info[i].compute_power / total_power;
        int gpu_work = (i == gpu_info.size() - 1) ?
            (total_work - assigned_work) :
            (int)(ratio * total_work);
        
        assigned_work += gpu_work;
        
        printf("GPU %d (%s): %d work units (%.1f%%)\n",
               gpu_info[i].id, gpu_info[i].name,
               gpu_work, ratio * 100.0f);
    }
}
```

### Feature-Based Task Assignment

```cpp
// Example: Assigning tasks based on GPU features
void assign_tasks_by_features(const std::vector<GPUInfo>& gpu_info, 
                             const std::vector<Task>& tasks) {
    // Group GPUs by compute capability
    std::map<int, std::vector<int>> gpu_groups;
    for (const auto& gpu : gpu_info) {
        gpu_groups[gpu.compute_capability].push_back(gpu.id);
    }
    
    // Assign tasks based on requirements
    for (const auto& task : tasks) {
        int required_cc = task.min_compute_capability;
        bool assigned = false;
        
        // Find suitable GPU group
        for (auto it = gpu_groups.rbegin(); it != gpu_groups.rend(); ++it) {
            if (it->first >= required_cc) {
                // Assign to first GPU in this group
                int gpu_id = it->second[0];
                printf("Assigning task %d to GPU %d (CC %d)\n",
                       task.id, gpu_id, it->first);
                
                // Rotate GPUs in this group for load balancing
                std::rotate(it->second.begin(), it->second.begin() + 1, it->second.end());
                
                assigned = true;
                break;
            }
        }
        
        if (!assigned) {
            printf("Task %d cannot be assigned (requires CC %d)\n",
                   task.id, required_cc);
        }
    }
}
```

## Case Study: Multi-GPU Matrix Multiplication

Let's implement a multi-GPU matrix multiplication algorithm to demonstrate these concepts:

```cpp
// Multi-GPU matrix multiplication
void multi_gpu_matrix_multiply(float* h_A, float* h_B, float* h_C,
                              int M, int N, int K, int num_gpus) {
    // Divide matrix A along rows
    int rows_per_gpu = (M + num_gpus - 1) / num_gpus;
    
    // Allocate device memory and streams
    float** d_A = new float*[num_gpus];
    float** d_B = new float*[num_gpus];
    float** d_C = new float*[num_gpus];
    cudaStream_t** streams = new cudaStream_t*[num_gpus];
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        
        // Calculate this GPU's portion
        int start_row = i * rows_per_gpu;
        int end_row = min((i + 1) * rows_per_gpu, M);
        int rows = end_row - start_row;
        
        if (rows <= 0) continue;
        
        // Allocate device memory
        cudaMalloc(&d_A[i], rows * K * sizeof(float));
        cudaMalloc(&d_B[i], K * N * sizeof(float));
        cudaMalloc(&d_C[i], rows * N * sizeof(float));
        
        // Create streams (one per chunk to enable overlap)
        const int num_streams = 4;
        streams[i] = new cudaStream_t[num_streams];
        for (int j = 0; j < num_streams; j++) {
            cudaStreamCreate(&streams[i][j]);
        }
        
        // Copy matrix B (needed by all GPUs)
        cudaMemcpy(d_B[i], h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        
        // Copy this GPU's portion of matrix A in chunks to overlap with computation
        int chunk_size = (rows + num_streams - 1) / num_streams;
        
        for (int j = 0; j < num_streams; j++) {
            int chunk_start = j * chunk_size;
            int chunk_end = min((j + 1) * chunk_size, rows);
            int chunk_rows = chunk_end - chunk_start;
            
            if (chunk_rows <= 0) continue;
            
            // Copy chunk of A
            cudaMemcpyAsync(d_A[i] + chunk_start * K,
                           h_A + (start_row + chunk_start) * K,
                           chunk_rows * K * sizeof(float),
                           cudaMemcpyHostToDevice, streams[i][j]);
            
            // Launch kernel for this chunk
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (chunk_rows + block.y - 1) / block.y);
            
            matrix_multiply_kernel<<<grid, block, 0, streams[i][j]>>>
                (d_A[i] + chunk_start * K, d_B[i], d_C[i] + chunk_start * N,
                 chunk_rows, N, K);
            
            // Copy result chunk back
            cudaMemcpyAsync(h_C + (start_row + chunk_start) * N,
                           d_C[i] + chunk_start * N,
                           chunk_rows * N * sizeof(float),
                           cudaMemcpyDeviceToHost, streams[i][j]);
        }
    }
    
    // Synchronize all GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        
        int start_row = i * rows_per_gpu;
        int end_row = min((i + 1) * rows_per_gpu, M);
        int rows = end_row - start_row;
        
        if (rows <= 0) continue;
        
        const int num_streams = 4;
        for (int j = 0; j < num_streams; j++) {
            cudaStreamSynchronize(streams[i][j]);
        }
    }
    
    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        
        int start_row = i * rows_per_gpu;
        int end_row = min((i + 1) * rows_per_gpu, M);
        int rows = end_row - start_row;
        
        if (rows <= 0) continue;
        
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
        
        const int num_streams = 4;
        for (int j = 0; j < num_streams; j++) {
            cudaStreamDestroy(streams[i][j]);
        }
        delete[] streams[i];
    }
    
    delete[] d_A;
    delete[] d_B;
    delete[] d_C;
    delete[] streams;
}

__global__ void matrix_multiply_kernel(float* A, float* B, float* C,
                                     int M, int N, int K) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output element coordinates
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

## Conclusion

Multi-GPU programming offers a powerful approach to scaling GPU applications beyond the limits of a single device. By effectively distributing work, managing communication, and accounting for system heterogeneity, you can achieve significant performance improvements for large-scale computational problems.

Key takeaways from this article include:

1. **Work Distribution**: Choose domain or task decomposition based on your problem structure and data dependencies
2. **Load Balancing**: Use static or dynamic load balancing to ensure efficient GPU utilization
3. **Communication**: Leverage P2P access, UVA, or libraries like NCCL for efficient inter-GPU communication
4. **Scaling Considerations**: Analyze communication overhead and computation-to-communication ratio to predict scaling efficiency
5. **Heterogeneous Systems**: Detect GPU capabilities and distribute work accordingly for optimal performance

In our next article, we'll explore GPU compilers and code generation, focusing on how GPU code is translated into efficient machine instructions.

## Exercises for Practice

1. **Multi-GPU Reduction**: Implement a multi-GPU reduction algorithm that combines partial results from each GPU to compute a global sum of a large array.

2. **Load Balancing**: Implement and compare static and dynamic load balancing strategies for a task-parallel application running on multiple GPUs.

3. **Communication Benchmarking**: Create a benchmark to measure the bandwidth and latency between different GPUs in your system using various communication methods (P2P, host memory, NCCL).

4. **Heterogeneous Processing**: Implement a task scheduler that assigns different types of tasks to GPUs based on their compute capabilities.

5. **Scaling Analysis**: Analyze the scaling efficiency of a multi-GPU application as you increase the number of GPUs, and identify the bottlenecks limiting performance.

## Further Resources

- [NVIDIA Multi-GPU Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-gpu-programming)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [Peer-to-Peer Communication in CUDA](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)
- [Multi-GPU Programming Models](https://www.nvidia.com/content/GTC-2010/pdfs/2012_GTC2010.pdf)
- [Scaling Deep Learning on Multiple GPUs](https://developer.nvidia.com/blog/scaling-deep-learning-training-with-tensorflow-gpu/)