# GPU Computing for Machine Learning

*Welcome to the thirteenth installment of our GPU programming series! In this article, we'll explore how GPUs have revolutionized machine learning, particularly deep learning. We'll dive into specialized libraries, tensor operations, and optimization techniques that make GPUs the backbone of modern AI systems.*

## Introduction to GPU-Accelerated Machine Learning

Machine learning, especially deep learning, involves massive matrix operations and parallel computations that are perfectly suited for GPU acceleration. The rise of deep learning has been closely tied to advances in GPU computing, enabling models that would be impractical to train on traditional CPUs.

In this article, we'll explore the specialized libraries, tensor operations, optimization techniques, and considerations for both training and inference that make GPUs indispensable for modern machine learning.

## CUDA Libraries for Deep Learning

NVIDIA provides several specialized libraries that form the foundation of GPU-accelerated machine learning frameworks.

### cuBLAS (CUDA Basic Linear Algebra Subroutines)

cuBLAS provides GPU-accelerated implementations of standard linear algebra operations, which are fundamental to machine learning algorithms.

```cpp
// Example: Matrix multiplication with cuBLAS
#include <cublas_v2.h>

void matrix_multiply_cublas(float* A, float* B, float* C, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // C = alpha * (A * B) + beta * C
    // Note: cuBLAS uses column-major order
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B, n,    // B is n x k
                A, k,    // A is m x k
                &beta,
                C, n);   // C is m x n
    
    cublasDestroy(handle);
}
```

### cuDNN (CUDA Deep Neural Network Library)

cuDNN provides highly optimized implementations of operations commonly used in deep neural networks, such as convolutions, pooling, normalization, and activation functions.

```cpp
// Example: 2D Convolution with cuDNN
#include <cudnn.h>

void convolution_2d_cudnn(float* input, float* filters, float* output,
                         int batch_size, int in_channels, int in_height, int in_width,
                         int out_channels, int filter_height, int filter_width) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    // Input tensor descriptor
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                              batch_size, in_channels, in_height, in_width);
    
    // Filter descriptor
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnCreateFilterDescriptor(&filter_descriptor);
    cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             out_channels, in_channels, filter_height, filter_width);
    
    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor, 0, 0, 1, 1, 1, 1,
                                   CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // Output tensor descriptor
    int out_height, out_width;
    cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, filter_descriptor,
                                         &batch_size, &out_channels, &out_height, &out_width);
    
    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                              batch_size, out_channels, out_height, out_width);
    
    // Find the best algorithm
    cudnnConvolutionFwdAlgo_t algorithm;
    cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor,
                                       conv_descriptor, output_descriptor,
                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm);
    
    // Allocate workspace
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor,
                                           conv_descriptor, output_descriptor,
                                           algorithm, &workspace_size);
    
    void* workspace;
    cudaMalloc(&workspace, workspace_size);
    
    // Perform convolution
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnConvolutionForward(cudnn, &alpha, input_descriptor, input,
                           filter_descriptor, filters, conv_descriptor, algorithm,
                           workspace, workspace_size, &beta, output_descriptor, output);
    
    // Clean up
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroy(cudnn);
}
```

### cuSPARSE and cuSOLVER

These libraries provide sparse matrix operations and numerical solvers that are useful for certain machine learning algorithms:

```cpp
// Example: Sparse matrix-vector multiplication with cuSPARSE
#include <cusparse.h>

void sparse_matrix_vector_multiply(int* row_ptr, int* col_indices, float* values,
                                 int num_rows, int num_cols, int nnz,
                                 float* x, float* y) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create sparse matrix in CSR format
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    // Perform SpMV: y = alpha * A * x + beta * y
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                  num_rows, num_cols, nnz, &alpha, descr,
                  values, row_ptr, col_indices, x, &beta, y);
    
    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
}
```

### NCCL (NVIDIA Collective Communications Library)

NCCL enables efficient multi-GPU and multi-node communication, which is crucial for distributed training of large models:

```cpp
// Example: All-reduce operation with NCCL
#include <nccl.h>

void distributed_gradient_update(float** gradients, int num_gpus, int gradient_size) {
    // Initialize NCCL
    ncclComm_t* comms = new ncclComm_t[num_gpus];
    ncclCommInitAll(comms, num_gpus, nullptr); // Assumes GPUs are in same node
    
    // Create CUDA streams
    cudaStream_t* streams = new cudaStream_t[num_gpus];
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }
    
    // Perform all-reduce to average gradients across GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        ncclAllReduce(gradients[i], gradients[i], gradient_size,
                     ncclFloat, ncclSum, comms[i], streams[i]);
    }
    
    // Scale gradients by 1/num_gpus to get average
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        float scale = 1.0f / num_gpus;
        cublasSscal(gradient_size, &scale, gradients[i], 1);
    }
    
    // Synchronize and clean up
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        ncclCommDestroy(comms[i]);
    }
    
    delete[] comms;
    delete[] streams;
}
```

### TensorRT

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime:

```cpp
// Example: TensorRT inference optimization
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO) std::cout << msg << std::endl;
    }
} logger;

void optimize_model_with_tensorrt(const std::string& onnx_model_path) {
    // Create builder
    auto builder = nvinfer1::createInferBuilder(logger);
    auto network = builder->createNetworkV2(1U << static_cast<uint32_t>
                                         (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, logger);
    
    // Parse ONNX model
    parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    
    // Set optimization configuration
    config->setMaxWorkspaceSize(1 << 30); // 1GB
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // Enable FP16 precision
    
    // Build optimized engine
    auto engine = builder->buildEngineWithConfig(*network, *config);
    
    // Serialize engine for later use
    auto serialized_engine = engine->serialize();
    std::ofstream engine_file("optimized_model.trt", std::ios::binary);
    engine_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    
    // Clean up
    serialized_engine->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    parser->destroy();
}
```

## Tensor Computations on GPUs

Tensors are the fundamental data structures in deep learning, and efficient tensor operations are key to high-performance implementations.

### Tensor Storage and Memory Layout

Efficient tensor storage and memory layout are crucial for performance:

```cpp
// Example: Different tensor memory layouts

// NCHW layout (channels-first) - common in cuDNN
void fill_tensor_nchw(float* tensor, int batch, int channels, int height, int width, float value) {
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = n * channels * height * width +
                              c * height * width +
                              h * width +
                              w;
                    tensor[idx] = value;
                }
            }
        }
    }
}

// NHWC layout (channels-last) - common in TensorFlow
void fill_tensor_nhwc(float* tensor, int batch, int channels, int height, int width, float value) {
    for (int n = 0; n < batch; n++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int idx = n * height * width * channels +
                              h * width * channels +
                              w * channels +
                              c;
                    tensor[idx] = value;
                }
            }
        }
    }
}
```

### Custom CUDA Kernels for Tensor Operations

Sometimes, you need custom kernels for specialized tensor operations:

```cuda
// Example: Custom element-wise tensor operation
__global__ void tensor_activation(float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Leaky ReLU activation
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * x;
    }
}

// Example: Custom tensor reduction
__global__ void tensor_row_sum(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += input[row * cols + col];
        }
        output[row] = sum;
    }
}
```

### Tensor Contraction and Broadcasting

Tensor contraction (generalized matrix multiplication) and broadcasting are common operations in deep learning:

```cuda
// Example: Tensor contraction (batched matrix multiplication)
__global__ void batched_matmul(float* A, float* B, float* C,
                             int batch_size, int m, int n, int k) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch < batch_size && row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            // A is batch_size x m x k, B is batch_size x k x n
            sum += A[batch * m * k + row * k + i] * B[batch * k * n + i * n + col];
        }
        C[batch * m * n + row * n + col] = sum;
    }
}

// Example: Broadcasting (adding a vector to each row of a matrix)
__global__ void broadcast_add(float* matrix, float* vector, float* output,
                            int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[row * cols + col] = matrix[row * cols + col] + vector[col];
    }
}
```

## Optimizing Neural Network Operations

Deep learning frameworks implement various optimizations for neural network operations.

### Convolution Algorithms

Convolution is a key operation in CNNs, and several algorithms exist with different performance characteristics:

```cpp
// Example: Different convolution algorithms with cuDNN
void convolution_algorithm_benchmark(float* input, float* filters, float* output,
                                   int batch_size, int in_channels, int height, int width,
                                   int out_channels, int filter_size) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    
    // Set up tensor descriptors (similar to previous example)
    // ...
    
    // Benchmark different algorithms
    cudnnConvolutionFwdAlgo_t algorithms[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    };
    
    const char* algo_names[] = {
        "IMPLICIT_GEMM",
        "IMPLICIT_PRECOMP_GEMM",
        "GEMM",
        "DIRECT",
        "FFT",
        "FFT_TILING",
        "WINOGRAD",
        "WINOGRAD_NONFUSED"
    };
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    for (int i = 0; i < 8; i++) {
        // Check if algorithm is supported
        size_t workspace_size;
        cudnnStatus_t status = cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_descriptor, filter_descriptor, conv_descriptor,
            output_descriptor, algorithms[i], &workspace_size);
        
        if (status != CUDNN_STATUS_SUCCESS) {
            printf("Algorithm %s not supported\n", algo_names[i]);
            continue;
        }
        
        // Allocate workspace
        void* workspace;
        cudaMalloc(&workspace, workspace_size);
        
        // Measure performance
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int iter = 0; iter < 10; iter++) {
            cudnnConvolutionForward(
                cudnn, &alpha, input_descriptor, input,
                filter_descriptor, filters, conv_descriptor, algorithms[i],
                workspace, workspace_size, &beta, output_descriptor, output);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Algorithm %s: %.3f ms (workspace: %.1f MB)\n",
               algo_names[i], milliseconds / 10.0f, workspace_size / (1024.0f * 1024.0f));
        
        // Clean up
        cudaFree(workspace);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Clean up descriptors
    // ...
    
    cudnnDestroy(cudnn);
}
```

### Memory Optimizations

Memory optimizations are crucial for large models:

```cpp
// Example: Memory-efficient activation storage
void memory_efficient_forward(float* layer_input, float* weights, float* output,
                            int batch_size, int input_size, int output_size) {
    // Forward pass with activation storage
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               output_size, batch_size, input_size,
               &alpha, weights, output_size, layer_input, input_size,
               &beta, output, output_size);
    
    // Apply ReLU in-place to save memory
    int total_elements = batch_size * output_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    relu_inplace<<<grid_size, block_size>>>(output, total_elements);
}

__global__ void relu_inplace(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}
```

### Mixed Precision Training

Mixed precision training uses lower precision (e.g., FP16) for most operations while maintaining model accuracy:

```cpp
// Example: Mixed precision matrix multiplication with cuBLAS
void mixed_precision_matmul(half* A_fp16, half* B_fp16, float* C_fp32,
                          int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable tensor cores for mixed precision
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Perform matrix multiplication in FP16, but accumulate in FP32
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B_fp16, CUDA_R_16F, n,
                A_fp16, CUDA_R_16F, k,
                &beta,
                C_fp32, CUDA_R_32F, n,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cublasDestroy(handle);
}

// Example: Converting between precisions
__global__ void convert_float_to_half(float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void convert_half_to_float(half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}
```

### Kernel Fusion

Kernel fusion combines multiple operations into a single kernel to reduce memory traffic:

```cuda
// Example: Fused bias addition and activation
__global__ void fused_bias_relu(float* input, float* bias, float* output,
                              int batch_size, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        // Calculate channel index for bias
        int c = (idx / (height * width)) % channels;
        
        // Fused bias add and ReLU
        float val = input[idx] + bias[c];
        output[idx] = fmaxf(0.0f, val);
    }
}

// Example: Fused batch normalization and activation
__global__ void fused_batchnorm_relu(float* input, float* gamma, float* beta,
                                   float* mean, float* var, float* output,
                                   int batch_size, int channels, int height, int width,
                                   float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        // Calculate channel index
        int c = (idx / (height * width)) % channels;
        
        // Fused batch normalization and ReLU
        float normalized = (input[idx] - mean[c]) / sqrtf(var[c] + epsilon);
        float scaled = gamma[c] * normalized + beta[c];
        output[idx] = fmaxf(0.0f, scaled);
    }
}
```

## Training vs. Inference Considerations

Training and inference have different requirements and optimizations.

### Training Optimizations

Training requires storing intermediate activations for backpropagation and handling gradient updates:

```cpp
// Example: Training loop with gradient accumulation
void training_loop(float* input, float* target, float* weights,
                 int batch_size, int input_size, int output_size,
                 float learning_rate, int num_batches) {
    // Allocate memory for activations, gradients, etc.
    float *output, *d_output, *d_weights;
    cudaMalloc(&output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    for (int batch = 0; batch < num_batches; batch++) {
        // Forward pass
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   output_size, batch_size, input_size,
                   &alpha, weights, output_size, input, input_size,
                   &beta, output, output_size);
        
        // Compute loss gradient
        int block_size = 256;
        int grid_size = (batch_size * output_size + block_size - 1) / block_size;
        compute_loss_gradient<<<grid_size, block_size>>>(output, target, d_output,
                                                      batch_size, output_size);
        
        // Backward pass (compute weight gradients)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                   output_size, input_size, batch_size,
                   &alpha, d_output, output_size, input, input_size,
                   &beta, d_weights, output_size);
        
        // Update weights
        grid_size = (input_size * output_size + block_size - 1) / block_size;
        update_weights<<<grid_size, block_size>>>(weights, d_weights,
                                               input_size * output_size, learning_rate);
    }
    
    // Clean up
    cublasDestroy(handle);
    cudaFree(output);
    cudaFree(d_output);
    cudaFree(d_weights);
}

__global__ void compute_loss_gradient(float* output, float* target, float* gradient,
                                    int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        // Simple MSE loss gradient
        gradient[idx] = 2.0f * (output[idx] - target[idx]);
    }
}

__global__ void update_weights(float* weights, float* gradients,
                             int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}
```

### Inference Optimizations

Inference focuses on minimizing latency and memory usage:

```cpp
// Example: Optimized inference with weight fusion and quantization
void optimized_inference(uint8_t* quantized_weights, float* scales, float* biases,
                       uint8_t* input, float input_scale, float* output,
                       int batch_size, int input_size, int output_size) {
    // Allocate device memory
    uint8_t *d_weights, *d_input;
    float *d_output, *d_scales, *d_biases;
    
    cudaMalloc(&d_weights, input_size * output_size * sizeof(uint8_t));
    cudaMalloc(&d_input, batch_size * input_size * sizeof(uint8_t));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_scales, output_size * sizeof(float));
    cudaMalloc(&d_biases, output_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_weights, quantized_weights, input_size * output_size * sizeof(uint8_t),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, batch_size * input_size * sizeof(uint8_t),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales, output_size * sizeof(float),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, output_size * sizeof(float),
              cudaMemcpyHostToDevice);
    
    // Launch optimized kernel
    dim3 block(256);
    dim3 grid((batch_size * output_size + block.x - 1) / block.x);
    
    quantized_fc_inference<<<grid, block>>>(d_weights, d_scales, d_biases,
                                          d_input, input_scale, d_output,
                                          batch_size, input_size, output_size);
    
    // Copy results back
    cudaMemcpy(output, d_output, batch_size * output_size * sizeof(float),
              cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scales);
    cudaFree(d_biases);
}

__global__ void quantized_fc_inference(uint8_t* weights, float* scales, float* biases,
                                     uint8_t* input, float input_scale, float* output,
                                     int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int batch = idx / output_size;
        int out_idx = idx % output_size;
        
        // Compute dot product with quantized values
        int32_t sum = 0;
        for (int i = 0; i < input_size; i++) {
            int w_idx = out_idx * input_size + i;
            int in_idx = batch * input_size + i;
            
            // Convert uint8 to int32 and subtract zero point (128)
            int32_t w = static_cast<int32_t>(weights[w_idx]) - 128;
            int32_t x = static_cast<int32_t>(input[in_idx]) - 128;
            
            sum += w * x;
        }
        
        // Dequantize and add bias
        float scale = scales[out_idx] * input_scale;
        output[idx] = sum * scale + biases[out_idx];
    }
}
```

### Model Compression Techniques

Model compression reduces model size and improves inference speed:

```cpp
// Example: Weight pruning and sparse matrix operations
void sparse_inference(float* weights, int* indices, int* indptr,
                    float* input, float* output,
                    int batch_size, int input_size, int output_size) {
    // Allocate device memory
    float *d_weights, *d_input, *d_output;
    int *d_indices, *d_indptr;
    
    cudaMalloc(&d_weights, indptr[output_size] * sizeof(float));
    cudaMalloc(&d_indices, indptr[output_size] * sizeof(int));
    cudaMalloc(&d_indptr, (output_size + 1) * sizeof(int));
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_weights, weights, indptr[output_size] * sizeof(float),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, indices, indptr[output_size] * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, indptr, (output_size + 1) * sizeof(int),
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, batch_size * input_size * sizeof(float),
              cudaMemcpyHostToDevice);
    
    // Launch sparse matrix-vector multiplication kernel
    dim3 block(256);
    dim3 grid((batch_size * output_size + block.x - 1) / block.x);
    
    sparse_fc_inference<<<grid, block>>>(d_weights, d_indices, d_indptr,
                                       d_input, d_output,
                                       batch_size, input_size, output_size);
    
    // Copy results back
    cudaMemcpy(output, d_output, batch_size * output_size * sizeof(float),
              cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_weights);
    cudaFree(d_indices);
    cudaFree(d_indptr);
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void sparse_fc_inference(float* weights, int* indices, int* indptr,
                                 float* input, float* output,
                                 int batch_size, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_size) {
        int batch = idx / output_size;
        int out_idx = idx % output_size;
        
        // Compute sparse matrix-vector product
        float sum = 0.0f;
        for (int j = indptr[out_idx]; j < indptr[out_idx + 1]; j++) {
            int col = indices[j];
            sum += weights[j] * input[batch * input_size + col];
        }
        
        output[idx] = sum;
    }
}
```

## Case Study: Implementing a CNN Layer

Let's implement a complete convolutional layer with various optimizations:

```cpp
// Example: Optimized CNN layer implementation
class ConvLayer {
public:
    ConvLayer(int batch_size, int in_channels, int in_height, int in_width,
             int out_channels, int kernel_size, int stride, int padding)
        : batch_size_(batch_size),
          in_channels_(in_channels),
          in_height_(in_height),
          in_width_(in_width),
          out_channels_(out_channels),
          kernel_size_(kernel_size),
          stride_(stride),
          padding_(padding) {
        
        // Calculate output dimensions
        out_height_ = (in_height_ + 2 * padding_ - kernel_size_) / stride_ + 1;
        out_width_ = (in_width_ + 2 * padding_ - kernel_size_) / stride_ + 1;
        
        // Initialize cuDNN
        cudnnCreate(&cudnn_);
        
        // Create tensor descriptors
        cudnnCreateTensorDescriptor(&input_descriptor_);
        cudnnCreateTensorDescriptor(&output_descriptor_);
        cudnnCreateFilterDescriptor(&filter_descriptor_);
        cudnnCreateConvolutionDescriptor(&conv_descriptor_);
        
        cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                  batch_size_, in_channels_, in_height_, in_width_);
        
        cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                  batch_size_, out_channels_, out_height_, out_width_);
        
        cudnnSetFilter4dDescriptor(filter_descriptor_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                 out_channels_, in_channels_, kernel_size_, kernel_size_);
        
        cudnnSetConvolution2dDescriptor(conv_descriptor_, padding_, padding_, stride_, stride_,
                                       1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        
        // Find the best algorithm
        cudnnGetConvolutionForwardAlgorithm(cudnn_, input_descriptor_, filter_descriptor_,
                                           conv_descriptor_, output_descriptor_,
                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algorithm_);
        
        // Allocate workspace
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_, input_descriptor_, filter_descriptor_,
                                               conv_descriptor_, output_descriptor_,
                                               algorithm_, &workspace_size_);
        
        cudaMalloc(&workspace_, workspace_size_);
        
        // Allocate weights and bias
        size_t weights_size = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
        size_t bias_size = out_channels_;
        
        cudaMalloc(&weights_, weights_size * sizeof(float));
        cudaMalloc(&bias_, bias_size * sizeof(float));
        
        // Initialize weights with Xavier/Glorot initialization
        std::vector<float> h_weights(weights_size);
        float scale = sqrtf(2.0f / (in_channels_ * kernel_size_ * kernel_size_ + out_channels_));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (size_t i = 0; i < weights_size; i++) {
            h_weights[i] = dist(gen);
        }
        
        cudaMemcpy(weights_, h_weights.data(), weights_size * sizeof(float),
                  cudaMemcpyHostToDevice);
        
        // Initialize bias to zero
        cudaMemset(bias_, 0, bias_size * sizeof(float));
    }
    
    ~ConvLayer() {
        cudaFree(weights_);
        cudaFree(bias_);
        cudaFree(workspace_);
        
        cudnnDestroyTensorDescriptor(input_descriptor_);
        cudnnDestroyTensorDescriptor(output_descriptor_);
        cudnnDestroyFilterDescriptor(filter_descriptor_);
        cudnnDestroyConvolutionDescriptor(conv_descriptor_);
        
        cudnnDestroy(cudnn_);
    }
    
    void Forward(float* input, float* output) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Perform convolution
        cudnnConvolutionForward(cudnn_, &alpha, input_descriptor_, input,
                               filter_descriptor_, weights_, conv_descriptor_, algorithm_,
                               workspace_, workspace_size_, &beta, output_descriptor_, output);
        
        // Add bias
        cudnnAddTensor(cudnn_, &alpha, bias_descriptor_, bias_,
                      &alpha, output_descriptor_, output);
        
        // Apply ReLU activation
        cudnnActivationForward(cudnn_, activation_desc_, &alpha,
                              output_descriptor_, output, &beta,
                              output_descriptor_, output);
    }
    
    void Backward(float* input, float* d_output, float* d_input, float learning_rate) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Compute gradient of weights
        cudnnConvolutionBackwardFilter(cudnn_, &alpha, input_descriptor_, input,
                                      output_descriptor_, d_output, conv_descriptor_,
                                      &beta, filter_descriptor_, d_weights_);
        
        // Compute gradient of bias
        cudnnConvolutionBackwardBias(cudnn_, &alpha, output_descriptor_, d_output,
                                    &beta, bias_descriptor_, d_bias_);
        
        // Compute gradient of input (if needed)
        if (d_input != nullptr) {
            cudnnConvolutionBackwardData(cudnn_, &alpha, filter_descriptor_, weights_,
                                       output_descriptor_, d_output, conv_descriptor_,
                                       &beta, input_descriptor_, d_input);
        }
        
        // Update weights and bias
        int block_size = 256;
        int weights_size = out_channels_ * in_channels_ * kernel_size_ * kernel_size_;
        int grid_size = (weights_size + block_size - 1) / block_size;
        
        update_parameters<<<grid_size, block_size>>>(weights_, d_weights_, weights_size, learning_rate);
        
        grid_size = (out_channels_ + block_size - 1) / block_size;
        update_parameters<<<grid_size, block_size>>>(bias_, d_bias_, out_channels_, learning_rate);
    }
    
private:
    int batch_size_;
    int in_channels_, in_height_, in_width_;
    int out_channels_, out_height_, out_width_;
    int kernel_size_, stride_, padding_;
    
    cudnnHandle_t cudnn_;
    cudnnTensorDescriptor_t input_descriptor_, output_descriptor_, bias_descriptor_;
    cudnnFilterDescriptor_t filter_descriptor_;
    cudnnConvolutionDescriptor_t conv_descriptor_;
    cudnnActivationDescriptor_t activation_desc_;
    cudnnConvolutionFwdAlgo_t algorithm_;
    
    size_t workspace_size_;
    void* workspace_;
    
    float* weights_;
    float* bias_;
    float* d_weights_;
    float* d_bias_;
};

__global__ void update_parameters(float* params, float* gradients, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * gradients[idx];
    }
}
```

## Conclusion

GPU computing has revolutionized machine learning by enabling the training and deployment of increasingly complex models. The specialized libraries, tensor operations, and optimization techniques we've explored in this article form the foundation of modern deep learning frameworks like TensorFlow, PyTorch, and MXNet.

Key takeaways from this article include:

1. **Specialized Libraries**: Libraries like cuBLAS, cuDNN, and TensorRT provide highly optimized implementations of common deep learning operations
2. **Tensor Operations**: Efficient tensor storage, memory layouts, and custom kernels are crucial for high-performance deep learning
3. **Optimization Techniques**: Techniques like mixed precision training, kernel fusion, and memory optimizations enable training larger models faster
4. **Training vs. Inference**: Different optimization strategies apply to training (where gradient computation is key) and inference (where latency and memory usage are priorities)
5. **Model Compression**: Techniques like quantization, pruning, and knowledge distillation can reduce model size and improve inference speed

In our next article, we'll explore raytracing on GPUs, focusing on how modern GPUs accelerate realistic rendering through hardware-accelerated ray tracing.

## Exercises for Practice

1. **cuDNN Exploration**: Implement a simple CNN using cuDNN and benchmark different convolution algorithms to find the fastest for your specific GPU and problem size.

2. **Mixed Precision Training**: Modify a simple neural network implementation to use mixed precision (FP16 computation with FP32 accumulation) and compare performance and accuracy with full FP32 training.

3. **Kernel Fusion**: Implement a fused kernel that combines batch normalization, activation, and dropout in a single pass, and compare its performance to separate kernels.

4. **Model Compression**: Implement weight quantization for a pre-trained model and measure the impact on inference speed and accuracy.

5. **Multi-GPU Training**: Implement data-parallel training of a neural network across multiple GPUs using NCCL for gradient synchronization.

## Further Resources

- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Mixed Precision Training Paper](https://arxiv.org/abs/1710.03740)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Efficient Methods and Hardware for Deep Learning (Stanford Course)](https://cs217.stanford.edu/)