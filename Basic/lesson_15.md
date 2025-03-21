# Lesson 15: Getting Started with Practical Projects

Welcome to Lesson 15 of our "Accelerating the Future" series! In this lesson, we'll move from theory to practice by exploring how to set up your environment and start working on real accelerated computing projects.

## Setting up your development environment

Before diving into accelerated computing projects, you need to set up a proper development environment:

- **Operating System**: While accelerated computing is possible on most operating systems, Linux distributions (particularly Ubuntu) often provide the best support for development tools.
- **Required Software**:
  - CUDA Toolkit (for NVIDIA GPUs)
  - ROCm (for AMD GPUs)
  - Intel oneAPI (for Intel accelerators)
  - Appropriate compilers (GCC, LLVM)
  - CMake for build management
  - Python with scientific packages (NumPy, SciPy) for high-level work

### Basic Setup Instructions

For NVIDIA CUDA development:
```bash
# Install CUDA Toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit
# Verify installation
nvcc --version
```

For AMD ROCm development:
```bash
# Add ROCm repository (Ubuntu example)
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms
# Verify installation
/opt/rocm/bin/rocminfo
```

## Choosing the right hardware for learning

You don't need the most expensive hardware to start learning:

- **For NVIDIA development**: GTX 1060 or higher (consumer cards work fine for learning)
- **For AMD development**: Radeon RX 580 or higher with ROCm support
- **For Intel development**: Recent Intel CPU with integrated graphics
- **Minimum system requirements**:
  - 16GB RAM recommended (8GB minimum)
  - SSD storage for faster compilation
  - Multi-core CPU (4+ cores recommended)

### Hardware Compatibility Tips

- Check the official compatibility lists before purchasing hardware
- Ensure your motherboard supports the required PCIe specifications
- Verify your power supply can handle the accelerator's requirements

## Cloud-based options for accessing accelerators

Don't have access to powerful hardware? No problem! Several cloud platforms offer GPU instances:

- **Google Colab**: Free access to NVIDIA GPUs with Python and CUDA support
- **AWS EC2**: Various GPU instance types (p2, p3, g4, etc.)
- **Azure**: NC and ND series VMs with NVIDIA GPUs
- **GCP**: Offers T4 and V100 GPU instances
- **Paperspace**: Developer-friendly platform with hourly GPU rentals

### Cloud Setup Example (Google Colab)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
4. Verify GPU access:
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## Simple starter projects with source code

Here are some beginner-friendly projects to get started:

### 1. Vector Addition

A classic first CUDA program:

```cuda
#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;  // Vector size
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < n; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            printf("Verification failed at %d\n", i);
            break;
        }
    }
    printf("Vector addition completed successfully\n");
    
    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

### 2. Matrix Multiplication

A fundamental operation in many computational workloads:

```cuda
#include <stdio.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *a, float *b, float *c, int width) {
    // Calculate row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < width; i++) {
            sum += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    int width = 1024;  // Matrix dimensions
    size_t bytes = width * width * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (width + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    printf("Matrix multiplication completed\n");
    
    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

## Image processing acceleration project walkthrough

Let's walk through a simple image processing example using CUDA to apply a blur filter:

### Gaussian Blur Implementation

```cuda
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// CUDA kernel for Gaussian blur
__global__ void gaussianBlur(unsigned char *input, unsigned char *output, 
                            int width, int height, int channels) {
    // Calculate pixel position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Ensure we're within image bounds
    if (col < width && row < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            float kernel[9] = {1/16.0f, 2/16.0f, 1/16.0f,
                              2/16.0f, 4/16.0f, 2/16.0f,
                              1/16.0f, 2/16.0f, 1/16.0f};
            int count = 0;
            
            // Apply 3x3 kernel
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int currentRow = row + i;
                    int currentCol = col + j;
                    
                    // Check boundaries
                    if (currentRow >= 0 && currentRow < height && 
                        currentCol >= 0 && currentCol < width) {
                        int index = (currentRow * width + currentCol) * channels + c;
                        sum += input[index] * kernel[count];
                    }
                    count++;
                }
            }
            
            // Write result
            output[(row * width + col) * channels + c] = (unsigned char)sum;
        }
    }
}

int main() {
    // Load image using OpenCV
    Mat image = imread("input.jpg");
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return -1;
    }
    
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    
    // Allocate host memory
    unsigned char *h_input = image.data;
    unsigned char *h_output = (unsigned char*)malloc(imageSize);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    
    // Copy input image to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    gaussianBlur<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Create output image
    Mat outputImage(height, width, image.type(), h_output);
    imwrite("blurred.jpg", outputImage);
    
    printf("Image blur completed\n");
    
    // Free memory
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
```

To compile this example, you'll need:
```bash
nvcc -o blur blur.cu `pkg-config --cflags --libs opencv4`
```

## Basic AI inference acceleration example

Here's a simple example of running inference with a pre-trained model using CUDA:

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Prepare image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform an image
image = Image.open("sample_image.jpg")
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0).to(device)

# Measure inference time
start_time = time.time()

# Run inference
with torch.no_grad():
    output = model(input_batch)

end_time = time.time()
inference_time = end_time - start_time

# Get the prediction
_, predicted_idx = torch.max(output, 1)

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Print results
print(f"Predicted class: {classes[predicted_idx]}")
print(f"Inference time: {inference_time:.4f} seconds")
```

## Performance measurement and comparison

When working with accelerated computing, it's essential to measure performance gains:

### Simple Timing Example

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Function to measure execution time
float measureExecutionTime(void (*function)()) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    function();
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// CPU implementation of vector addition
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Wrapper for CPU function
void runCPU() {
    int n = 10000000;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    
    vectorAddCPU(a, b, c, n);
    
    delete[] a;
    delete[] b;
    delete[] c;
}

// CUDA kernel for vector addition
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Wrapper for GPU function
void runGPU() {
    int n = 10000000;
    size_t bytes = n * sizeof(float);
    
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    
    // Initialize data
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    vectorAddGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    // Warm up
    runCPU();
    runGPU();
    
    // Measure CPU time
    float cpuTime = measureExecutionTime(runCPU);
    printf("CPU execution time: %.2f ms\n", cpuTime);
    
    // Measure GPU time
    float gpuTime = measureExecutionTime(runGPU);
    printf("GPU execution time: %.2f ms\n", gpuTime);
    
    // Calculate speedup
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    
    return 0;
}
```

### Performance Analysis Tools

- **NVIDIA Nsight**: Comprehensive profiling for CUDA applications
- **NVIDIA Visual Profiler**: Visual performance analysis
- **AMD Radeon GPU Profiler**: For AMD GPU performance analysis
- **Intel VTune Profiler**: For Intel accelerator profiling

## Resources for further learning and practice

### Books
- "CUDA by Example" by Jason Sanders and Edward Kandrot
- "Programming Massively Parallel Processors" by David Kirk and Wen-mei Hwu
- "Heterogeneous Computing with OpenCL" by Benedict Gaster et al.

### Online Courses
- NVIDIA's Deep Learning Institute courses
- Coursera's "Heterogeneous Parallel Programming"
- Udacity's "Intro to Parallel Programming"

### Repositories and Examples
- NVIDIA CUDA Samples: https://github.com/NVIDIA/cuda-samples
- AMD ROCm Examples: https://github.com/ROCm-Developer-Tools/HIP-Examples
- Intel oneAPI Samples: https://github.com/oneapi-src/oneAPI-samples

### Communities
- NVIDIA Developer Forums
- ROCm GitHub Issues and Discussions
- Stack Overflow tags: cuda, rocm, opencl, oneapi

## Key Terminology Definitions

- **Kernel**: A function that runs on the accelerator device
- **Thread**: The smallest unit of execution in parallel computing
- **Block/Workgroup**: A collection of threads that can cooperate
- **Grid**: A collection of blocks for executing a kernel
- **Host**: The CPU side of the application
- **Device**: The accelerator (GPU, FPGA, etc.)
- **Memory transfer**: Moving data between host and device memory
- **Occupancy**: How effectively the hardware resources are utilized

## Common Misconceptions Addressed

1. **"GPUs are always faster than CPUs"**: Not true for all workloads. GPUs excel at parallel tasks but may be slower for sequential operations.

2. **"More cores always mean better performance"**: Performance depends on the algorithm's parallelizability and memory access patterns.

3. **"Accelerated computing is only for scientific applications"**: Modern applications from gaming to web browsers use acceleration.

4. **"You need expensive hardware to learn"**: Cloud options and even older consumer GPUs are sufficient for learning.

5. **"CUDA is the only way to program GPUs"**: There are multiple frameworks including OpenCL, HIP, SYCL, and high-level libraries.

## Quick Recap

In this lesson, we've covered:
- Setting up development environments for accelerated computing
- Hardware options for learning and development
- Cloud-based alternatives for accessing accelerators
- Practical starter projects with source code
- Image processing and AI inference examples
- Performance measurement techniques
- Resources for continued learning

## Preview of Next Lesson

In Lesson 16, we'll explore "The Future of Accelerated Computing," including emerging hardware architectures, photonic computing, quantum acceleration, and neuromorphic computing. We'll also discuss career opportunities in this rapidly evolving field.