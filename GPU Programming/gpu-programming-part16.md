# Real-time Signal Processing on GPUs

*Welcome to the sixteenth installment of our GPU programming series! In this article, we'll explore real-time signal processing on GPUs, focusing on audio processing algorithms, image and video processing pipelines, FFT and convolution implementations, and streaming data processing.*

## Introduction to Signal Processing on GPUs

Signal processing involves analyzing, modifying, and synthesizing signals such as audio, images, and video. These tasks are often computationally intensive but highly parallelizable, making them excellent candidates for GPU acceleration. Real-time signal processing adds the constraint that processing must complete within strict time limits to maintain the illusion of continuity.

In this article, we'll explore how GPUs can accelerate various signal processing tasks, from audio effects to video encoding, and discuss techniques for handling streaming data efficiently.

## Audio Processing Algorithms

Audio processing involves manipulating sound signals, typically sampled at rates between 44.1kHz and 192kHz. While audio has lower data rates compared to images or video, complex audio effects and synthesis algorithms can still benefit significantly from GPU acceleration.

### Audio Effects

Many audio effects can be implemented efficiently on GPUs:

```cuda
// Example: Parallel reverb effect implementation
__global__ void reverb_kernel(
    float* input,          // Input audio buffer
    float* output,         // Output audio buffer
    float* delay_buffer,   // Delay buffer for reverb
    int buffer_length,     // Length of input/output buffers
    int delay_length,      // Length of delay buffer
    int num_taps,          // Number of delay taps
    float* tap_delays,     // Delay times for each tap
    float* tap_gains,      // Gain for each tap
    float feedback,        // Feedback coefficient
    int sample_rate        // Audio sample rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffer_length) return;
    
    // Start with dry signal
    float sample = input[i];
    output[i] = sample;
    
    // Add delayed samples from each tap
    for (int tap = 0; tap < num_taps; tap++) {
        int delay_samples = (int)(tap_delays[tap] * sample_rate);
        int delay_idx = (i + delay_length - delay_samples) % delay_length;
        
        // Add delayed sample to output
        output[i] += delay_buffer[delay_idx] * tap_gains[tap];
    }
    
    // Update delay buffer with feedback
    delay_buffer[i % delay_length] = sample + output[i] * feedback;
}
```

### Spectral Processing

Many audio effects operate in the frequency domain, requiring FFT and inverse FFT operations:

```cuda
// Example: Spectral noise reduction
__global__ void spectral_noise_reduction(
    cufftComplex* spectrum,    // FFT spectrum of audio frame
    float* noise_profile,      // Estimated noise floor
    int fft_size,              // Size of FFT
    float threshold,           // Noise gate threshold
    float reduction_factor     // Amount of reduction
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fft_size / 2 + 1) return; // Only process non-redundant bins
    
    // Compute magnitude
    float re = spectrum[i].x;
    float im = spectrum[i].y;
    float magnitude = sqrtf(re * re + im * im);
    float phase = atan2f(im, re);
    
    // Apply noise reduction
    float noise_level = noise_profile[i];
    if (magnitude < noise_level * threshold) {
        // Reduce magnitude if below threshold
        magnitude *= reduction_factor;
    }
    
    // Convert back to complex
    spectrum[i].x = magnitude * cosf(phase);
    spectrum[i].y = magnitude * sinf(phase);
    
    // Mirror for negative frequencies (if needed)
    if (i > 0 && i < fft_size / 2) {
        int mirror_idx = fft_size - i;
        spectrum[mirror_idx].x = spectrum[i].x;
        spectrum[mirror_idx].y = -spectrum[i].y; // Conjugate
    }
}
```

### Audio Synthesis

GPUs can generate complex audio through synthesis algorithms:

```cuda
// Example: Additive synthesis
__global__ void additive_synthesis_kernel(
    float* output,         // Output audio buffer
    int buffer_length,     // Length of output buffer
    float* frequencies,    // Array of oscillator frequencies
    float* amplitudes,     // Array of oscillator amplitudes
    float* phases,         // Array of oscillator phases
    int num_oscillators,   // Number of oscillators
    float sample_rate      // Audio sample rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffer_length) return;
    
    float time = (float)i / sample_rate;
    float sample = 0.0f;
    
    // Sum all oscillators
    for (int osc = 0; osc < num_oscillators; osc++) {
        float freq = frequencies[osc];
        float amp = amplitudes[osc];
        float phase = phases[osc];
        
        // Add sine oscillator
        sample += amp * sinf(2.0f * M_PI * freq * time + phase);
    }
    
    output[i] = sample;
}

// Example: FM synthesis
__global__ void fm_synthesis_kernel(
    float* output,         // Output audio buffer
    int buffer_length,     // Length of output buffer
    float carrier_freq,    // Carrier frequency
    float modulator_freq,  // Modulator frequency
    float mod_index,       // Modulation index
    float sample_rate      // Audio sample rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= buffer_length) return;
    
    float time = (float)i / sample_rate;
    
    // Compute modulator
    float modulator = sinf(2.0f * M_PI * modulator_freq * time);
    
    // Apply modulation to carrier frequency
    float instantaneous_freq = carrier_freq + mod_index * modulator_freq * modulator;
    
    // Integrate frequency to get phase
    float phase = 2.0f * M_PI * carrier_freq * time + mod_index * modulator_freq * cosf(2.0f * M_PI * modulator_freq * time);
    
    // Generate carrier with modulated phase
    output[i] = sinf(phase);
}
```

### Real-time Audio Processing Pipeline

A complete real-time audio processing system requires careful management of buffers and processing blocks:

```cpp
// Example: Real-time audio processing pipeline
class GPUAudioProcessor {
private:
    // Audio buffers
    float* d_input_buffer;
    float* d_output_buffer;
    float* d_processing_buffer;
    
    // FFT resources
    cufftHandle fft_plan;
    cufftComplex* d_spectrum;
    
    // Effect parameters
    float* d_reverb_buffer;
    float* d_eq_gains;
    
    // Stream for asynchronous processing
    cudaStream_t stream;
    
    // Buffer sizes
    int buffer_size;
    int fft_size;
    
 public:
    GPUAudioProcessor(int buffer_size, int fft_size) {
        this->buffer_size = buffer_size;
        this->fft_size = fft_size;
        
        // Allocate device memory
        cudaMalloc(&d_input_buffer, buffer_size * sizeof(float));
        cudaMalloc(&d_output_buffer, buffer_size * sizeof(float));
        cudaMalloc(&d_processing_buffer, buffer_size * sizeof(float));
        cudaMalloc(&d_spectrum, fft_size * sizeof(cufftComplex));
        cudaMalloc(&d_reverb_buffer, 48000 * 5 * sizeof(float)); // 5 seconds at 48kHz
        cudaMalloc(&d_eq_gains, 10 * sizeof(float)); // 10-band EQ
        
        // Create FFT plan
        cufftPlan1d(&fft_plan, fft_size, CUFFT_R2C, 1);
        
        // Create stream
        cudaStreamCreate(&stream);
    }
    
    ~GPUAudioProcessor() {
        // Free resources
        cudaFree(d_input_buffer);
        cudaFree(d_output_buffer);
        cudaFree(d_processing_buffer);
        cudaFree(d_spectrum);
        cudaFree(d_reverb_buffer);
        cudaFree(d_eq_gains);
        
        cufftDestroy(fft_plan);
        cudaStreamDestroy(stream);
    }
    
    void processBuffer(float* input, float* output) {
        // Copy input to device
        cudaMemcpyAsync(d_input_buffer, input, buffer_size * sizeof(float),
                       cudaMemcpyHostToDevice, stream);
        
        // Apply pre-processing effects
        int block_size = 256;
        int grid_size = (buffer_size + block_size - 1) / block_size;
        
        // Apply compressor
        compressor_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input_buffer, d_processing_buffer, buffer_size,
            threshold, ratio, attack, release);
        
        // Apply EQ (using FFT)
        cufftExecR2C(fft_plan, d_processing_buffer, d_spectrum);
        
        equalizer_kernel<<<(fft_size/2+1+255)/256, 256, 0, stream>>>(
            d_spectrum, d_eq_gains, fft_size);
        
        cufftExecC2R(fft_plan, d_spectrum, d_processing_buffer);
        
        // Apply reverb
        reverb_kernel<<<grid_size, block_size, 0, stream>>>(
            d_processing_buffer, d_output_buffer, d_reverb_buffer,
            buffer_size, 48000 * 5, num_taps, d_tap_delays, d_tap_gains,
            feedback, 48000);
        
        // Copy output back to host
        cudaMemcpyAsync(output, d_output_buffer, buffer_size * sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        
        // Wait for processing to complete
        cudaStreamSynchronize(stream);
    }
};
```

## Image and Video Processing Pipelines

Image and video processing involve manipulating 2D or 3D arrays of pixel data. These operations are highly parallelizable and can benefit greatly from GPU acceleration.

### Basic Image Filters

Simple image filters apply a transformation to each pixel:

```cuda
// Example: Brightness and contrast adjustment
__global__ void brightness_contrast_kernel(
    uchar4* input,
    uchar4* output,
    int width,
    int height,
    float brightness, // -1.0 to 1.0
    float contrast    // 0.0 to 2.0
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    uchar4 pixel = input[idx];
    
    // Convert to float
    float r = pixel.x / 255.0f;
    float g = pixel.y / 255.0f;
    float b = pixel.z / 255.0f;
    
    // Apply brightness
    r += brightness;
    g += brightness;
    b += brightness;
    
    // Apply contrast
    float factor = (259.0f * (contrast + 255.0f)) / (255.0f * (259.0f - contrast));
    r = factor * (r - 0.5f) + 0.5f;
    g = factor * (g - 0.5f) + 0.5f;
    b = factor * (b - 0.5f) + 0.5f;
    
    // Clamp and convert back to uchar
    output[idx].x = (unsigned char)(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
    output[idx].y = (unsigned char)(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    output[idx].z = (unsigned char)(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
    output[idx].w = pixel.w; // Preserve alpha
}
```

### Convolution-Based Filters

Many image filters use convolution with a kernel:

```cuda
// Example: 2D convolution for image filtering
__global__ void convolution_kernel(
    uchar4* input,
    uchar4* output,
    float* filter,
    int filter_width,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = filter_width / 2;
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    float filter_sum = 0.0f;
    
    // Apply filter
    for (int fy = -radius; fy <= radius; fy++) {
        for (int fx = -radius; fx <= radius; fx++) {
            int nx = x + fx;
            int ny = y + fy;
            
            // Handle boundary conditions (clamp to edge)
            nx = max(0, min(width - 1, nx));
            ny = max(0, min(height - 1, ny));
            
            // Get pixel and filter value
            uchar4 pixel = input[ny * width + nx];
            float filter_val = filter[(fy + radius) * filter_width + (fx + radius)];
            
            // Accumulate weighted sum
            r_sum += pixel.x * filter_val;
            g_sum += pixel.y * filter_val;
            b_sum += pixel.z * filter_val;
            filter_sum += filter_val;
        }
    }
    
    // Normalize if necessary
    if (filter_sum != 0.0f) {
        r_sum /= filter_sum;
        g_sum /= filter_sum;
        b_sum /= filter_sum;
    }
    
    // Clamp and write output
    int idx = y * width + x;
    output[idx].x = (unsigned char)(fminf(fmaxf(r_sum, 0.0f), 255.0f));
    output[idx].y = (unsigned char)(fminf(fmaxf(g_sum, 0.0f), 255.0f));
    output[idx].z = (unsigned char)(fminf(fmaxf(b_sum, 0.0f), 255.0f));
    output[idx].w = input[idx].w; // Preserve alpha
}
```

### Shared Memory Optimization

Using shared memory can significantly improve performance for convolution operations:

```cuda
// Example: Optimized convolution using shared memory
__global__ void convolution_shared_kernel(
    uchar4* input,
    uchar4* output,
    float* filter,
    int filter_width,
    int width,
    int height
) {
    extern __shared__ uchar4 shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int radius = filter_width / 2;
    int block_width = blockDim.x + 2 * radius;
    
    // Load input to shared memory (including halo region)
    for (int dy = -radius; dy <= radius + blockDim.y; dy += blockDim.y) {
        for (int dx = -radius; dx <= radius + blockDim.x; dx += blockDim.x) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Clamp to image boundaries
            nx = max(0, min(width - 1, nx));
            ny = max(0, min(height - 1, ny));
            
            // Compute shared memory index
            int sx = tx + dx + radius;
            int sy = ty + dy + radius;
            
            if (sx >= 0 && sx < block_width && sy >= 0 && sy < block_width) {
                shared_input[sy * block_width + sx] = input[ny * width + nx];
            }
        }
    }
    
    __syncthreads();
    
    // Only process valid output pixels
    if (x < width && y < height) {
        float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
        float filter_sum = 0.0f;
        
        // Apply filter
        for (int fy = -radius; fy <= radius; fy++) {
            for (int fx = -radius; fx <= radius; fx++) {
                // Get pixel and filter value from shared memory
                int sx = tx + fx + radius;
                int sy = ty + fy + radius;
                uchar4 pixel = shared_input[sy * block_width + sx];
                float filter_val = filter[(fy + radius) * filter_width + (fx + radius)];
                
                // Accumulate weighted sum
                r_sum += pixel.x * filter_val;
                g_sum += pixel.y * filter_val;
                b_sum += pixel.z * filter_val;
                filter_sum += filter_val;
            }
        }
        
        // Normalize if necessary
        if (filter_sum != 0.0f) {
            r_sum /= filter_sum;
            g_sum /= filter_sum;
            b_sum /= filter_sum;
        }
        
        // Write output
        int idx = y * width + x;
        output[idx].x = (unsigned char)(fminf(fmaxf(r_sum, 0.0f), 255.0f));
        output[idx].y = (unsigned char)(fminf(fmaxf(g_sum, 0.0f), 255.0f));
        output[idx].z = (unsigned char)(fminf(fmaxf(b_sum, 0.0f), 255.0f));
        output[idx].w = input[idx].w; // Preserve alpha
    }
}
```

### Video Processing Pipeline

Video processing involves applying operations to a sequence of frames, often with temporal dependencies:

```cpp
// Example: Video processing pipeline
class GPUVideoProcessor {
private:
    // Frame buffers
    uchar4* d_input_frame;
    uchar4* d_output_frame;
    uchar4* d_previous_frame;
    
    // Processing buffers
    float* d_motion_vectors;
    float* d_filter_kernels;
    
    // Frame dimensions
    int width;
    int height;
    
    // CUDA resources
    cudaStream_t stream;
    
public:
    GPUVideoProcessor(int width, int height) {
        this->width = width;
        this->height = height;
        
        // Allocate device memory
        cudaMalloc(&d_input_frame, width * height * sizeof(uchar4));
        cudaMalloc(&d_output_frame, width * height * sizeof(uchar4));
        cudaMalloc(&d_previous_frame, width * height * sizeof(uchar4));
        cudaMalloc(&d_motion_vectors, width * height * 2 * sizeof(float));
        cudaMalloc(&d_filter_kernels, 5 * 5 * 3 * sizeof(float)); // 3 5x5 kernels
        
        // Create stream
        cudaStreamCreate(&stream);
        
        // Initialize filter kernels
        initializeFilters();
    }
    
    ~GPUVideoProcessor() {
        cudaFree(d_input_frame);
        cudaFree(d_output_frame);
        cudaFree(d_previous_frame);
        cudaFree(d_motion_vectors);
        cudaFree(d_filter_kernels);
        
        cudaStreamDestroy(stream);
    }
    
    void processFrame(uchar4* input_frame, uchar4* output_frame) {
        // Copy input frame to device
        cudaMemcpyAsync(d_input_frame, input_frame, width * height * sizeof(uchar4),
                       cudaMemcpyHostToDevice, stream);
        
        // Set up kernel launch parameters
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        // Step 1: Motion estimation
        motion_estimation_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input_frame, d_previous_frame, d_motion_vectors, width, height);
        
        // Step 2: Temporal noise reduction
        temporal_denoise_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input_frame, d_previous_frame, d_motion_vectors, d_output_frame,
            width, height, noise_threshold);
        
        // Step 3: Spatial filtering (e.g., sharpening)
        int shared_mem_size = (block_size.x + 4) * (block_size.y + 4) * sizeof(uchar4);
        convolution_shared_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
            d_output_frame, d_output_frame, d_filter_kernels, 5, width, height);
        
        // Step 4: Color correction
        color_correction_kernel<<<grid_size, block_size, 0, stream>>>(
            d_output_frame, d_output_frame, width, height,
            brightness, contrast, saturation);
        
        // Copy output frame back to host
        cudaMemcpyAsync(output_frame, d_output_frame, width * height * sizeof(uchar4),
                       cudaMemcpyDeviceToHost, stream);
        
        // Store current frame as previous frame for next iteration
        uchar4* temp = d_previous_frame;
        d_previous_frame = d_input_frame;
        d_input_frame = temp;
        
        // Wait for processing to complete
        cudaStreamSynchronize(stream);
    }
};
```

## FFT and Convolution Implementations

The Fast Fourier Transform (FFT) is a fundamental algorithm in signal processing that converts signals between time/space and frequency domains.

### FFT Implementation

While libraries like cuFFT provide optimized implementations, understanding the basics is valuable:

```cuda
// Example: Simple 1D FFT implementation (Cooley-Tukey algorithm)
__global__ void fft_kernel(
    float2* input,     // Complex input array
    float2* output,    // Complex output array
    int n,             // Size of FFT (power of 2)
    int step           // Current step in the recursion
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;
    
    int even_idx = idx * 2;
    int odd_idx = even_idx + 1;
    
    // Twiddle factor
    float angle = -2.0f * M_PI * idx / n;
    float2 twiddle = make_float2(cosf(angle), sinf(angle));
    
    // Get even and odd elements
    float2 even = input[even_idx];
    float2 odd = input[odd_idx];
    
    // Complex multiplication: odd * twiddle
    float2 odd_twiddle;
    odd_twiddle.x = odd.x * twiddle.x - odd.y * twiddle.y;
    odd_twiddle.y = odd.x * twiddle.y + odd.y * twiddle.x;
    
    // Butterfly operation
    output[idx] = make_float2(even.x + odd_twiddle.x, even.y + odd_twiddle.y);
    output[idx + n/2] = make_float2(even.x - odd_twiddle.x, even.y - odd_twiddle.y);
}

// Host function to perform FFT
void perform_fft(float2* d_data, int n) {
    float2 *d_temp;
    cudaMalloc(&d_temp, n * sizeof(float2));
    
    // Initial bit-reversal permutation
    bit_reversal_kernel<<<(n+255)/256, 256>>>(d_data, d_temp, n);
    
    // FFT stages
    float2 *d_in = d_temp;
    float2 *d_out = d_data;
    
    for (int step = 2; step <= n; step *= 2) {
        fft_kernel<<<(n/2+255)/256, 256>>>(d_in, d_out, n, step);
        
        // Swap input and output for next stage
        float2 *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }
    
    // If result ended up in d_temp, copy back to d_data
    if (d_in == d_temp) {
        cudaMemcpy(d_data, d_temp, n * sizeof(float2), cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_temp);
}
```

### Fast Convolution using FFT

Convolution can be accelerated using the FFT, especially for large kernels:

```cuda
// Example: Fast convolution using FFT
void fft_convolution(
    cufftHandle fft_plan,
    cufftHandle ifft_plan,
    float* d_input,
    float* d_kernel,
    float* d_output,
    cufftComplex* d_freq_domain,
    cufftComplex* d_kernel_freq,
    int signal_length,
    int kernel_length
) {
    // Zero-pad input and kernel to avoid circular convolution effects
    int padded_length = signal_length + kernel_length - 1;
    
    // Ensure padded_length is a power of 2 for efficient FFT
    padded_length = next_power_of_2(padded_length);
    
    // Perform forward FFT on input signal
    cufftExecR2C(fft_plan, d_input, d_freq_domain);
    
    // Perform forward FFT on kernel
    cufftExecR2C(fft_plan, d_kernel, d_kernel_freq);
    
    // Multiply in frequency domain
    int complex_elements = padded_length / 2 + 1;
    complex_multiply_kernel<<<(complex_elements+255)/256, 256>>>(
        d_freq_domain, d_kernel_freq, d_freq_domain, complex_elements);
    
    // Perform inverse FFT
    cufftExecC2R(ifft_plan, d_freq_domain, d_output);
    
    // Scale the output (CUFFT doesn't normalize)
    scale_kernel<<<(padded_length+255)/256, 256>>>(
        d_output, d_output, 1.0f / padded_length, padded_length);
}

__global__ void complex_multiply_kernel(
    cufftComplex* a,
    cufftComplex* b,
    cufftComplex* c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float a_re = a[i].x;
    float a_im = a[i].y;
    float b_re = b[i].x;
    float b_im = b[i].y;
    
    c[i].x = a_re * b_re - a_im * b_im;
    c[i].y = a_re * b_im + a_im * b_re;
}
```

### 2D FFT for Image Processing

For image processing, 2D FFTs are commonly used:

```cpp
// Example: 2D FFT-based image filtering
void fft_image_filter(
    uchar4* d_input,
    uchar4* d_output,
    float* d_filter,
    int width,
    int height
) {
    // Create 2D FFT plans
    cufftHandle fft_plan, ifft_plan;
    cufftPlan2d(&fft_plan, height, width, CUFFT_R2C);
    cufftPlan2d(&ifft_plan, height, width, CUFFT_C2R);
    
    // Allocate memory for processing
    int complex_width = width / 2 + 1;
    float* d_input_channels[3];
    cufftComplex* d_freq_domain[3];
    cufftComplex* d_filter_freq;
    
    for (int i = 0; i < 3; i++) {
        cudaMalloc(&d_input_channels[i], width * height * sizeof(float));
        cudaMalloc(&d_freq_domain[i], height * complex_width * sizeof(cufftComplex));
    }
    cudaMalloc(&d_filter_freq, height * complex_width * sizeof(cufftComplex));
    
    // Extract RGB channels
    extract_channels_kernel<<<(width*height+255)/256, 256>>>(
        d_input, d_input_channels[0], d_input_channels[1], d_input_channels[2],
        width, height);
    
    // Prepare filter (zero-padded and shifted)
    prepare_filter_kernel<<<(width*height+255)/256, 256>>>(
        d_filter, d_input_channels[0], width, height);
    
    // Transform filter to frequency domain
    cufftExecR2C(fft_plan, d_input_channels[0], d_filter_freq);
    
    // Process each channel
    for (int i = 0; i < 3; i++) {
        // Forward FFT
        cufftExecR2C(fft_plan, d_input_channels[i], d_freq_domain[i]);
        
        // Apply filter in frequency domain
        complex_multiply_kernel<<<(height*complex_width+255)/256, 256>>>(
            d_freq_domain[i], d_filter_freq, d_freq_domain[i], height * complex_width);
        
        // Inverse FFT
        cufftExecC2R(ifft_plan, d_freq_domain[i], d_input_channels[i]);
        
        // Scale (CUFFT doesn't normalize)
        scale_kernel<<<(width*height+255)/256, 256>>>(
            d_input_channels[i], d_input_channels[i], 1.0f / (width * height), width * height);
    }
    
    // Combine channels back to output image
    combine_channels_kernel<<<(width*height+255)/256, 256>>>(
        d_input_channels[0], d_input_channels[1], d_input_channels[2], d_output,
        width, height);
    
    // Clean up
    for (int i = 0; i < 3; i++) {
        cudaFree(d_input_channels[i]);
        cudaFree(d_freq_domain[i]);
    }
    cudaFree(d_filter_freq);
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
}
```

## Streaming Data Processing

Processing streaming data requires efficient handling of continuous data flows with minimal latency.

### Circular Buffers

Circular buffers are essential for streaming data processing:

```cuda
// Example: Circular buffer implementation for streaming data
class GPUCircularBuffer {
private:
    float* d_buffer;
    int capacity;
    int head;
    int tail;
    int count;
    
public:
    GPUCircularBuffer(int capacity) {
        this->capacity = capacity;
        cudaMalloc(&d_buffer, capacity * sizeof(float));
        head = 0;
        tail = 0;
        count = 0;
    }
    
    ~GPUCircularBuffer() {
        cudaFree(d_buffer);
    }
    
    bool push(float* data, int size) {
        if (count + size > capacity) return false;
        
        // Copy data to buffer
        if (head + size <= capacity) {
            // Continuous copy
            cudaMemcpy(d_buffer + head, data, size * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            // Split copy
            int first_part = capacity - head;
            int second_part = size - first_part;
            
            cudaMemcpy(d_buffer + head, data, first_part * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_buffer, data + first_part, second_part * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        head = (head + size) % capacity;
        count += size;
        return true;
    }
    
    bool pop(float* data, int size) {
        if (count < size) return false;
        
        // Copy data from buffer
        if (tail + size <= capacity) {
            // Continuous copy
            cudaMemcpy(data, d_buffer + tail, size * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            // Split copy
            int first_part = capacity - tail;
            int second_part = size - first_part;
            
            cudaMemcpy(data, d_buffer + tail, first_part * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(data + first_part, d_buffer, second_part * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        tail = (tail + size) % capacity;
        count -= size;
        return true;
    }
    
    // Process data in-place without removing it
    void process(int size, cudaStream_t stream, void (*kernel)(float*, int, cudaStream_t)) {
        if (count < size) return;
        
        if (tail + size <= capacity) {
            // Continuous processing
            kernel(d_buffer + tail, size, stream);
        } else {
            // Split processing (more complex, may need temporary buffer)
            // ...
        }
    }
};
```

### Streaming Signal Processing

Processing streaming signals often involves overlapping windows:

```cpp
// Example: Streaming FFT processing with overlapping windows
class StreamingFFTProcessor {
private:
    cufftHandle fft_plan;
    float* d_window;
    float* d_buffer;
    cufftComplex* d_spectrum;
    int buffer_size;
    int window_size;
    int hop_size;
    int position;
    
public:
    StreamingFFTProcessor(int window_size, int hop_size) {
        this->window_size = window_size;
        this->hop_size = hop_size;
        this->buffer_size = window_size * 2; // Extra space for overlap
        this->position = 0;
        
        cudaMalloc(&d_buffer, buffer_size * sizeof(float));
        cudaMalloc(&d_window, window_size * sizeof(float));
        cudaMalloc(&d_spectrum, (window_size / 2 + 1) * sizeof(cufftComplex));
        
        // Create FFT plan
        cufftPlan1d(&fft_plan, window_size, CUFFT_R2C, 1);
        
        // Initialize Hann window
        float* h_window = new float[window_size];
        for (int i = 0; i < window_size; i++) {
            h_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (window_size - 1)));
        }
        cudaMemcpy(d_window, h_window, window_size * sizeof(float), cudaMemcpyHostToDevice);
        delete[] h_window;
    }
    
    ~StreamingFFTProcessor() {
        cudaFree(d_buffer);
        cudaFree(d_window);
        cudaFree(d_spectrum);
        cufftDestroy(fft_plan);
    }
    
    void processChunk(float* input, int input_size, void (*callback)(cufftComplex*, int)) {
        // Copy input to buffer
        cudaMemcpy(d_buffer + position, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
        position += input_size;
        
        // Process complete windows
        while (position >= window_size) {
            // Apply window function
            apply_window_kernel<<<(window_size+255)/256, 256>>>(
                d_buffer, d_window, window_size);
            
            // Perform FFT
            cufftExecR2C(fft_plan, d_buffer, d_spectrum);
            
            // Call callback with spectrum
            callback(d_spectrum, window_size / 2 + 1);
            
            // Shift buffer by hop size
            if (hop_size < position) {
                cudaMemcpy(d_buffer, d_buffer + hop_size, (position - hop_size) * sizeof(float), cudaMemcpyDeviceToDevice);
                position -= hop_size;
            } else {
                position = 0;
                break;
            }
        }
    }
};

__global__ void apply_window_kernel(float* buffer, float* window, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        buffer[i] *= window[i];
    }
}
```

### Real-time Video Streaming

Real-time video streaming requires efficient handling of frame buffers and processing pipelines:

```cpp
// Example: Real-time video streaming processor
class VideoStreamProcessor {
private:
    // Frame buffers (triple buffering)
    uchar4* d_frame_buffers[3];
    int current_write_buffer;
    int current_process_buffer;
    int current_read_buffer;
    
    // Frame dimensions
    int width;
    int height;
    
    // Processing resources
    cudaStream_t streams[2]; // One for copy, one for processing
    
    // Synchronization
    std::mutex buffer_mutex;
    std::condition_variable buffer_cv;
    bool frame_ready;
    bool processing_done;
    
public:
    VideoStreamProcessor(int width, int height) {
        this->width = width;
        this->height = height;
        
        // Allocate frame buffers
        for (int i = 0; i < 3; i++) {
            cudaMalloc(&d_frame_buffers[i], width * height * sizeof(uchar4));
        }
        
        // Create streams
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);
        
        // Initialize buffer indices
        current_write_buffer = 0;
        current_process_buffer = 1;
        current_read_buffer = 2;
        
        frame_ready = false;
        processing_done = true;
    }
    
    ~VideoStreamProcessor() {
        for (int i = 0; i < 3; i++) {
            cudaFree(d_frame_buffers[i]);
        }
        
        cudaStreamDestroy(streams[0]);
        cudaStreamDestroy(streams[1]);
    }
    
    void writeFrame(uchar4* frame) {
        // Wait until previous write is processed
        std::unique_lock<std::mutex> lock(buffer_mutex);
        buffer_cv.wait(lock, [this]{ return processing_done; });
        
        // Copy frame to current write buffer
        cudaMemcpyAsync(d_frame_buffers[current_write_buffer], frame,
                       width * height * sizeof(uchar4),
                       cudaMemcpyHostToDevice, streams[0]);
        
        // Rotate buffers
        int temp = current_process_buffer;
        current_process_buffer = current_write_buffer;
        current_write_buffer = temp;
        
        // Signal that frame is ready for processing
        frame_ready = true;
        processing_done = false;
        buffer_cv.notify_one();
    }
    
    void processFrames() {
        while (true) {
            // Wait for a frame to be ready
            std::unique_lock<std::mutex> lock(buffer_mutex);
            buffer_cv.wait(lock, [this]{ return frame_ready; });
            
            // Process the frame
            dim3 block_size(16, 16);
            dim3 grid_size((width + block_size.x - 1) / block_size.x,
                          (height + block_size.y - 1) / block_size.y);
            
            // Apply processing kernels
            process_frame_kernel<<<grid_size, block_size, 0, streams[1]>>>(
                d_frame_buffers[current_process_buffer],
                d_frame_buffers[current_read_buffer],
                width, height);
            
            // Rotate read buffer
            int temp = current_read_buffer;
            current_read_buffer = current_process_buffer;
            current_process_buffer = temp;
            
            // Signal that processing is done
            frame_ready = false;
            processing_done = true;
            buffer_cv.notify_one();
        }
    }
    
    void readFrame(uchar4* frame) {
        // Copy processed frame back to host
        cudaMemcpy(frame, d_frame_buffers[current_read_buffer],
                  width * height * sizeof(uchar4),
                  cudaMemcpyDeviceToHost);
    }
};
```

## Conclusion

GPU-accelerated signal processing enables real-time manipulation of audio, images, and video with performance that would be impossible on CPUs alone. By leveraging the massive parallelism of GPUs, developers can implement complex processing pipelines that operate on high-resolution, high-frame-rate data streams with minimal latency.

Key takeaways from this article include:

1. **Audio Processing**: GPUs can accelerate audio effects, spectral processing, and synthesis algorithms
2. **Image and Video Processing**: Filters, convolutions, and complex processing pipelines benefit greatly from GPU parallelism
3. **FFT and Convolution**: Efficient implementations of these fundamental operations are critical for many signal processing tasks
4. **Streaming Data**: Techniques like circular buffers and overlapping windows enable efficient processing of continuous data streams

In our next article, we'll explore cryptography and blockchain on GPUs, focusing on parallel cryptographic algorithms and mining considerations.

## Exercises for Practice

1. **Audio Processor**: Implement a real-time audio effect processor that applies multiple effects (e.g., reverb, EQ, compression) to streaming audio.

2. **Image Filter Library**: Create a library of GPU-accelerated image filters with a unified interface, comparing performance against CPU implementations.

3. **Video Processing Pipeline**: Implement a video processing pipeline that applies temporal and spatial filters to a video stream in real-time.

4. **FFT Visualization**: Build a real-time audio spectrum analyzer that uses GPU-accelerated FFT to visualize audio frequencies.

5. **Streaming Data Challenge**: Implement a streaming data processor that can handle continuous sensor data, applying filters and detecting patterns in real-time.

## Further Resources

- [NVIDIA NPP (NVIDIA Performance Primitives)](https://developer.nvidia.com/npp) - GPU-accelerated image and signal processing functions
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/index.html) - NVIDIA's FFT library
- [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) - Hardware-accelerated video encoding and decoding
- [NVIDIA Audio SDK](https://developer.nvidia.com/audio-sdk) - GPU-accelerated audio processing
- [OpenCV GPU Module](https://docs.opencv.org/master/d0/d05/group__gpu.html) - GPU-accelerated computer vision functions