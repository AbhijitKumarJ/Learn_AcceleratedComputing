# Part 6: Real-time Inference on NPUs

As we continue our journey into Neural Processing Unit (NPU) programming, this sixth installment focuses on a critical aspect of deployment: achieving and optimizing real-time inference. In many applications—from augmented reality to autonomous systems—the ability to process neural network inferences with minimal latency and maximum efficiency is paramount. Let's dive into the key considerations and techniques for mastering real-time inference on NPUs.

## Latency vs. Throughput Considerations

When optimizing for real-time inference, understanding the fundamental tradeoff between latency and throughput is essential.

### Understanding Latency

Latency refers to the time taken from input to output for a single inference request. For real-time applications, this metric is often the most critical. Latency components include:

- **Data Transfer Latency**: Time to move data from host memory to NPU memory
- **Computation Latency**: Time for the actual matrix operations on the NPU
- **Post-processing Latency**: Time for any result processing after inference
- **Framework Overhead**: Additional time from software stack operations

Real-time applications typically have strict latency budgets:
- AR/VR: 10-20ms for responsive experiences
- Autonomous driving: 50-100ms for safe operation
- Speech recognition: 200-300ms for natural interaction

### Understanding Throughput

Throughput measures how many inferences can be processed per unit of time, usually expressed as inferences per second (IPS). Throughput optimization focuses on maximizing the total processing capacity of the NPU, which is important for:

- Server-based applications handling multiple requests
- Batch processing of videos or images
- High-density edge applications (e.g., multi-camera surveillance)

### Balancing the Tradeoff

The key insight: optimizing for latency often sacrifices throughput, and vice versa. For example:

- Batching increases throughput but adds latency to individual requests
- Running at lower precision improves both metrics but may reduce accuracy
- Aggressive operator fusion improves latency but may reduce NPU utilization

**Implementation Example:**
```python
# Latency-focused configuration
inference_config = {
    "batch_size": 1,          # No batching for lowest latency
    "priority": "high",       # Prioritize this workload
    "power_mode": "performance", # Maximum clock frequency
    "io_binding": True        # Direct memory access
}

# Throughput-focused configuration
batch_inference_config = {
    "batch_size": 8,          # Process multiple inputs together
    "priority": "normal",     # Standard priority
    "power_mode": "efficiency", # Balance power and performance
    "io_binding": False       # Standard memory handling
}
```

Real-time applications typically prioritize consistent, predictable latency over maximum throughput. Some NPUs offer specific modes or governors that prioritize latency consistency over peak performance.

## Pipeline Optimization

Neural network inference can be broken down into stages that form a processing pipeline. Optimizing this pipeline is crucial for real-time performance.

### Pipeline Stages

A typical inference pipeline includes:

1. **Pre-processing**: Resize images, normalize values, format transformations
2. **Input Preparation**: Transfer data to NPU memory, setup execution buffers
3. **Inference Execution**: Run the model on the NPU
4. **Post-processing**: Process raw outputs (e.g., apply NMS for object detection)
5. **Result Handling**: Act on inference results (display, store, or trigger actions)

### Serial vs. Pipelined Execution

In serial processing, each inference waits for the entire pipeline to complete before starting the next one. Pipelined execution overlaps stages from different inferences:

```
Serial:    [Pre→Inf→Post][Pre→Inf→Post][Pre→Inf→Post]
Pipelined: [Pre→Inf→Post]
             [Pre→Inf→Post]
                [Pre→Inf→Post]
```

This parallelization significantly improves throughput and can reduce end-to-end latency for streams of inputs.

### Implementation Techniques

1. **Double Buffering**: Use alternating input/output buffers to overlap data transfer with computation
   ```cpp
   // Double buffering example (pseudo-code)
   Buffer inputBuffers[2];
   Buffer outputBuffers[2];
   
   int activeBuffer = 0;
   while (hasMoreFrames()) {
       // Transfer next input while processing current one
       prepareInputAsync(nextFrame, inputBuffers[1-activeBuffer]);
       
       // Process current frame
       npuInference(inputBuffers[activeBuffer], outputBuffers[activeBuffer]);
       
       // Process previous output while NPU works on current frame
       if (frameIndex > 0) {
           postProcess(outputBuffers[1-activeBuffer]);
       }
       
       waitForCompletion();
       activeBuffer = 1 - activeBuffer;
   }
   ```

2. **Stage Balancing**: Ensure each pipeline stage takes approximately the same time
   - Move pre/post-processing to CPU if NPU is the bottleneck
   - Optimize the slowest stage aggressively
   - Consider hardware-accelerated pre/post-processing when available

3. **Asynchronous API Usage**: Most NPU frameworks offer async execution modes
   ```python
   # Pseudo-code for asynchronous execution
   input_queue = Queue()
   output_queue = Queue()
   
   def preprocessing_thread():
       while True:
           frame = capture_frame()
           processed = preprocess(frame)
           input_queue.put(processed)
   
   def inference_thread():
       while True:
           input_tensor = input_queue.get()
           result = npu.execute_async(input_tensor)
           output_queue.put(result)
   
   def postprocessing_thread():
       while True:
           result = output_queue.get()
           detections = postprocess(result)
           display(detections)
   
   # Start all threads
   threading.Thread(target=preprocessing_thread).start()
   threading.Thread(target=inference_thread).start()
   threading.Thread(target=postprocessing_thread).start()
   ```

4. **Model Partitioning**: Split large models into smaller sections that can be pipelined
   - Early layers → Middle layers → Final layers
   - This can be particularly effective for very deep networks

## Concurrent Execution Strategies

Modern NPUs often support multiple concurrent workloads, enabling sophisticated execution strategies.

### Workload Types in Real-time Systems

Real-time systems typically have a mix of:
- **Critical Path Tasks**: Must complete within strict deadlines (obstacle detection)
- **Background Tasks**: Can tolerate delays (map building, environment modeling)
- **Periodic Tasks**: Run at regular intervals (sensor fusion, state estimation)
- **Event-Triggered Tasks**: Run in response to specific events (anomaly detection)

### Parallel Model Execution

Running multiple models concurrently offers several advantages:
- Better utilization of NPU resources
- Lower overall latency than sequential execution
- Ability to handle multiple input streams

**Implementation Approaches:**

1. **Spatial Partitioning**: Dedicate different NPU cores or tiles to different models
   ```cpp
   // NPU spatial partitioning (vendor-specific API example)
   NPUPartition detection_partition = npu.createPartition({
       "cores": [0, 1],  // Use first two cores
       "memory": "1GB"   // Allocate 1GB to this partition
   });
   
   NPUPartition classification_partition = npu.createPartition({
       "cores": [2, 3],  // Use other two cores
       "memory": "1GB"   // Allocate remaining memory
   });
   
   // Load models to specific partitions
   detection_model.loadToPartition(detection_partition);
   classification_model.loadToPartition(classification_partition);
   
   // Run concurrently
   auto detection_future = detection_model.executeAsync(camera_frame);
   auto classification_future = classification_model.executeAsync(roi_frame);
   ```

2. **Time-Division Multiplexing**: Schedule different models at different time slices
   - Useful when spatial partitioning isn't supported
   - Requires careful priority management

3. **Priority-Based Execution**: Assign priorities to different inference workloads
   ```python
   # Priority-based execution example
   critical_model.set_execution_priority(Priority.HIGH)
   background_model.set_execution_priority(Priority.LOW)
   
   # The NPU scheduler will preempt lower priority work
   # when high priority inference is requested
   critical_result = critical_model.execute(sensor_data)  # Runs immediately
   background_model.execute_async(background_data)  # Runs when resources available
   ```

### Task Scheduling Strategies

1. **Deadline-Aware Scheduling**: Prioritize tasks based on their deadlines
   - Earliest Deadline First (EDF) algorithm
   - Rate-Monotonic Scheduling (RMS) for periodic tasks

2. **Preemptive vs. Cooperative Multitasking**:
   - Preemptive: High-priority tasks can interrupt ongoing lower-priority tasks
   - Cooperative: Tasks must yield control explicitly
   - Most NPUs support some form of preemption for critical tasks

3. **Work Stealing**: Idle processing elements can "steal" work from busy ones
   - Implemented in some advanced NPU runtimes
   - Improves load balancing across the NPU

## Handling Dynamic Input Sizes

Real-world inputs rarely have fixed dimensions, creating challenges for optimized NPU execution.

### Challenges of Dynamic Inputs

NPUs typically perform best with fixed-shape inputs because:
- Compile-time optimizations target specific tensor dimensions
- Memory can be pre-allocated efficiently
- Data layout can be optimized for specific dimensions

However, real-time applications often have:
- Varying image resolutions from cameras
- Different sequence lengths in speech or text
- Dynamic batch sizes based on current load

### Resolution Adaptation Strategies

1. **Input Resizing**: Simplest approach, but may lose information
   ```python
   def preprocess_dynamic_input(image):
       # Resize all inputs to the model's expected dimensions
       resized = cv2.resize(image, (model_width, model_height))
       return normalize(resized)
   ```

2. **Multiple Compiled Models**: Prepare several versions for common sizes
   ```python
   # Prepare models for common resolutions
   models = {
       (640, 480): load_compiled_model("model_640x480"),
       (1280, 720): load_compiled_model("model_1280x720"),
       (1920, 1080): load_compiled_model("model_1920x1080")
   }
   
   def get_best_model(width, height):
       # Find closest model size
       closest_size = min(models.keys(), 
                         key=lambda s: abs(s[0]-width) + abs(s[1]-height))
       return models[closest_size]
   ```

3. **Dynamic Shape Support**: Some NPUs support true dynamic shapes
   ```python
   # Using TensorRT with dynamic shapes
   import tensorrt as trt
   
   builder = trt.Builder(logger)
   network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
   
   # Define input with dynamic dimensions
   input_tensor = network.add_input("input", trt.float32, (-1, 3, -1, -1))
   
   # Specify optimization profiles
   profile = builder.create_optimization_profile()
   profile.set_shape("input", 
                    min=(1, 3, 240, 320),    # Minimum dimensions
                    opt=(1, 3, 480, 640),    # Optimal dimensions
                    max=(1, 3, 1080, 1920))  # Maximum dimensions
   
   config = builder.create_builder_config()
   config.add_optimization_profile(profile)
   ```

4. **Tiling Approaches**: Process large inputs in tiles of fixed size
   - Useful for very large images or when model has receptive field limitations
   - Requires post-processing to stitch results

### Sequence Length Adaptation

For variable-length inputs (text, audio, time series):

1. **Padding**: Add zeros or special tokens to reach fixed length
   - Simple but computationally wasteful

2. **Bucketing**: Group similar-length inputs together
   ```python
   # Bucketing example for text processing
   buckets = [16, 32, 64, 128, 256]
   
   def get_bucket_size(sequence_length):
       return min(filter(lambda x: x >= sequence_length, buckets))
   
   def preprocess_sequence(sequence):
       bucket_size = get_bucket_size(len(sequence))
       return pad_sequence(sequence, bucket_size)
   ```

3. **Streaming Models**: Process input incrementally
   - Stateful models that maintain hidden states between chunks
   - Useful for audio and video processing

## Profiling and Benchmarking

Effective optimization requires accurate measurement and identification of bottlenecks.

### Key Performance Metrics

1. **Latency Metrics**:
   - End-to-end latency: Total time from input to result
   - Inference-only latency: NPU computation time
   - Tail latency: 95th or 99th percentile latency (important for real-time guarantees)
   - Initialization time: One-time cost to prepare the NPU

2. **Throughput Metrics**:
   - Inferences per second (IPS)
   - Frames per second (FPS) for video applications
   - Tokens per second for NLP applications

3. **Resource Utilization**:
   - NPU utilization percentage
   - Memory bandwidth usage
   - Power consumption (critical for battery-powered devices)
   - Thermal impact (important for sustained performance)

### Profiling Tools and Techniques

1. **Vendor-Specific Profilers**:
   - Apple's Core ML Tools and Instruments
   - Qualcomm AI Model Efficiency Toolkit
   - Intel's OpenVINO Deep Learning Workbench
   - ARM's Ethos NPU tools

2. **Framework Profilers**:
   - TensorFlow Lite Profiler
   - ONNX Runtime Profiling API
   - PyTorch Mobile Performance Benchmarks

3. **Layer-by-Layer Analysis**:
   ```python
   # Example of per-layer profiling with ONNX Runtime
   import onnxruntime as ort
   
   options = ort.SessionOptions()
   options.enable_profiling = True
   
   session = ort.InferenceSession("model.onnx", options)
   result = session.run(None, {"input": input_data})
   
   profile = session.end_profiling()
   
   # Analyze profile output
   with open(profile, 'r') as f:
       profile_data = json.load(f)
       
   for entry in profile_data:
       print(f"Layer: {entry['name']}, Time: {entry['dur']}us")
   ```

4. **End-to-End Application Profiling**:
   - Systrace for Android applications
   - Xcode Instruments for iOS applications
   - Custom logging with high-precision timers

   ```cpp
   // High-precision timing example
   auto start = std::chrono::high_resolution_clock::now();
   
   // Run inference
   model.execute(input_tensor, output_tensor);
   
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   
   log_latency(duration.count());
   ```

### Benchmarking Best Practices

1. **Warm-up Runs**: First few inferences often have higher latency
   ```python
   # Example of proper benchmarking with warm-up
   def benchmark_model(model, inputs, num_runs=100, warm_up=10):
       # Warm-up phase
       for i in range(warm_up):
           _ = model.run(inputs)
       
       # Measurement phase
       latencies = []
       for i in range(num_runs):
           start = time.perf_counter()
           _ = model.run(inputs)
           end = time.perf_counter()
           latencies.append((end - start) * 1000)  # Convert to ms
       
       return {
           "mean": statistics.mean(latencies),
           "median": statistics.median(latencies),
           "p95": percentile(latencies, 95),
           "min": min(latencies),
           "max": max(latencies)
       }
   ```

2. **Representative Inputs**: Use real-world data distributions
   - Test with varied inputs, not just synthetic data
   - Include challenging cases (e.g., crowded scenes for object detection)

3. **Sustained Performance Testing**: Run tests over extended periods
   - Important to detect thermal throttling
   - Identify memory leaks or performance degradation

4. **Power Measurement**: Include power metrics for edge devices
   ```python
   # Example using a power monitoring API
   with PowerMonitor() as pm:
       pm.start_measurement()
       run_inference_benchmark(model, inputs)
       power_stats = pm.end_measurement()
       
   efficiency = power_stats["average_power"] / benchmark_results["throughput"]
   print(f"Power efficiency: {efficiency} inferences/watt")
   ```

5. **Cross-Platform Comparison**: Use standardized benchmarks like MLPerf

## Conclusion

Real-time inference on NPUs requires careful attention to latency, throughput, pipeline design, concurrent execution, dynamic input handling, and performance measurement. By applying the techniques discussed in this article, developers can achieve responsive AI applications that make the most of NPU capabilities while meeting strict timing requirements.

In the next part of our series, we'll explore Edge AI Applications with NPUs, focusing on concrete implementations across computer vision, NLP, audio processing, and multi-modal systems. We'll see how the optimization techniques covered here apply to real-world use cases and product deployments.

## Additional Resources

- [MLPerf Inference Benchmarks](https://mlcommons.org/benchmarks/)
- [Real-time ML Systems Design Patterns](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [TensorFlow Lite Performance Best Practices](https://www.tensorflow.org/lite/performance/best_practices)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [Qualcomm AI Engine Direct Documentation](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
