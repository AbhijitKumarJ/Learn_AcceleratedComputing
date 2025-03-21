# Lesson 1: Spatial Computing Architectures

## Introduction
Spatial computing represents a fundamental shift in how we approach computation, moving away from the sequential, instruction-driven model that has dominated computing for decades. In spatial architectures, computation is distributed across physical space, with operations occurring in parallel across dedicated hardware units. This paradigm is particularly well-suited for data-intensive applications like machine learning, signal processing, and scientific computing, where traditional architectures struggle with the "memory wall" and power constraints.

## Subtopics:
- Introduction to spatial computing paradigms
- Coarse-grained reconfigurable arrays (CGRAs)
- Dataflow architectures and their advantages
- Spatial vs. temporal computing models
- Programming models for spatial architectures
- Comparison with traditional von Neumann architectures
- Real-world implementations: Wave Computing, Cerebras, SambaNova
- Future directions in spatial computing design

## Key Terminology and Concept Definitions
- **Spatial Computing**: Computing paradigm where computations are mapped directly to physical hardware locations, enabling parallel execution. Unlike temporal computing where operations share hardware resources over time, spatial computing dedicates specific hardware to specific operations.

- **CGRA (Coarse-Grained Reconfigurable Array)**: Reconfigurable architecture with functional units larger than individual gates but smaller than full processors. CGRAs typically consist of an array of processing elements (PEs) that can be configured to perform various operations and connected through a reconfigurable interconnect network.

- **Dataflow Architecture**: Hardware design where operations execute as soon as their input data becomes available, rather than following a program counter. This execution model naturally exposes parallelism and reduces control overhead.

- **Von Neumann Bottleneck**: Performance limitation caused by the separation of processing and memory in traditional architectures, resulting in a communication bottleneck between the CPU and memory. This bottleneck becomes increasingly problematic as computational capabilities outpace memory bandwidth improvements.

- **Processing Element (PE)**: The basic computational unit in a spatial architecture, typically containing ALUs, registers, and local memory.

- **Systolic Array**: A specialized form of spatial architecture where data flows rhythmically through an array of processing elements, with each element performing some computation on the data before passing it to its neighbors.

- **Spatial Locality**: The property of data elements being physically located near each other in memory or in a computational array, enabling efficient access patterns.

- **Reconfigurable Interconnect**: The network connecting processing elements in a spatial architecture, which can be reconfigured to support different dataflow patterns.

## Architectural Diagrams and Visual Explanations

### Spatial vs. Temporal Execution Models
In temporal computing, a single processing unit executes different operations over time, while in spatial computing, different operations are executed simultaneously by dedicated hardware units. This fundamental difference can be visualized as:

```
Temporal Computing (Von Neumann):
Time →  t1      t2      t3      t4
      +------+------+------+------+
CPU   | Op A | Op B | Op C | Op D |
      +------+------+------+------+

Spatial Computing:
      +------+------+------+------+
      | Op A | Op B | Op C | Op D |  ← All operations
      +------+------+------+------+    execute at once
        PE 1   PE 2   PE 3   PE 4
```

### CGRA Architecture
A typical CGRA consists of an array of processing elements (PEs) connected by a reconfigurable interconnect network:

```
    +-----+     +-----+     +-----+     +-----+
    | PE  |<--->| PE  |<--->| PE  |<--->| PE  |
    +-----+     +-----+     +-----+     +-----+
       ↑           ↑           ↑           ↑
       ↓           ↓           ↓           ↓
    +-----+     +-----+     +-----+     +-----+
    | PE  |<--->| PE  |<--->| PE  |<--->| PE  |
    +-----+     +-----+     +-----+     +-----+
       ↑           ↑           ↑           ↑
       ↓           ↓           ↓           ↓
    +-----+     +-----+     +-----+     +-----+
    | PE  |<--->| PE  |<--->| PE  |<--->| PE  |
    +-----+     +-----+     +-----+     +-----+
```

Each PE typically contains:
- Arithmetic Logic Unit (ALU)
- Local register file
- Configuration memory
- Input/output buffers

### Dataflow Execution Visualization
In a dataflow architecture, operations execute as soon as their inputs are available:

```
    +-------+        +-------+
    | Input |------->| Add   |
    | Data  |        |       |
    +-------+        +-------+
                         |
                         v
    +-------+        +-------+        +-------+
    | Const |------->| Mult  |------->| Output|
    | Value |        |       |        |       |
    +-------+        +-------+        +-------+
```

This contrasts with control flow execution where operations execute according to a program counter, regardless of data availability.

## Comparative Analysis with Conventional Approaches

| Feature | Spatial Computing | Von Neumann Architecture |
|---------|-------------------|--------------------------|
| Parallelism | Inherent spatial parallelism with hundreds to thousands of PEs operating simultaneously | Sequential with limited parallelism through multi-core, SIMD, or threading |
| Memory Access | Distributed, local memory with direct PE-to-PE communication reducing memory traffic | Centralized memory hierarchy with high latency and energy cost for data movement |
| Programming Model | Dataflow, spatial mapping requiring explicit parallelism and data orchestration | Imperative, sequential programming with implicit control flow |
| Energy Efficiency | Higher for parallel workloads due to reduced data movement and specialized hardware | Lower due to memory transfers and general-purpose design |
| Flexibility | Application-specific, often optimized for specific domains like ML or signal processing | General purpose, capable of running diverse workloads |
| Scalability | Scales well with problem size by adding more processing elements | Limited by memory bandwidth and coherence protocols |
| Clock Frequency | Often operates at lower frequencies, compensating with massive parallelism | Relies on high clock frequencies for performance |
| Compilation | Requires specialized compilers that understand spatial mapping and resource constraints | Well-established compilation techniques and optimization strategies |

## Spatial Computing Paradigms in Detail

### Systolic Arrays
Systolic arrays represent one of the earliest and most successful forms of spatial computing. They consist of a regular grid of processing elements where data flows rhythmically through the array, with each PE performing computations on the data before passing it to its neighbors.

Key characteristics:
- Regular, rhythmic data flow (like a heartbeat, hence "systolic")
- Simple, identical processing elements
- Local communication only between adjacent PEs
- Highly efficient for matrix operations

Google's Tensor Processing Unit (TPU) is a modern implementation of a systolic array, optimized for matrix multiplication in deep learning applications.

### Coarse-Grained Reconfigurable Arrays (CGRAs)

CGRAs occupy a middle ground between fine-grained FPGAs and fixed-function ASICs. They consist of an array of processing elements that can be configured to perform various operations, connected through a reconfigurable interconnect network.

Key characteristics:
- Processing elements operate on word-level data (16-32 bits) rather than individual bits
- Reconfigurable at runtime, often in cycles rather than milliseconds (unlike FPGAs)
- Higher computational density than FPGAs
- Lower power consumption than general-purpose processors
- More flexible than fixed-function accelerators

Examples include:
- ADRES (Architecture for Dynamically Reconfigurable Embedded Systems)
- MorphoSys
- TRIPS (Tera-op Reliable Intelligently adaptive Processing System)
- Plasticine from Stanford

### Dataflow Architectures

Dataflow architectures execute operations based on data availability rather than control flow. This naturally exposes parallelism and reduces the overhead of instruction fetching and decoding.

Key characteristics:
- Data-driven execution model
- No program counter
- Operations fire when all inputs are available
- Natural expression of parallelism
- Reduced control overhead

Modern implementations include:
- Wave Computing's DPU (Dataflow Processing Unit)
- SambaNova's RDU (Reconfigurable Dataflow Unit)
- Graphcore's IPU (Intelligence Processing Unit)

## Current Research Highlights and Breakthrough Technologies

### Cerebras Wafer-Scale Engine (WSE)
Cerebras has pushed the boundaries of spatial computing with its Wafer-Scale Engine, which integrates an entire wafer as a single chip:

- 46,225 mm² silicon area (compared to ~800 mm² for large GPUs)
- 2.6 trillion transistors
- 850,000 AI-optimized cores
- 40 GB on-chip memory with 20 PB/s memory bandwidth
- 220 petabit/s interconnect bandwidth

The WSE-2, their second-generation chip, further improves on these specifications and demonstrates the extreme scale possible with spatial architectures. The massive on-chip memory and interconnect bandwidth address the fundamental bottlenecks in deep learning workloads.

### SambaNova's Reconfigurable Dataflow Unit (RDU)
SambaNova Systems has developed a spatial architecture specifically designed for dataflow computing:

- Cardinal SN10 RDU chip with reconfigurable dataflow architecture
- Software-defined hardware approach that adapts to different workloads
- Optimized for both training and inference
- Dataflow graph compiler that maps neural networks directly to hardware
- Demonstrated 6x performance improvement over leading GPUs for certain workloads

SambaNova's approach focuses on the software stack as much as the hardware, with their SambaFlow software mapping dataflow graphs directly to the reconfigurable hardware.

### Stanford's Plasticine
Plasticine is a research architecture from Stanford University that represents a new approach to CGRAs:

- Designed specifically for parallel patterns common in high-performance applications
- Hierarchical reconfigurable architecture with two types of cores:
  - Pattern Compute Units (PCUs) for computation
  - Pattern Memory Units (PMUs) for memory access
- Specialized for pipeline parallelism and nested patterns
- Demonstrated 50-100x energy efficiency improvement over FPGAs

Plasticine addresses the limitations of previous CGRAs by focusing on higher-level patterns rather than individual operations.

### Wave Computing's Dataflow Processing Unit (DPU)
Wave Computing developed a dataflow architecture specifically for deep learning:

- Thousands of processing elements organized in clusters
- Self-timed dataflow execution model
- Coarse-grained reconfigurable fabric
- Native support for sparse computations
- Optimized for both training and inference

Wave Computing's architecture demonstrates how dataflow principles can be applied to specific domains like deep learning.

## Industry Adoption Status and Commercial Availability

### Cerebras Systems
- Commercial availability of CS-2 systems powered by the WSE-2
- Deployed in major research labs, enterprises, and supercomputing centers
- Partnerships with pharmaceutical companies for drug discovery
- Integration with supercomputing centers like EPCC and Argonne National Laboratory
- Cerebras CS-2 systems available through cloud providers for on-demand access
- Pricing in the millions of dollars per system, targeting enterprise and research markets

### SambaNova Systems
- DataScale SN30 systems commercially available
- Offered both as hardware purchase and as "Dataflow-as-a-Service"
- Deployed in financial services, healthcare, manufacturing, and research
- Partnership with Lawrence Livermore National Laboratory
- SambaNova AI Platform provides end-to-end solution from model development to deployment
- Subscription-based pricing model makes the technology more accessible

### Graphcore
- Intelligence Processing Unit (IPU) incorporates spatial computing concepts
- Commercially available in Graphcore M2000 systems
- IPU-POD configurations for scaling from research to production
- Poplar SDK provides software tools for programming IPUs
- Available through cloud providers like Microsoft Azure
- Second-generation Colossus MK2 GC200 IPU with 1,472 processor cores and 8,832 parallel threads

### Research and Academic Platforms
- FPGA-based CGRA implementations available for research
- Open-source CGRA frameworks like CGRA-ME and OpenCGRA
- University research platforms not commercially productized
- Simulation environments for spatial architecture exploration

### Adoption Challenges
- Software ecosystem maturity compared to GPUs
- Programming complexity requiring specialized expertise
- Integration with existing workflows and frameworks
- High initial investment costs
- Limited availability of benchmarks and performance comparisons

## Programming Considerations and Software Ecosystems

### Domain-Specific Languages for Spatial Hardware

Programming spatial architectures requires expressing both computation and the mapping of that computation to physical hardware resources. Several domain-specific languages (DSLs) have been developed to address this challenge:

- **Spatial**: A DSL developed at Stanford specifically for programming spatial architectures. It provides abstractions for expressing parallelism, memory access patterns, and pipelining.

- **Halide**: Originally designed for image processing, Halide separates algorithm description from scheduling, making it well-suited for spatial architectures.

- **TensorFlow XLA (Accelerated Linear Algebra)**: Compiles TensorFlow computations into optimized code for various accelerators, including spatial architectures.

- **MLIR (Multi-Level Intermediate Representation)**: Provides a framework for representing and transforming programs at multiple levels of abstraction, facilitating mapping to spatial hardware.

### Compiler Technologies for Spatial Mapping

Compiling for spatial architectures involves several unique challenges:

- **Resource Allocation**: Mapping operations to physical processing elements while respecting hardware constraints.

- **Data Movement Optimization**: Minimizing data movement between processing elements and memory.

- **Pipeline Scheduling**: Determining the timing of operations to maximize throughput.

- **Memory Banking**: Organizing data in memory to enable parallel access.

Advanced compiler techniques include:
- Polyhedral optimization for loop transformations
- Graph partitioning for workload distribution
- Dataflow analysis for identifying parallelism
- Simulation-based performance modeling

### Programming Abstractions

Several programming abstractions help manage the complexity of spatial architectures:

- **Dataflow Graphs**: Representing computation as a graph where nodes are operations and edges represent data dependencies.

- **Stream Processing**: Modeling computation as operations on continuous data streams.

- **Parallel Patterns**: High-level patterns like map, reduce, stencil that can be efficiently mapped to spatial hardware.

- **Tiling and Blocking**: Techniques for partitioning data to match hardware capabilities.

### Frameworks and Tools

Several frameworks and tools support programming spatial architectures:

- **SambaNova SambaFlow**: End-to-end framework for the SambaNova RDU, including a Python API and dataflow compiler.

- **Cerebras Software Platform**: Integrates with popular ML frameworks like TensorFlow and PyTorch.

- **Graphcore Poplar SDK**: Programming environment for the Graphcore IPU.

- **CGRA-ME**: Open-source framework for modeling and evaluating CGRAs.

- **VTR (Verilog-to-Routing)**: Tool for mapping designs to reconfigurable architectures.

### Debugging and Profiling

Debugging and profiling spatial architectures present unique challenges:

- **Visualization Tools**: Tools for visualizing dataflow graphs and their mapping to hardware.

- **Simulation Environments**: Cycle-accurate simulators for performance prediction.

- **Hardware Monitors**: On-chip monitoring for runtime performance analysis.

- **Trace Analysis**: Tools for analyzing execution traces to identify bottlenecks.

## Hands-on Examples

### Example 1: Matrix Multiplication on a Spatial Architecture

Matrix multiplication is a fundamental operation that benefits significantly from spatial computing. Here's how it can be implemented on a systolic array:

```python
# Pseudocode for matrix multiplication on a systolic array
# Assume matrices A (M x K) and B (K x N)

# Configuration phase - set up the systolic array
for i in range(M):
    for j in range(N):
        configure_pe(i, j, operation="multiply_accumulate")

# Execution phase
# Data is fed into the array from the edges
for k in range(K):
    for i in range(M):
        feed_input_a(i, 0, A[i][k])  # Feed A from left edge
    
    for j in range(N):
        feed_input_b(0, j, B[k][j])  # Feed B from top edge

# Result collection phase
for i in range(M):
    for j in range(N):
        C[i][j] = read_output(i, j)  # Read result from each PE
```

In an actual implementation, the data feeding would be pipelined, with each PE performing a multiply-accumulate operation as soon as its inputs arrive.

### Example 2: Mapping a Convolutional Neural Network to a CGRA

Convolutional Neural Networks (CNNs) are naturally suited for spatial architectures. Here's a simplified example of mapping a CNN layer to a CGRA:

```python
# Pseudocode for mapping a CNN layer to a CGRA
# Assume input feature map (H x W x C_in) and output (H' x W' x C_out)
# with kernel size K x K

# Tile the output feature map to match the CGRA size
tile_height = min(H', cgra_height)
tile_width = min(W', cgra_width)

for c_out in range(0, C_out, cgra_depth):  # Process output channels in groups
    for h in range(0, H', tile_height):
        for w in range(0, W', tile_width):
            # Configure the CGRA for this tile
            for i in range(tile_height):
                for j in range(tile_width):
                    for c in range(min(C_out - c_out, cgra_depth)):
                        # Configure PE at position (i,j,c) for convolution
                        configure_pe(i, j, c, operation="conv")
                        load_weights(i, j, c, weights[c_out+c])
            
            # Stream input data to the CGRA
            for c_in in range(C_in):
                for i in range(h, min(h + tile_height + K - 1, H)):
                    for j in range(w, min(w + tile_width + K - 1, W)):
                        stream_input(input_feature_map[i][j][c_in])
            
            # Collect output data
            for i in range(tile_height):
                for j in range(tile_width):
                    for c in range(min(C_out - c_out, cgra_depth)):
                        output_feature_map[h+i][w+j][c_out+c] = read_output(i, j, c)
```

This example demonstrates the tiling strategy needed to map a large CNN to a fixed-size CGRA, as well as the streaming of input data and collection of results.

### Example 3: Dataflow Graph Specification for Image Processing

Here's an example of specifying a simple image processing pipeline as a dataflow graph:

```python
# Using a hypothetical dataflow graph API

# Create a dataflow graph for a simple image processing pipeline
graph = DataflowGraph()

# Define input and output
input_image = graph.create_input("input", shape=(height, width, channels))
output_image = graph.create_output("output", shape=(height, width, channels))

# Define operations
# 1. Convert to grayscale
grayscale = graph.add_node("grayscale_conversion", inputs=[input_image])

# 2. Apply Gaussian blur
blurred = graph.add_node("gaussian_blur", inputs=[grayscale], kernel_size=5, sigma=1.0)

# 3. Apply Sobel edge detection
edges_x = graph.add_node("sobel_filter", inputs=[blurred], direction="horizontal")
edges_y = graph.add_node("sobel_filter", inputs=[blurred], direction="vertical")

# 4. Compute edge magnitude
edge_magnitude = graph.add_node("magnitude", inputs=[edges_x, edges_y])

# 5. Apply threshold
thresholded = graph.add_node("threshold", inputs=[edge_magnitude], threshold=128)

# Connect to output
graph.connect(thresholded, output_image)

# Compile the graph for a specific spatial architecture
compiled_graph = compiler.compile(graph, target="example_spatial_architecture")

# Execute the compiled graph
result = runtime.execute(compiled_graph, {input_image: load_image("example.jpg")})
```

This example shows how a complex image processing pipeline can be expressed as a dataflow graph, which can then be compiled and mapped to a spatial architecture.

### Example 4: Performance Comparison

Here's a simplified example comparing the performance of matrix multiplication on a CPU, GPU, and a spatial architecture:

```python
import numpy as np
import time
import spatial_simulator  # Hypothetical spatial architecture simulator

# Define matrix dimensions
M, K, N = 1024, 1024, 1024

# Generate random matrices
A = np.random.rand(M, K).astype(np.float32)
B = np.random.rand(K, N).astype(np.float32)

# CPU implementation
start_time = time.time()
C_cpu = np.matmul(A, B)
cpu_time = time.time() - start_time
print(f"CPU time: {cpu_time:.4f} seconds")

# GPU implementation (using numpy with GPU backend)
import cupy as cp
A_gpu = cp.asarray(A)
B_gpu = cp.asarray(B)
cp.cuda.Stream.null.synchronize()
start_time = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start_time
print(f"GPU time: {gpu_time:.4f} seconds")

# Spatial architecture implementation
spatial_arch = spatial_simulator.SystolicArray(size=32)  # 32x32 systolic array
start_time = time.time()
C_spatial = spatial_arch.matmul(A, B)
spatial_time = time.time() - start_time
print(f"Spatial architecture time: {spatial_time:.4f} seconds")

# Verify results
np.testing.assert_allclose(C_cpu, C_gpu.get(), rtol=1e-5)
np.testing.assert_allclose(C_cpu, C_spatial, rtol=1e-5)

# Performance comparison
print(f"GPU speedup over CPU: {cpu_time / gpu_time:.2f}x")
print(f"Spatial speedup over CPU: {cpu_time / spatial_time:.2f}x")
print(f"Spatial speedup over GPU: {gpu_time / spatial_time:.2f}x")

# Energy efficiency comparison (hypothetical values)
cpu_energy = cpu_time * 150  # Watts
gpu_energy = gpu_time * 300  # Watts
spatial_energy = spatial_time * 50  # Watts
print(f"CPU energy: {cpu_energy:.2f} Joules")
print(f"GPU energy: {gpu_energy:.2f} Joules")
print(f"Spatial energy: {spatial_energy:.2f} Joules")
print(f"Spatial energy efficiency vs CPU: {cpu_energy / spatial_energy:.2f}x")
print(f"Spatial energy efficiency vs GPU: {gpu_energy / spatial_energy:.2f}x")
```

This example demonstrates how to compare performance and energy efficiency across different architectures, highlighting the advantages of spatial computing for specific workloads.

## Future Outlook and Research Directions

### Integration with Emerging Memory Technologies

The future of spatial computing is closely tied to advances in memory technology:

- **3D-stacked memory**: Enabling higher bandwidth and lower latency access to data, critical for data-intensive spatial computing applications.

- **Processing-in-memory (PIM)**: Integrating computational capabilities directly into memory arrays, further reducing the data movement bottleneck.

- **Non-volatile memory (NVM)**: Technologies like ReRAM, PCM, and MRAM offering persistence, higher density, and potential for in-memory computing.

- **Near-memory processing**: Placing processing elements close to memory to reduce data movement while maintaining flexibility.

Research is focusing on heterogeneous integration of these memory technologies with spatial computing architectures to create systems that optimize both computation and data access.

### Dynamic Reconfiguration for Adaptive Spatial Computing

Current spatial architectures often require significant time for reconfiguration, limiting their adaptability:

- **Fast reconfiguration mechanisms**: Reducing the time needed to reconfigure the architecture for different workloads.

- **Partial reconfiguration**: Allowing portions of the architecture to be reconfigured while others continue operating.

- **Context switching**: Supporting rapid switching between pre-loaded configurations.

- **Runtime adaptation**: Enabling the architecture to adapt to changing workload characteristics during execution.

These capabilities would allow spatial architectures to efficiently support a wider range of applications and adapt to dynamic workload requirements.

### Heterogeneous Spatial Architectures

Future spatial architectures will likely combine different types of processing elements to address diverse computational needs:

- **Mixed-granularity PEs**: Combining fine-grained, medium-grained, and coarse-grained reconfigurable elements.

- **Specialized functional units**: Integrating domain-specific accelerators for common operations like FFT, cryptography, or compression.

- **Neuromorphic elements**: Incorporating brain-inspired computing units for specific AI workloads.

- **Quantum-classical integration**: Exploring interfaces between classical spatial architectures and quantum computing elements.

This heterogeneity will enable more efficient execution of complex applications with diverse computational requirements.

### Standardization of Programming Interfaces

For spatial computing to achieve mainstream adoption, standardized programming interfaces are essential:

- **Common intermediate representations**: Developing standard IRs that can target multiple spatial architectures.

- **High-level programming abstractions**: Creating programming models that hide the complexity of spatial hardware.

- **Portable performance models**: Enabling performance prediction across different spatial architectures.

- **Interoperability standards**: Allowing spatial accelerators to work seamlessly with other system components.

Initiatives like MLIR (Multi-Level IR) and SYCL (Standard for Heterogeneous Computing) are steps in this direction.

### Application Expansion Beyond AI

While AI has been a primary driver for spatial computing, the technology has potential in many other domains:

- **Scientific computing**: Computational fluid dynamics, molecular dynamics, and climate modeling.

- **Signal processing**: Software-defined radio, radar processing, and sensor fusion.

- **Genomics**: DNA sequence alignment, variant calling, and protein folding simulation.

- **Financial modeling**: Risk analysis, option pricing, and high-frequency trading.

- **Cryptography**: Blockchain mining, homomorphic encryption, and post-quantum cryptography.

Research is exploring how spatial architectures can be optimized for these diverse application domains.

### Addressing Scalability Challenges

As spatial architectures grow in size, several scalability challenges emerge:

- **Power delivery and thermal management**: Developing techniques to efficiently power and cool large spatial arrays.

- **Yield and fault tolerance**: Creating architectures that can tolerate manufacturing defects and runtime faults.

- **Programming model scalability**: Ensuring programming models remain effective as system size increases.

- **Interconnect scalability**: Developing network-on-chip architectures that scale to thousands or millions of processing elements.

Addressing these challenges will be critical for the continued scaling of spatial computing capabilities.

### Convergence with Other Computing Paradigms

The future may see convergence between spatial computing and other emerging computing paradigms:

- **Neuromorphic computing**: Brain-inspired architectures that naturally map to spatial organization.

- **Quantum computing**: Hybrid systems where spatial architectures handle pre- and post-processing for quantum algorithms.

- **Probabilistic computing**: Spatial implementations of probabilistic computing models.

- **Approximate computing**: Spatial architectures that trade precision for efficiency in error-tolerant applications.

This convergence could lead to novel computing systems that combine the strengths of multiple paradigms to address complex computational challenges.