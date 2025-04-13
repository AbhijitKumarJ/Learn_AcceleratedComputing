# NPU Architecture Fundamentals

*Second in a series on NPU programming and optimization*

In our previous article, we introduced Neural Processing Units (NPUs) and explored their importance in the AI hardware ecosystem. Now, we'll dive deeper into how these specialized processors are architected to efficiently execute neural network workloads.

Understanding NPU architecture is crucial for developers seeking to optimize their AI applications. By grasping the underlying hardware principles, you'll be better equipped to write code that fully leverages these accelerators' capabilities and avoids their limitations.

## NPU Internal Architecture Overview

Neural Processing Units vary significantly in their designs, reflecting different performance goals, power constraints, and target applications. However, most modern NPUs share certain architectural principles that distinguish them from general-purpose processors.

### Core Architectural Philosophy

The fundamental design philosophy behind NPUs can be summarized as:

1. **Specialized over general**: Optimize for the specific computational patterns of neural networks rather than maintaining flexibility for arbitrary code.

2. **Parallelism at multiple levels**: Exploit the inherent parallelism in neural computations at various granularities.

3. **Data movement minimization**: Recognize that data movement, not computation, is often the performance and energy bottleneck.

4. **Precision flexibility**: Support reduced precision formats that maintain sufficient accuracy while increasing computational throughput.

### Common Architecture Patterns

Most NPUs implement one or more of these architectural patterns:

#### Systolic Array Architecture

A systolic array organizes processing elements (PEs) in a grid-like structure where data flows rhythmically between neighbors. This architecture is particularly well-suited for matrix multiplication operations, which dominate neural network computation.

Example implementations include:
- Google's TPU uses large systolic arrays (128×128 in TPUv3)
- Many edge NPUs implement smaller systolic arrays optimized for power efficiency

Key advantages:
- High computational density
- Efficient data reuse
- Regular data flow patterns
- Scalable design

#### SIMD/Vector Processing

Single Instruction, Multiple Data (SIMD) architectures execute the same operation across multiple data elements simultaneously. This approach maps well to the vector/tensor operations in neural networks.

Example implementations:
- Arm's Ethos-N NPUs leverage wide SIMD processing
- Intel's Neural Compute Engine in Movidius VPUs uses vector processors

Key advantages:
- Flexible compute capability
- Programmability with vector instructions
- Efficient handling of activation functions and element-wise operations
- Good utilization for smaller networks

#### Dataflow Architecture

Dataflow architectures organize computation based on the flow of data through the neural network graph, rather than executing explicit instructions. Processing elements are activated when their inputs become available.

Example implementations:
- Graphcore's IPU uses a dataflow approach
- Many research NPUs implement variations of this pattern

Key advantages:
- Minimizes control overhead
- Natural mapping to neural network graphs
- Efficient handling of irregular computation patterns
- Better for recurrent and dynamic models

#### Spatial Architecture

Spatial architectures physically map neural network layers onto hardware resources, allowing different parts of the network to process simultaneously with minimal global communication.

Example implementations:
- Tesla's FSD (Full Self-Driving) chip
- Various research proposals like Eyeriss

Key advantages:
- Reduced global data movement
- High energy efficiency
- Natural parallelism across layers
- Good for feed-forward networks

### Hardware Blocks in Modern NPUs

Typical NPUs contain several specialized hardware blocks:

#### Compute Engines

- **Matrix Multiplication Units (MMUs)**: Dedicated to accelerating the core matrix/tensor operations
- **Vector Processing Units (VPUs)**: Handle activation functions and element-wise operations
- **Specialized Function Units (SFUs)**: Hardware implementations of common functions like ReLU, sigmoid, tanh
- **Convolution Engines**: Optimized specifically for convolutional operations

#### Memory Subsystem

- **Local Scratchpad Memory**: Fast, programmer-managed memory close to compute units
- **Weight Caches**: Specialized storage for network parameters
- **Feature Map Buffers**: Optimized for storing activation data between layers
- **DMA Engines**: Manage data movement between memory hierarchies

#### Control Logic

- **Sequencer**: Orchestrates the execution of network layers
- **Dependency Manager**: Tracks data dependencies between operations
- **Power Management**: Controls clock gating and power domains
- **Scheduling Units**: Allocate compute resources to different parts of the model

#### Interconnect

- **Network-on-Chip (NoC)**: Facilitates communication between processing elements
- **Memory Interfaces**: Connect to external DRAM and system memory
- **Peripheral Interfaces**: Allow communication with host processor and I/O

Understanding these hardware components helps developers reason about how their neural network models will map to the physical execution resources.

## Tensor Cores and Processing Elements

At the heart of most NPUs are specialized processing elements designed specifically for tensor operations. These are the computational workhorses that enable the massive parallelism needed for neural network inference.

### Processing Element Design

A Processing Element (PE) in an NPU typically contains:

1. **Multiply-Accumulate (MAC) Units**: The fundamental computational unit for dot products and convolutions. Modern NPUs feature hundreds to thousands of MAC units.

2. **Local Register Files**: Tiny, fast storage for immediate operands.

3. **Control Logic**: Simple instruction decoding or activation logic.

4. **Local Interconnect**: Connections to neighboring PEs and memory.

The efficiency of NPUs largely comes from how these PEs are:
- Optimized for the specific numerical formats used in neural networks
- Arranged to maximize data reuse and minimize movement
- Simplified compared to general CPU cores by removing unnecessary features
- Replicated to enable massive parallelism

### Tensor Cores

Tensor cores are a more sophisticated evolution of basic processing elements, designed specifically to accelerate tensor operations with higher throughput and efficiency.

#### Key features of tensor cores:

1. **Multi-dimensional Processing**: Native support for operations on 2D, 3D, or 4D tensors rather than just vectors.

2. **Fused Operations**: Combined multiply-accumulate with activation functions and normalization in single operations.

3. **Mixed Precision Support**: Hardware support for multiple numerical formats (FP32, FP16, INT8, etc.).

4. **Sparse Computation**: Some tensor cores include specialized hardware for efficiently handling sparse tensors.

#### Examples in industry:

- **NVIDIA Tensor Cores**: First introduced in the Volta architecture, these process 4×4 matrix operations in a single clock cycle.

- **Apple AMX Blocks**: Advanced Matrix Extensions in Apple Silicon provide accelerated tensor processing.

- **Google TPU v4 Tensor Cores**: Optimized for both training and inference with configurable precision.

### Numerical Formats and Precision

A critical aspect of NPU architecture is support for various numerical formats:

#### Common formats in NPU computation:

- **FP32 (32-bit floating point)**: Traditional single precision, used mostly for training or high-precision inference.

- **FP16 (16-bit floating point)**: Half precision, balances accuracy and efficiency for many applications.

- **BF16 (Brain Floating Point)**: 16-bit format with FP32's dynamic range but reduced precision, popular for training.

- **INT8 (8-bit integer)**: Quantized format widely used for efficient inference.

- **INT4/INT2/INT1**: Ultra-low precision formats for maximum throughput at the cost of precision.

Most modern NPUs support multiple formats, with performance typically increasing as precision decreases. For example, INT8 operations might deliver 4× the throughput of FP32 operations on the same hardware.

### Processing Element Utilization

Efficient utilization of processing elements requires careful consideration of:

1. **Compute Intensity**: Operations with higher compute-to-memory ratios (like convolutions) utilize PEs more effectively than memory-bound operations.

2. **Workload Balancing**: Ensuring even distribution of computation across available PEs.

3. **Batching Strategy**: Processing multiple inputs simultaneously to amortize weight loading costs.

4. **Operator Fusion**: Combining multiple operations to keep intermediate results in PE registers.

The organization and utilization of tensor cores and processing elements fundamentally determine the performance characteristics of an NPU implementation.

## Memory Hierarchy in NPUs

Memory systems in NPUs are arguably as important as compute resources for determining overall performance. Due to the high computational intensity of neural networks, memory bandwidth often becomes the limiting factor.

### Memory Hierarchy Levels

NPUs typically implement a deeper and more specialized memory hierarchy than CPUs:

#### On-chip Memories

1. **PE Register Files**: Smallest, fastest storage (bytes to KB) directly attached to processing elements.

2. **Accumulator Memory**: Specialized storage for partial results during tensor operations (often higher precision than inputs).

3. **Local Scratchpad**: Programmer-managed buffers (KB to MB) with deterministic access times.

4. **Global Buffer**: Larger on-chip memory (MB) shared across processing elements.

#### Off-chip Memories

5. **Device DRAM**: Dedicated memory on the NPU module (GB).

6. **System Memory**: Shared with the host processor, accessed via system interconnect.

7. **Storage**: SSD/flash storage for model parameters and large datasets.

Each level in this hierarchy represents a trade-off between:
- Capacity
- Bandwidth
- Access latency
- Energy cost per access
- Programmer visibility/control

### Memory Access Patterns

NPU memory systems are optimized for the specific access patterns of neural networks:

#### Weight Reuse

Neural network weights are typically reused across multiple inputs or input regions. NPU memory systems optimize for:
- Weight broadcasting to multiple PEs
- Weight caching to avoid redundant loads
- Weight compression to reduce bandwidth requirements

#### Activation Data Flow

Activation data (feature maps) typically flow through the network in a pipelined fashion:
- Double-buffering to overlap computation and memory access
- Tiling strategies to keep working sets on-chip
- Ping-pong buffers for producer-consumer layer pairs

#### Memory Banking

To support high-bandwidth parallel access:
- Multiple independent memory banks
- Conflict-free access patterns
- Crossbar interconnects to allow flexible routing

### Bandwidth Considerations

Memory bandwidth is often the primary bottleneck in NPU designs:

#### Bandwidth Requirements Analysis

Consider a simple example:
- A convolutional layer with 3×3 kernels
- 64 input channels, 64 output channels
- 112×112 input feature map

This requires:
- Weights: 3×3×64×64 = 36,864 parameters (144KB at FP32)
- Input activation: 112×112×64 = 802,816 elements (3.1MB at FP32)
- Output activation: 112×112×64 = 802,816 elements (3.1MB at FP32)

If we need to process this layer in 1ms, the required bandwidth exceeds 6GB/s just for this single layer. Real networks with dozens of layers have even higher requirements.

#### Bandwidth Optimization Techniques

NPUs employ various techniques to mitigate bandwidth limitations:

1. **Data Compression**: Weight compression (often 2-4× reduction) and activation compression.

2. **In-memory Computing**: Performing partial computations directly in memory arrays.

3. **Hierarchical Tiling**: Breaking computations into tiles that fit in progressively smaller memory levels.

4. **Predictive Prefetching**: Loading data before it's needed to hide latency.

5. **Sparse Data Handling**: Special memory formats and addressing for sparse tensors.

Understanding memory hierarchy constraints is essential for optimizing neural network models for NPU execution.

## Data Flow Patterns

The movement of data within an NPU fundamentally determines its efficiency. Different NPU architectures implement various dataflow strategies, each with distinct trade-offs.

### Dataflow Taxonomy

NPU dataflow patterns are typically classified based on how they prioritize different types of data reuse:

#### Weight Stationary

- **Approach**: Weights remain fixed in PE registers/local memory while different input activations stream through.
- **Benefits**: Minimizes weight movement, good for networks with large parameter counts.
- **Drawbacks**: May require more activation data movement.
- **Examples**: Many edge NPUs prioritize this approach to minimize DRAM accesses.

#### Output Stationary

- **Approach**: Output partial sums remain fixed in accumulator registers while inputs and weights stream in.
- **Benefits**: Minimizes movement of partial results, reduces accumulation precision loss.
- **Drawbacks**: May require repeated loading of weights and inputs.
- **Examples**: Common in NPUs targeting computer vision applications.

#### Input Stationary

- **Approach**: Input activations remain fixed while different weights stream through.
- **Benefits**: Maximizes input data reuse, good for high input resolution.
- **Drawbacks**: May increase weight data movement.
- **Examples**: Often used in NPUs for video processing.

#### Row Stationary

- **Approach**: Optimizes for reuse across multiple dimensions (weights, inputs, outputs) by mapping computation to minimize total data movement.
- **Benefits**: Better balance of different reuse types.
- **Drawbacks**: More complex control logic and mapping.
- **Examples**: MIT Eyeriss architecture pioneered this approach.

#### Flexible/Hybrid Dataflow

- **Approach**: Reconfigurable dataflow that can adapt to different layer characteristics.
- **Benefits**: Near-optimal efficiency across diverse network layers.
- **Drawbacks**: Increased hardware complexity and programming difficulty.
- **Examples**: More recent NPU designs like Google's TPUv4 and NVIDIA's Hopper architecture.

### Mapping Neural Networks to Dataflows

Different neural network operations map differently to these dataflow patterns:

#### Convolution Layers

For convolutional layers, the dataflow choice depends on:
- Kernel size (larger kernels → weight stationary becomes more efficient)
- Feature map size (larger feature maps → input stationary becomes more efficient)
- Batch size (larger batches → output stationary becomes more efficient)

#### Fully-Connected Layers

Fully-connected layers typically map best to:
- Weight stationary for large output dimensions
- Output stationary for small batches with large input dimensions

#### Recurrent Layers

RNN/LSTM/GRU layers often benefit from:
- Hybrid approaches due to complex dependencies
- Specialized dataflows that handle temporal dependencies

### Dataflow Programming Models

How dataflow patterns expose themselves to programmers varies by NPU:

#### Implicit Dataflow

Most high-level NPU frameworks handle dataflow implicitly:
- Compiler automatically maps operations to the hardware's preferred dataflow
- Programmer provides computation graphs without specifying data movement

#### Explicit Dataflow

Some NPUs, particularly research and high-performance systems, expose dataflow control:
- Manual specification of tiling strategies
- Explicit buffer management
- Dataflow directive annotations

Understanding the underlying dataflow model of an NPU helps developers structure their networks to align with hardware preferences.

## Power Efficiency Considerations

Power efficiency is a primary design goal for most NPUs, especially in edge and mobile deployments. Several architectural techniques contribute to NPU energy efficiency.

### Sources of Power Consumption

To understand NPU power efficiency, we first need to recognize where energy is consumed:

#### Computation Energy

The energy cost of performing arithmetic operations:
- A 32-bit floating-point multiply-add consumes roughly 20× more energy than an 8-bit integer equivalent
- Activation functions implemented in hardware consume significantly less energy than software implementations
- Idle or unused processing elements still consume static power

#### Memory Access Energy

The energy cost of moving data:
- DRAM access consumes roughly 200× more energy than a floating-point operation
- On-chip memory access might consume 5-10× more energy than computation
- The energy cost scales with distance (PE register → local buffer → global buffer → DRAM)

#### Control and Scheduling Overhead

The energy consumed by control logic:
- Instruction decode and dispatch
- Dependency tracking and scheduling
- Clock distribution networks
- Synchronization mechanisms

### Power Efficiency Techniques

NPUs implement various techniques to maximize computational efficiency per watt:

#### Computational Efficiency

1. **Reduced Precision**: Using lower-precision formats dramatically reduces both computation and memory transfer energy.

2. **Specialized Hardware**: Custom circuits for common functions like ReLU or sigmoid consume much less power than general ALU implementations.

3. **Sparsity Exploitation**: Skipping computations on zero values can reduce energy consumption proportionally to model sparsity.

4. **Clock and Power Gating**: Selectively disabling unused hardware blocks to eliminate both dynamic and static power consumption.

#### Memory Efficiency

1. **Hierarchical Memory Organization**: Keeping data in the smallest possible memory structure minimizes energy cost.

2. **Compression Techniques**: Weight and activation compression reduce memory traffic and thus energy consumption.

3. **Predictive Activation**: Techniques that predict which neurons will be significant can reduce unnecessary computation.

4. **In-memory Computing**: Performing operations directly in memory arrays eliminates costly data movement.

#### System-level Efficiency

1. **Dynamic Voltage and Frequency Scaling (DVFS)**: Adjusting operating parameters based on workload demands.

2. **Workload-aware Scheduling**: Batching inference requests to amortize wake-up energy costs.

3. **Hardware-software Co-design**: Designing models with hardware energy constraints in mind.

4. **Heterogeneous Computing**: Offloading different parts of computation to the most efficient processor.

### Power Efficiency Metrics

Several metrics help quantify NPU power efficiency:

#### TOPS/W (Tera Operations Per Second per Watt)

The standard metric for NPU efficiency, measuring computational throughput per unit of power:
- Edge NPUs: 2-10 TOPS/W
- Mobile NPUs: 5-15 TOPS/W
- Data center NPUs: 1-5 TOPS/W (higher absolute performance but lower efficiency)

#### Energy per Inference

The total energy consumed to run a specific model:
- Measured in millijoules (mJ) or microjoules (μJ) per inference
- More application-relevant than TOPS/W
- Takes into account both computation and memory access energy

#### Computational Density

Performance per unit area:
- TOPS/mm² measures how efficiently silicon area is utilized
- Important for cost-sensitive applications
- Typically trades off with power efficiency

Understanding these power efficiency considerations helps developers make informed decisions about model architecture, quantization strategies, and deployment options.

## Putting It All Together: NPU System Architecture

The components we've discussed don't exist in isolation—they form an integrated system designed to accelerate neural network execution efficiently.

### NPU Integration Models

NPUs are integrated into broader computing systems in several ways:

#### Discrete NPUs

- **Characteristics**: Separate chips with dedicated memory
- **Interfaces**: PCIe, USB, custom interconnects
- **Examples**: Google Coral, Intel Neural Compute Stick, server-class accelerators
- **Advantages**: Maximum dedicated resources, scalable deployment
- **Challenges**: Higher communication latency with host processor

#### Integrated NPUs (within SoCs)

- **Characteristics**: NPU block within a larger system-on-chip
- **Interfaces**: Internal AXI/ACE bus, shared memory architecture
- **Examples**: Smartphone SoCs, Apple M-series, embedded processors
- **Advantages**: Lower communication latency, power-efficient, compact
- **Challenges**: Shared resources, thermal constraints

#### Hybrid CPU/NPU Designs

- **Characteristics**: CPU cores with neural network acceleration capabilities
- **Interfaces**: Extended ISA, shared cache hierarchy
- **Examples**: ARM's Cortex-M with Helium extensions, x86 with neural instructions
- **Advantages**: Seamless integration with general computation, unified programming model
- **Challenges**: Limited peak performance compared to dedicated NPUs

### End-to-End Execution Flow

When a neural network executes on an NPU, it typically follows this process:

1. **Model Loading**: Network structure and weights are transferred from storage to NPU-accessible memory.

2. **Compilation/Mapping**: The network is compiled to the NPU's instruction set or configured into its hardware structures.

3. **Input Preparation**: Input data is preprocessed and transferred to NPU memory.

4. **Layer Execution**: The NPU processes each layer according to its dataflow pattern:
   - Loads weights and inputs from appropriate memory levels
   - Performs computation across processing elements
   - Stores output activations to memory

5. **Pipeline Management**: Multiple layers may execute simultaneously in a pipelined fashion.

6. **Output Transfer**: Final results are transferred back to the host processor.

7. **Post-processing**: Results are processed for application consumption (e.g., classification labels, bounding boxes).

This execution flow is orchestrated by a combination of NPU hardware controllers, firmware, and driver software.

### System-level Considerations

Beyond the NPU itself, several system-level factors affect neural network execution:

#### Host Interface Bandwidth

- The connection between the host processor and NPU often becomes a bottleneck
- Design decisions: direct memory access capabilities, shared vs. separate memory

#### Memory Subsystem Design

- Unified vs. separate memory hierarchies
- Cache coherence protocols between NPU and CPU
- Memory consistency models and synchronization mechanisms

#### Software Stack Integration

- Kernel driver interfaces
- Runtime libraries and frameworks
- Programming models and abstractions

#### Power Management

- System-wide power states
- Thermal management and throttling
- Battery life optimization for mobile devices

A holistic understanding of the entire system architecture is crucial for maximizing NPU performance in real-world applications.

## Conclusion and Looking Ahead

In this article, we've explored the fundamental architectural principles that make NPUs highly efficient for neural network computation. We've examined internal organization, processing elements, memory hierarchies, dataflow patterns, and power efficiency considerations that shape modern NPU designs.

Understanding these architectural foundations is essential for developers looking to optimize their neural network models for NPU execution. By aligning model structure, operations, and memory access patterns with the underlying hardware capabilities, developers can achieve significant performance gains and energy savings.

In the next article in our series, we'll build on this architectural knowledge to explore practical aspects of NPU programming. We'll cover development environments, programming models, optimization techniques, and debugging tools that enable developers to effectively leverage NPU hardware.

## Additional Resources

For those interested in deeper explorations of NPU architecture:

- **Research Papers**:
  - "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks" (Chen et al.)
  - "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Google's TPU paper)
  - "Efficient Processing of Deep Neural Networks" (Sze et al.)

- **Architecture Documentation**:
  - ARM Ethos NPU Architecture Guide
  - Qualcomm AI Engine Documentation
  - NVIDIA Deep Learning Accelerator (NVDLA) Technical Reference Manual

- **Online Courses**:
  - Hardware-Software Co-Design for Efficient Neural Network Acceleration
  - Efficient Deep Learning Computing: Algorithms, Systems, and Hardware

Stay tuned for Part 3: Getting Started with NPU Programming, coming next week.

*Note: This article will be regularly updated as NPU architectures evolve.*
