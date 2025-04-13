# GPU Computing for Scientific Simulations

*Welcome to the fifteenth installment of our GPU programming series! In this article, we'll explore how GPUs accelerate scientific simulations, focusing on finite element methods, fluid dynamics, molecular dynamics, and high-precision scientific computing considerations.*

## Introduction to Scientific Computing on GPUs

Scientific computing involves using computers to solve complex mathematical models that describe physical phenomena. These simulations often require massive computational resources, making them ideal candidates for GPU acceleration. The highly parallel nature of GPUs can provide orders of magnitude speedup for many scientific algorithms compared to traditional CPU implementations.

In this article, we'll explore several key areas where GPUs have revolutionized scientific computing, including finite element methods, fluid dynamics, molecular dynamics, and considerations for high-precision calculations.

## Finite Element Methods on GPUs

Finite Element Methods (FEM) are numerical techniques for solving partial differential equations (PDEs) that describe various physical phenomena, from structural mechanics to electromagnetics. FEM works by dividing a complex domain into simpler subdomains (finite elements) and approximating the solution within each element.

### Basic FEM Workflow

A typical FEM simulation involves these steps:
1. Domain discretization (mesh generation)
2. Element stiffness matrix computation
3. Global stiffness matrix assembly
4. Boundary condition application
5. Linear system solution
6. Post-processing of results

GPUs can accelerate several of these steps, particularly the computationally intensive ones.

### Element Stiffness Matrix Computation

Computing element stiffness matrices is highly parallelizable and well-suited for GPU acceleration:

```cuda
// Example: Computing element stiffness matrices for 2D linear triangular elements
__global__ void compute_element_stiffness_matrices(
    float* coordinates,     // Node coordinates [num_nodes x 2]
    int* elements,          // Element connectivity [num_elements x 3]
    float* material_props,  // Material properties [num_elements x 2] (E, nu)
    float* stiffness_matrices, // Output [num_elements x 36]
    int num_elements
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;
    
    // Get element nodes
    int n1 = elements[e * 3 + 0];
    int n2 = elements[e * 3 + 1];
    int n3 = elements[e * 3 + 2];
    
    // Get node coordinates
    float x1 = coordinates[n1 * 2 + 0];
    float y1 = coordinates[n1 * 2 + 1];
    float x2 = coordinates[n2 * 2 + 0];
    float y2 = coordinates[n2 * 2 + 1];
    float x3 = coordinates[n3 * 2 + 0];
    float y3 = coordinates[n3 * 2 + 1];
    
    // Compute element area
    float area = 0.5f * fabsf((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    
    // Compute shape function derivatives
    float b11 = (y2 - y3) / (2.0f * area);
    float b12 = (y3 - y1) / (2.0f * area);
    float b13 = (y1 - y2) / (2.0f * area);
    float b21 = (x3 - x2) / (2.0f * area);
    float b22 = (x1 - x3) / (2.0f * area);
    float b23 = (x2 - x1) / (2.0f * area);
    
    // Get material properties
    float E = material_props[e * 2 + 0];  // Young's modulus
    float nu = material_props[e * 2 + 1]; // Poisson's ratio
    
    // Compute constitutive matrix for plane stress
    float factor = E / (1.0f - nu * nu);
    float D11 = factor;
    float D12 = factor * nu;
    float D21 = D12;
    float D22 = D11;
    float D33 = factor * (1.0f - nu) / 2.0f;
    
    // Compute B matrix
    float B[6][6] = {0};
    B[0][0] = b11; B[0][2] = b12; B[0][4] = b13;
    B[1][1] = b21; B[1][3] = b22; B[1][5] = b23;
    B[2][0] = b21; B[2][1] = b11; B[2][2] = b22;
    B[2][3] = b12; B[2][4] = b23; B[2][5] = b13;
    
    // Compute D*B
    float DB[6][6] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            DB[i][j] = D11 * B[0][j] + D12 * B[1][j] + D12 * B[2][j];
            DB[i+3][j] = D21 * B[0][j] + D22 * B[1][j] + D33 * B[2][j];
        }
    }
    
    // Compute B^T*D*B*area (element stiffness matrix)
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += B[k][i] * DB[k][j];
            }
            stiffness_matrices[e * 36 + i * 6 + j] = sum * area;
        }
    }
}
```

### Global Matrix Assembly

Assembling the global stiffness matrix can be challenging on GPUs due to race conditions when multiple elements contribute to the same global matrix entries. One approach is to use atomic operations:

```cuda
// Example: Global stiffness matrix assembly using atomic operations
__global__ void assemble_global_stiffness_matrix(
    int* elements,              // Element connectivity [num_elements x 3]
    float* element_matrices,    // Element stiffness matrices [num_elements x 36]
    float* global_matrix,       // Global stiffness matrix [num_dofs x num_dofs]
    int num_elements,
    int num_dofs_per_node,
    int num_nodes_per_element
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements) return;
    
    // Get element nodes
    int element_nodes[3];
    for (int i = 0; i < num_nodes_per_element; i++) {
        element_nodes[i] = elements[e * num_nodes_per_element + i];
    }
    
    // Assemble element matrix into global matrix
    for (int i = 0; i < num_nodes_per_element; i++) {
        int row_node = element_nodes[i];
        for (int j = 0; j < num_nodes_per_element; j++) {
            int col_node = element_nodes[j];
            
            for (int di = 0; di < num_dofs_per_node; di++) {
                int row = row_node * num_dofs_per_node + di;
                for (int dj = 0; dj < num_dofs_per_node; dj++) {
                    int col = col_node * num_dofs_per_node + dj;
                    
                    int local_i = i * num_dofs_per_node + di;
                    int local_j = j * num_dofs_per_node + dj;
                    float value = element_matrices[e * 36 + local_i * 6 + local_j];
                    
                    // Use atomic add to avoid race conditions
                    atomicAdd(&global_matrix[row * num_dofs + col], value);
                }
            }
        }
    }
}
```

Alternatively, a coloring scheme can be used to avoid race conditions:

```cuda
// Example: Global matrix assembly using graph coloring
void assemble_global_matrix_with_coloring(
    int* elements,
    float* element_matrices,
    float* global_matrix,
    int* element_colors,
    int num_colors,
    int num_elements
) {
    // Process elements color by color
    for (int color = 0; color < num_colors; color++) {
        // Launch kernel for current color
        assemble_elements_of_color<<<(num_elements + 255) / 256, 256>>>(
            elements, element_matrices, global_matrix, element_colors, color, num_elements);
        cudaDeviceSynchronize();
    }
}

__global__ void assemble_elements_of_color(
    int* elements,
    float* element_matrices,
    float* global_matrix,
    int* element_colors,
    int current_color,
    int num_elements
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_elements || element_colors[e] != current_color) return;
    
    // Assembly code (same as before but without atomics)
    // ...
}
```

### Linear System Solution

Solving the resulting linear system is often the most time-consuming part of FEM. GPU-accelerated libraries like cuSPARSE and cuSOLVER can be used for direct or iterative solvers:

```cpp
// Example: Conjugate Gradient solver using cuSPARSE and cuBLAS
void solve_cg(
    cusparseHandle_t cusparse_handle,
    cublasHandle_t cublas_handle,
    int n,
    int nnz,
    float* d_val,
    int* d_row_ptr,
    int* d_col_ind,
    float* d_b,
    float* d_x,
    float tol,
    int max_iter
) {
    // Allocate memory for vectors
    float *d_r, *d_p, *d_Ap;
    cudaMalloc(&d_r, n * sizeof(float));
    cudaMalloc(&d_p, n * sizeof(float));
    cudaMalloc(&d_Ap, n * sizeof(float));
    
    // Initialize x to zero
    cudaMemset(d_x, 0, n * sizeof(float));
    
    // r = b - A*x (initially r = b since x = 0)
    cudaMemcpy(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // p = r
    cudaMemcpy(d_p, d_r, n * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Compute initial residual norm
    float r_norm, b_norm, alpha, beta;
    cublasSdot(cublas_handle, n, d_r, 1, d_r, 1, &r_norm);
    cublasSdot(cublas_handle, n, d_b, 1, d_b, 1, &b_norm);
    
    float threshold = tol * tol * b_norm;
    
    // Main CG loop
    for (int iter = 0; iter < max_iter && r_norm > threshold; iter++) {
        // Compute Ap
        float one = 1.0f, zero = 0.0f;
        cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one, mat_descr, vec_descr_p, &zero, vec_descr_Ap,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        
        // Compute alpha = r^T * r / (p^T * Ap)
        float p_Ap;
        cublasSdot(cublas_handle, n, d_p, 1, d_Ap, 1, &p_Ap);
        alpha = r_norm / p_Ap;
        
        // Update x and r
        cublasSaxpy(cublas_handle, n, &alpha, d_p, 1, d_x, 1);
        
        float neg_alpha = -alpha;
        cublasSaxpy(cublas_handle, n, &neg_alpha, d_Ap, 1, d_r, 1);
        
        // Compute new residual norm and beta
        float r_norm_new;
        cublasSdot(cublas_handle, n, d_r, 1, d_r, 1, &r_norm_new);
        beta = r_norm_new / r_norm;
        r_norm = r_norm_new;
        
        // Update p
        cublasSscal(cublas_handle, n, &beta, d_p, 1);
        cublasSaxpy(cublas_handle, n, &one, d_r, 1, d_p, 1);
    }
    
    // Clean up
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
}
```

## Fluid Dynamics Simulations

Computational Fluid Dynamics (CFD) simulates fluid flow by solving the Navier-Stokes equations. GPUs can significantly accelerate these computationally intensive simulations.

### Lattice Boltzmann Method (LBM)

The Lattice Boltzmann Method is particularly well-suited for GPU implementation due to its explicit, local nature:

```cuda
// Example: 2D Lattice Boltzmann Method (D2Q9 model)
#define Q 9  // Number of discrete velocities

__constant__ float weights[Q] = {4.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f};
__constant__ int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

__global__ void lbm_collision_streaming(
    float* f,          // Distribution functions
    float* f_new,      // New distribution functions
    int* obstacles,    // Obstacle map (1 for solid, 0 for fluid)
    int width,
    int height,
    float omega,       // Relaxation parameter
    float gravity_x,   // External force in x direction
    float gravity_y    // External force in y direction
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    if (obstacles[idx]) {
        // Bounce-back for obstacle nodes
        for (int i = 0; i < Q; i++) {
            int opp = (i == 0) ? 0 : (i + 4) % 8 + (i > 4);
            f_new[idx * Q + i] = f[idx * Q + opp];
        }
        return;
    }
    
    // Compute macroscopic variables
    float rho = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;
    
    for (int i = 0; i < Q; i++) {
        float fi = f[idx * Q + i];
        rho += fi;
        ux += cx[i] * fi;
        uy += cy[i] * fi;
    }
    
    ux = ux / rho + 0.5f * gravity_x;
    uy = uy / rho + 0.5f * gravity_y;
    
    // Collision step (BGK approximation)
    for (int i = 0; i < Q; i++) {
        float cu = 3.0f * (cx[i] * ux + cy[i] * uy);
        float feq = weights[i] * rho * (1.0f + cu + 0.5f * cu * cu - 1.5f * (ux * ux + uy * uy));
        float fi = f[idx * Q + i];
        fi = fi - omega * (fi - feq);
        
        // Streaming step
        int nx = x + cx[i];
        int ny = y + cy[i];
        
        // Periodic boundary conditions
        if (nx < 0) nx += width;
        if (nx >= width) nx -= width;
        if (ny < 0) ny += height;
        if (ny >= height) ny -= height;
        
        int nidx = ny * width + nx;
        f_new[nidx * Q + i] = fi;
    }
}
```

### Smoothed Particle Hydrodynamics (SPH)

SPH is a meshless method that represents fluid as a set of particles. GPU implementation requires efficient neighbor search algorithms:

```cuda
// Example: SPH density and force computation
__global__ void compute_density(
    float* positions,    // Particle positions [num_particles x 3]
    float* densities,    // Output densities [num_particles]
    int* cell_start,     // Grid cell start indices
    int* cell_end,       // Grid cell end indices
    int* grid_cells,     // Grid cell indices for each particle
    float h,             // Smoothing length
    float mass,          // Particle mass
    int num_particles,
    int3 grid_size,
    float3 grid_min,
    float cell_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float3 pos_i = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    float h2 = h * h;
    float density = 0.0f;
    
    // Get grid cell for this particle
    int3 cell = make_int3(
        (pos_i.x - grid_min.x) / cell_size,
        (pos_i.y - grid_min.y) / cell_size,
        (pos_i.z - grid_min.z) / cell_size
    );
    
    // Loop over neighboring cells
    for (int z = max(0, cell.z - 1); z <= min(grid_size.z - 1, cell.z + 1); z++) {
        for (int y = max(0, cell.y - 1); y <= min(grid_size.y - 1, cell.y + 1); y++) {
            for (int x = max(0, cell.x - 1); x <= min(grid_size.x - 1, cell.x + 1); x++) {
                int cell_idx = z * grid_size.y * grid_size.x + y * grid_size.x + x;
                
                // Loop over particles in this cell
                for (int j = cell_start[cell_idx]; j < cell_end[cell_idx]; j++) {
                    float3 pos_j = make_float3(positions[j*3], positions[j*3+1], positions[j*3+2]);
                    float3 r = pos_i - pos_j;
                    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
                    
                    if (r2 < h2) {
                        // Cubic spline kernel
                        float q = sqrtf(r2) / h;
                        float kernel = 0.0f;
                        
                        if (q <= 1.0f) {
                            if (q <= 0.5f) {
                                float q2 = q * q;
                                float q3 = q2 * q;
                                kernel = 6.0f * (q3 - q2) + 1.0f;
                            } else {
                                float tmp = 1.0f - q;
                                kernel = 2.0f * tmp * tmp * tmp;
                            }
                            kernel *= 8.0f / (3.14159f * h2 * h);
                            density += mass * kernel;
                        }
                    }
                }
            }
        }
    }
    
    densities[i] = density;
}

__global__ void compute_forces(
    float* positions,    // Particle positions
    float* velocities,   // Particle velocities
    float* densities,    // Particle densities
    float* forces,       // Output forces
    int* cell_start,     // Grid cell start indices
    int* cell_end,       // Grid cell end indices
    float h,             // Smoothing length
    float mass,          // Particle mass
    float rho0,          // Rest density
    float pressure_k,    // Pressure constant
    float visc_mu,       // Viscosity constant
    int num_particles,
    int3 grid_size,
    float3 grid_min,
    float cell_size,
    float3 gravity
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float3 pos_i = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    float3 vel_i = make_float3(velocities[i*3], velocities[i*3+1], velocities[i*3+2]);
    float rho_i = densities[i];
    float pressure_i = pressure_k * (rho_i - rho0);
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float h2 = h * h;
    
    // Get grid cell for this particle
    int3 cell = make_int3(
        (pos_i.x - grid_min.x) / cell_size,
        (pos_i.y - grid_min.y) / cell_size,
        (pos_i.z - grid_min.z) / cell_size
    );
    
    // Loop over neighboring cells (similar to density computation)
    // ...
    
    // Add gravity
    force += gravity * rho_i;
    
    forces[i*3] = force.x;
    forces[i*3+1] = force.y;
    forces[i*3+2] = force.z;
}
```

### Computational Fluid Dynamics (CFD)

For grid-based CFD methods, GPUs can accelerate the solution of the Navier-Stokes equations:

```cuda
// Example: 3D Navier-Stokes solver (pressure projection method)
__global__ void advection_kernel(
    float* velocity_x,
    float* velocity_y,
    float* velocity_z,
    float* velocity_x_new,
    float* velocity_y_new,
    float* velocity_z_new,
    int nx, int ny, int nz,
    float dx, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    // Semi-Lagrangian advection
    float x = i * dx;
    float y = j * dx;
    float z = k * dx;
    
    // Trace particle back along velocity field
    float u = velocity_x[idx];
    float v = velocity_y[idx];
    float w = velocity_z[idx];
    
    float x_back = x - dt * u;
    float y_back = y - dt * v;
    float z_back = z - dt * w;
    
    // Convert to grid indices
    float i_back = x_back / dx;
    float j_back = y_back / dx;
    float k_back = z_back / dx;
    
    // Interpolate velocity at backtraced position
    // (Trilinear interpolation code omitted for brevity)
    // ...
    
    velocity_x_new[idx] = interpolated_u;
    velocity_y_new[idx] = interpolated_v;
    velocity_z_new[idx] = interpolated_w;
}

__global__ void pressure_solve_jacobi_iteration(
    float* pressure,
    float* pressure_new,
    float* divergence,
    int* cell_type,  // 0 for fluid, 1 for solid
    int nx, int ny, int nz,
    float dx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    if (cell_type[idx] == 1) {
        pressure_new[idx] = pressure[idx];
        return;
    }
    
    // Count fluid neighbors
    int num_fluid_neighbors = 0;
    float neighbor_sum = 0.0f;
    
    if (i > 0 && cell_type[idx - 1] == 0) {
        neighbor_sum += pressure[idx - 1];
        num_fluid_neighbors++;
    }
    if (i < nx - 1 && cell_type[idx + 1] == 0) {
        neighbor_sum += pressure[idx + 1];
        num_fluid_neighbors++;
    }
    // Similar for j and k neighbors
    // ...
    
    if (num_fluid_neighbors > 0) {
        pressure_new[idx] = (neighbor_sum - dx * dx * divergence[idx]) / num_fluid_neighbors;
    } else {
        pressure_new[idx] = 0.0f;
    }
}

__global__ void velocity_projection(
    float* velocity_x,
    float* velocity_y,
    float* velocity_z,
    float* pressure,
    int* cell_type,
    int nx, int ny, int nz,
    float dx, float dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = k * nx * ny + j * nx + i;
    
    if (cell_type[idx] == 1) return; // Skip solid cells
    
    // Compute pressure gradient
    float grad_p_x = 0.0f;
    float grad_p_y = 0.0f;
    float grad_p_z = 0.0f;
    
    if (i > 0 && i < nx - 1) {
        grad_p_x = (pressure[idx + 1] - pressure[idx - 1]) / (2.0f * dx);
    }
    // Similar for y and z gradients
    // ...
    
    // Project velocity
    velocity_x[idx] -= dt * grad_p_x;
    velocity_y[idx] -= dt * grad_p_y;
    velocity_z[idx] -= dt * grad_p_z;
}
```

## Molecular Dynamics

Molecular Dynamics (MD) simulations model the physical movements of atoms and molecules. GPUs can dramatically accelerate these simulations, which involve computing forces between large numbers of particles.

### Particle Interaction Computation

The most computationally intensive part of MD is computing interactions between particles:

```cuda
// Example: Lennard-Jones potential force calculation
__global__ void compute_forces_lj(
    float4* positions,     // Positions and types (x,y,z,type)
    float4* forces,        // Forces and potential energy (fx,fy,fz,pe)
    float* parameters,     // Force field parameters
    int* neighbor_list,    // Neighbor list
    int* neighbor_count,   // Number of neighbors per particle
    int max_neighbors,
    float cutoff_squared,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos_i = positions[i];
    float type_i = pos_i.w;
    
    float fx = 0.0f, fy = 0.0f, fz = 0.0f, pe = 0.0f;
    
    int num_neighbors = neighbor_count[i];
    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_list[i * max_neighbors + n];
        float4 pos_j = positions[j];
        float type_j = pos_j.w;
        
        // Compute distance
        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < cutoff_squared) {
            // Get Lennard-Jones parameters for this pair
            int param_idx = (int)(type_i * num_types + type_j);
            float epsilon = parameters[param_idx * 2];
            float sigma = parameters[param_idx * 2 + 1];
            
            // Compute Lennard-Jones force
            float sigma2 = sigma * sigma;
            float r2i = sigma2 / r2;
            float r6i = r2i * r2i * r2i;
            float force = 48.0f * epsilon * r6i * (r6i - 0.5f) / r2;
            
            // Accumulate force
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
            
            // Compute potential energy
            pe += 4.0f * epsilon * r6i * (r6i - 1.0f);
        }
    }
    
    // Store computed force and energy
    forces[i] = make_float4(fx, fy, fz, pe);
}
```

### Neighbor List Construction

Efficient neighbor list construction is crucial for MD performance:

```cuda
// Example: Cell-based neighbor list construction
__global__ void build_cell_list(
    float4* positions,
    int* cell_list,
    int* particle_cell,
    int* cell_count,
    int num_particles,
    float3 box_min,
    float3 box_size,
    int3 cell_dims,
    float cell_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos = positions[i];
    
    // Compute cell index
    int x = (pos.x - box_min.x) / cell_size;
    int y = (pos.y - box_min.y) / cell_size;
    int z = (pos.z - box_min.z) / cell_size;
    
    // Clamp to valid range
    x = max(0, min(cell_dims.x - 1, x));
    y = max(0, min(cell_dims.y - 1, y));
    z = max(0, min(cell_dims.z - 1, z));
    
    int cell_idx = z * cell_dims.y * cell_dims.x + y * cell_dims.x + x;
    
    // Store cell index for this particle
    particle_cell[i] = cell_idx;
    
    // Add particle to cell (using atomic operation)
    int idx = atomicAdd(&cell_count[cell_idx], 1);
    cell_list[cell_idx * max_particles_per_cell + idx] = i;
}

__global__ void build_neighbor_list(
    float4* positions,
    int* cell_list,
    int* cell_count,
    int* neighbor_list,
    int* neighbor_count,
    int max_particles_per_cell,
    int max_neighbors,
    float cutoff_squared,
    int num_particles,
    int3 cell_dims
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos_i = positions[i];
    int count = 0;
    
    // Get cell for this particle
    int cell_idx = particle_cell[i];
    int cell_x = cell_idx % cell_dims.x;
    int cell_y = (cell_idx / cell_dims.x) % cell_dims.y;
    int cell_z = cell_idx / (cell_dims.x * cell_dims.y);
    
    // Loop over neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        int z = cell_z + dz;
        if (z < 0 || z >= cell_dims.z) continue;
        
        for (int dy = -1; dy <= 1; dy++) {
            int y = cell_y + dy;
            if (y < 0 || y >= cell_dims.y) continue;
            
            for (int dx = -1; dx <= 1; dx++) {
                int x = cell_x + dx;
                if (x < 0 || x >= cell_dims.x) continue;
                
                int neighbor_cell = z * cell_dims.y * cell_dims.x + y * cell_dims.x + x;
                int cell_particles = cell_count[neighbor_cell];
                
                // Loop over particles in this cell
                for (int p = 0; p < cell_particles; p++) {
                    int j = cell_list[neighbor_cell * max_particles_per_cell + p];
                    
                    if (i == j) continue; // Skip self
                    
                    float4 pos_j = positions[j];
                    
                    // Compute distance
                    float dx = pos_i.x - pos_j.x;
                    float dy = pos_i.y - pos_j.y;
                    float dz = pos_i.z - pos_j.z;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    
                    if (r2 < cutoff_squared) {
                        if (count < max_neighbors) {
                            neighbor_list[i * max_neighbors + count] = j;
                            count++;
                        }
                    }
                }
            }
        }
    }
    
    neighbor_count[i] = count;
}
```

### Integration

Velocity Verlet integration is commonly used in MD simulations:

```cuda
// Example: Velocity Verlet integration
__global__ void velocity_verlet_step1(
    float4* positions,
    float4* velocities,
    float4* forces,
    float dt,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 pos = positions[i];
    float4 vel = velocities[i];
    float4 force = forces[i];
    float mass = 1.0f; // Assuming unit mass for simplicity
    
    // Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    pos.x += vel.x * dt + 0.5f * force.x * dt * dt / mass;
    pos.y += vel.y * dt + 0.5f * force.y * dt * dt / mass;
    pos.z += vel.z * dt + 0.5f * force.z * dt * dt / mass;
    
    // Update velocity (half step): v(t+dt/2) = v(t) + 0.5*a(t)*dt
    vel.x += 0.5f * force.x * dt / mass;
    vel.y += 0.5f * force.y * dt / mass;
    vel.z += 0.5f * force.z * dt / mass;
    
    positions[i] = pos;
    velocities[i] = vel;
}

__global__ void velocity_verlet_step2(
    float4* velocities,
    float4* forces,
    float dt,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    
    float4 vel = velocities[i];
    float4 force = forces[i];
    float mass = 1.0f;
    
    // Update velocity (second half step): v(t+dt) = v(t+dt/2) + 0.5*a(t+dt)*dt
    vel.x += 0.5f * force.x * dt / mass;
    vel.y += 0.5f * force.y * dt / mass;
    vel.z += 0.5f * force.z * dt / mass;
    
    velocities[i] = vel;
}
```

## High-Precision Scientific Computing Considerations

Scientific computing often requires high numerical precision, which presents challenges for GPU implementations.

### Double Precision Performance

Modern GPUs support double precision, but performance varies significantly between models:

```cuda
// Example: Double precision matrix multiplication
__global__ void dgemm_kernel(
    double* A,
    double* B,
    double* C,
    int m, int n, int k
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}
```

### Mixed Precision Techniques

Mixed precision techniques can improve performance while maintaining accuracy:

```cuda
// Example: Mixed precision iterative refinement for linear systems
void mixed_precision_solver(
    double* A, double* b, double* x, int n,
    double tolerance, int max_iterations
) {
    // Allocate memory
    double *r_dp, *z_dp;
    float *A_sp, *r_sp, *z_sp;
    
    // Convert double precision matrix to single precision
    for (int i = 0; i < n * n; i++) {
        A_sp[i] = (float)A[i];
    }
    
    // Initial residual: r = b - A*x
    matrix_vector_multiply(A, x, r_dp, n);
    vector_subtract(b, r_dp, r_dp, n);
    
    double residual_norm = vector_norm(r_dp, n);
    double initial_norm = residual_norm;
    
    for (int iter = 0; iter < max_iterations && residual_norm > tolerance * initial_norm; iter++) {
        // Convert residual to single precision
        for (int i = 0; i < n; i++) {
            r_sp[i] = (float)r_dp[i];
        }
        
        // Solve A_sp * z_sp = r_sp in single precision
        solve_linear_system_sp(A_sp, r_sp, z_sp, n);
        
        // Convert correction to double precision
        for (int i = 0; i < n; i++) {
            z_dp[i] = (double)z_sp[i];
        }
        
        // Update solution: x = x + z
        vector_add(x, z_dp, x, n);
        
        // Compute new residual: r = b - A*x
        matrix_vector_multiply(A, x, r_dp, n);
        vector_subtract(b, r_dp, r_dp, n);
        
        residual_norm = vector_norm(r_dp, n);
    }
}
```

### Reproducibility Challenges

Ensuring reproducible results across different runs and hardware is challenging:

```cuda
// Example: Reproducible summation using Kahan algorithm
__global__ void kahan_sum_kernel(
    float* input,
    float* output,
    int n
) {
    __shared__ float shared_sum[256];
    __shared__ float shared_error[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize
    float sum = 0.0f;
    float error = 0.0f;
    
    // Process multiple elements per thread
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float y = input[i] - error;
        float t = sum + y;
        error = (t - sum) - y;
        sum = t;
    }
    
    // Store in shared memory
    shared_sum[tid] = sum;
    shared_error[tid] = error;
    __syncthreads();
    
    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float y = shared_sum[tid + s] - shared_error[tid];
            float t = shared_sum[tid] + y;
            shared_error[tid] = (t - shared_sum[tid]) - y;
            shared_sum[tid] = t;
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}
```

## Conclusion

GPU computing has revolutionized scientific simulations by providing massive parallelism for computationally intensive problems. From finite element methods to fluid dynamics and molecular dynamics, GPUs enable researchers to tackle larger problems and achieve results faster than ever before.

Key takeaways from this article include:

1. **Finite Element Methods**: GPUs can accelerate element matrix computation, assembly, and linear system solution
2. **Fluid Dynamics**: Methods like Lattice Boltzmann and SPH are well-suited for GPU implementation
3. **Molecular Dynamics**: GPUs excel at computing particle interactions and neighbor list construction
4. **High-Precision Computing**: Considerations for double precision, mixed precision, and reproducibility

In our next article, we'll explore real-time signal processing on GPUs, focusing on audio, image, and video processing algorithms.

## Exercises for Practice

1. **FEM Implementation**: Implement a simple 2D heat transfer simulation using the finite element method on a GPU.

2. **Fluid Simulation**: Create a 2D Lattice Boltzmann fluid simulation and visualize the results in real-time.

3. **Molecular Dynamics**: Implement a basic molecular dynamics simulation of a Lennard-Jones fluid and measure the performance compared to a CPU implementation.

4. **Mixed Precision**: Experiment with mixed precision techniques in a linear system solver and analyze the trade-offs between performance and accuracy.

5. **Optimization Challenge**: Take an existing scientific simulation code and optimize it for GPU execution, measuring the speedup achieved.

## Further Resources

- [NVIDIA SimNet](https://developer.nvidia.com/simnet) - Physics-informed neural networks for scientific computing
- [AmgX](https://developer.nvidia.com/amgx) - GPU-accelerated algebraic multigrid solver
- [GROMACS](http://www.gromacs.org/) - GPU-accelerated molecular dynamics package
- [LAMMPS](https://lammps.sandia.gov/) - Large-scale atomic/molecular massively parallel simulator
- [OpenFOAM](https://www.openfoam.com/) - Open-source CFD software with GPU support