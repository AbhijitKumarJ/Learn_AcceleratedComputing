# Raytracing on GPUs

*Welcome to the fourteenth installment of our GPU programming series! In this article, we'll explore raytracing on GPUs, including the fundamentals of raytracing, hardware acceleration with RTX, implementing a basic raytracer, and hybrid rendering techniques.*

## Introduction to Raytracing

Raytracing is a rendering technique that simulates the physical behavior of light to create highly realistic images. Unlike traditional rasterization, which projects 3D objects onto a 2D screen, raytracing works by tracing the path of light rays as they interact with virtual objects.

The basic algorithm is conceptually simple:
1. Cast rays from the camera through each pixel of the image plane
2. Determine which objects the rays intersect with
3. Calculate the color at the intersection points based on material properties and light sources
4. Recursively trace additional rays for reflections, refractions, and shadows

## RTX and Hardware-Accelerated Raytracing

NVIDIA's RTX technology introduced dedicated hardware (RT cores) for accelerating raytracing operations. These cores specialize in ray-triangle intersection tests and bounding volume hierarchy (BVH) traversal, which are the most computationally intensive parts of raytracing.

### Ray Tracing Cores

RT cores accelerate two key operations:

1. **BVH Traversal**: Efficiently finding which objects might be intersected by a ray
2. **Ray-Triangle Intersection**: Determining exactly where a ray hits a triangle

### DirectX Raytracing (DXR)

Microsoft's DirectX Raytracing API provides a standardized interface for hardware-accelerated raytracing:

```cpp
// Example: Simple DXR raytracing pipeline setup
// Note: This is a simplified example showing the key components

// Create acceleration structures
D3D12_RAYTRACING_GEOMETRY_DESC geometryDesc = {};
geometryDesc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
geometryDesc.Triangles.VertexBuffer.StartAddress = vertexBuffer->GetGPUVirtualAddress();
geometryDesc.Triangles.VertexBuffer.StrideInBytes = sizeof(Vertex);
geometryDesc.Triangles.VertexCount = vertexCount;
geometryDesc.Triangles.IndexBuffer = indexBuffer->GetGPUVirtualAddress();
geometryDesc.Triangles.IndexCount = indexCount;
geometryDesc.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
geometryDesc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

// Build bottom-level acceleration structure (BLAS)
D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bottomLevelBuildDesc = {};
bottomLevelBuildDesc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
bottomLevelBuildDesc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
bottomLevelBuildDesc.Inputs.NumDescs = 1;
bottomLevelBuildDesc.Inputs.pGeometryDescs = &geometryDesc;
// ... additional setup ...

commandList->BuildRaytracingAccelerationStructure(&bottomLevelBuildDesc, 0, nullptr);

// Build top-level acceleration structure (TLAS)
D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc = {};
// ... setup similar to BLAS but with instance descriptions ...

commandList->BuildRaytracingAccelerationStructure(&topLevelBuildDesc, 0, nullptr);

// Create ray generation shader and hit groups
// ... shader compilation code ...

// Create raytracing pipeline state object
D3D12_STATE_OBJECT_DESC pipelineDesc = {};
pipelineDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
// ... add subobjects for shaders, hit groups, etc. ...

ID3D12StateObject* rtpso;
device->CreateStateObject(&pipelineDesc, IID_PPV_ARGS(&rtpso));

// Dispatch rays
D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
// ... setup dispatch parameters ...

commandList->SetPipelineState1(rtpso);
commandList->DispatchRays(&dispatchDesc);
```

### NVIDIA OptiX

NVIDIA's OptiX is a high-level API specifically designed for raytracing applications:

```cpp
// Example: Basic OptiX 7 raytracing setup

// Initialize OptiX
OPTIX_CHECK(optixInit());

// Create context
OptixDeviceContext context;
CUDA_CHECK(cudaSetDevice(0));
CUcontext cuContext = 0;
OPTIX_CHECK(optixDeviceContextCreate(cuContext, 0, &context));

// Create module from PTX code
OptixModule module;
OptixPipelineCompileOptions pipelineCompileOptions = {};
// ... set options ...

OPTIX_CHECK(optixModuleCreateFromPTX(context, &moduleCompileOptions,
                                    &pipelineCompileOptions,
                                    ptxCode, ptxSize, nullptr, nullptr, &module));

// Create program groups (ray generation, miss, hit)
OptixProgramGroup raygenPG, missPG, hitPG;
// ... create program groups ...

// Create pipeline
OptixPipeline pipeline;
OptixPipelineLinkOptions pipelineLinkOptions = {};
// ... set options ...

OptixProgramGroup programGroups[] = { raygenPG, missPG, hitPG };
OPTIX_CHECK(optixPipelineCreate(context, &pipelineCompileOptions,
                               &pipelineLinkOptions, programGroups,
                               sizeof(programGroups) / sizeof(programGroups[0]),
                               nullptr, nullptr, &pipeline));

// Build acceleration structure
OptixTraversableHandle gas_handle;
// ... build acceleration structure ...

// Launch parameters
Params params;
// ... set parameters ...

CUdeviceptr d_params;
CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(params), cudaMemcpyHostToDevice));

// Launch rays
OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height, 1));
```

## Implementing a Basic Raytracer

Let's implement a simple CUDA-based raytracer without hardware acceleration to understand the core concepts:

```cuda
// Basic CUDA raytracer

// Structures for scene representation
struct Ray {
    float3 origin;
    float3 direction;
    
    __device__ Ray(float3 o, float3 d) : origin(o), direction(d) {
        direction = normalize(direction);
    }
    
    __device__ float3 at(float t) const {
        return origin + t * direction;
    }
};

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    float specular;
    float reflective;
    
    __device__ bool intersect(const Ray& ray, float& t) const {
        float3 oc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(oc, ray.direction);
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0) return false;
        
        float sqrtd = sqrtf(discriminant);
        float t1 = (-b - sqrtd) / (2.0f * a);
        float t2 = (-b + sqrtd) / (2.0f * a);
        
        if (t1 > 0.001f) {
            t = t1;
            return true;
        }
        
        if (t2 > 0.001f) {
            t = t2;
            return true;
        }
        
        return false;
    }
    
    __device__ float3 normal(const float3& point) const {
        return normalize(point - center);
    }
};

struct Light {
    float3 position;
    float intensity;
};

// Global scene data
__constant__ Sphere d_spheres[10];
__constant__ Light d_lights[5];
__constant__ int d_sphere_count;
__constant__ int d_light_count;
__constant__ float3 d_ambient_light;
__constant__ float3 d_background_color;

// Compute color for a ray
__device__ float3 trace_ray(const Ray& ray, int depth) {
    if (depth <= 0) return make_float3(0.0f);
    
    float closest_t = FLT_MAX;
    int closest_sphere = -1;
    
    // Find closest intersection
    for (int i = 0; i < d_sphere_count; i++) {
        float t;
        if (d_spheres[i].intersect(ray, t) && t < closest_t) {
            closest_t = t;
            closest_sphere = i;
        }
    }
    
    if (closest_sphere == -1) return d_background_color;
    
    // Compute intersection point and normal
    float3 point = ray.at(closest_t);
    float3 normal = d_spheres[closest_sphere].normal(point);
    float3 view = -ray.direction;
    
    // Start with ambient lighting
    float3 color = d_ambient_light * d_spheres[closest_sphere].color;
    
    // Add contribution from each light
    for (int i = 0; i < d_light_count; i++) {
        float3 light_dir = normalize(d_lights[i].position - point);
        
        // Shadow check
        bool in_shadow = false;
        Ray shadow_ray(point + 0.001f * normal, light_dir);
        
        for (int j = 0; j < d_sphere_count; j++) {
            float t;
            if (d_spheres[j].intersect(shadow_ray, t)) {
                in_shadow = true;
                break;
            }
        }
        
        if (!in_shadow) {
            // Diffuse lighting
            float diffuse = max(0.0f, dot(normal, light_dir));
            
            // Specular lighting
            float3 reflect_dir = 2.0f * dot(normal, light_dir) * normal - light_dir;
            float specular = powf(max(0.0f, dot(view, reflect_dir)), d_spheres[closest_sphere].specular);
            
            color += d_lights[i].intensity * (
                d_spheres[closest_sphere].color * diffuse +
                make_float3(1.0f) * specular
            );
        }
    }
    
    // Compute reflection
    float reflective = d_spheres[closest_sphere].reflective;
    if (reflective > 0.0f && depth > 0) {
        float3 reflect_dir = 2.0f * dot(normal, view) * normal - view;
        Ray reflect_ray(point + 0.001f * normal, reflect_dir);
        float3 reflect_color = trace_ray(reflect_ray, depth - 1);
        color = color * (1.0f - reflective) + reflect_color * reflective;
    }
    
    return color;
}

// Kernel to render the image
__global__ void render_kernel(float3* output, int width, int height, float fov) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float aspect_ratio = (float)width / height;
    float scale = tanf(fov * 0.5f * M_PI / 180.0f);
    
    // Compute ray direction
    float u = (2.0f * ((x + 0.5f) / width) - 1.0f) * scale * aspect_ratio;
    float v = (1.0f - 2.0f * ((y + 0.5f) / height)) * scale;
    
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = normalize(make_float3(u, v, -1.0f));
    
    Ray ray(origin, direction);
    float3 color = trace_ray(ray, 3); // Max recursion depth of 3
    
    // Clamp and write output
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);
    
    output[y * width + x] = color;
}

// Host function to set up and launch the kernel
void render_scene(float3* output, int width, int height) {
    // Set up scene (spheres, lights, etc.)
    Sphere spheres[3];
    spheres[0] = {make_float3(0.0f, 0.0f, -5.0f), 1.0f, make_float3(1.0f, 0.0f, 0.0f), 50.0f, 0.2f}; // Red
    spheres[1] = {make_float3(2.0f, 0.0f, -6.0f), 1.0f, make_float3(0.0f, 1.0f, 0.0f), 10.0f, 0.4f}; // Green
    spheres[2] = {make_float3(-2.0f, 0.0f, -4.0f), 1.0f, make_float3(0.0f, 0.0f, 1.0f), 100.0f, 0.6f}; // Blue
    
    Light lights[2];
    lights[0] = {make_float3(5.0f, 5.0f, 0.0f), 0.6f};
    lights[1] = {make_float3(-5.0f, 3.0f, 0.0f), 0.4f};
    
    int sphere_count = 3;
    int light_count = 2;
    float3 ambient_light = make_float3(0.1f);
    float3 background_color = make_float3(0.2f, 0.3f, 0.4f);
    
    // Copy scene data to device
    cudaMemcpyToSymbol(d_spheres, spheres, sphere_count * sizeof(Sphere));
    cudaMemcpyToSymbol(d_lights, lights, light_count * sizeof(Light));
    cudaMemcpyToSymbol(d_sphere_count, &sphere_count, sizeof(int));
    cudaMemcpyToSymbol(d_light_count, &light_count, sizeof(int));
    cudaMemcpyToSymbol(d_ambient_light, &ambient_light, sizeof(float3));
    cudaMemcpyToSymbol(d_background_color, &background_color, sizeof(float3));
    
    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                  (height + block_size.y - 1) / block_size.y);
    
    render_kernel<<<grid_size, block_size>>>(output, width, height, 60.0f); // 60 degree FOV
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
}
```

## Acceleration Structures

Efficient raytracing requires acceleration structures to reduce the number of ray-object intersection tests:

### Bounding Volume Hierarchy (BVH)

A BVH organizes scene objects in a tree structure, with each node containing a bounding volume that encloses all objects in its subtree:

```cpp
// Example: Simple BVH node structure
struct BVHNode {
    float3 min_bounds;
    float3 max_bounds;
    int left_child;  // Index of left child or first primitive if leaf
    int right_child; // Index of right child or number of primitives if leaf
    bool is_leaf;
};

// Example: Ray-AABB intersection test
__device__ bool intersect_aabb(const Ray& ray, const float3& min_bounds, const float3& max_bounds,
                             float& t_min, float& t_max) {
    float3 inv_dir = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    float3 t0s = (min_bounds - ray.origin) * inv_dir;
    float3 t1s = (max_bounds - ray.origin) * inv_dir;
    
    float3 tsmaller = fminf(t0s, t1s);
    float3 tbigger = fmaxf(t0s, t1s);
    
    t_min = fmaxf(fmaxf(tsmaller.x, tsmaller.y), fmaxf(tsmaller.z, t_min));
    t_max = fminf(fminf(tbigger.x, tbigger.y), fminf(tbigger.z, t_max));
    
    return t_min <= t_max;
}

// Example: BVH traversal
__device__ bool trace_bvh(const Ray& ray, const BVHNode* nodes, const Triangle* triangles,
                        HitInfo& hit_info) {
    // Stack-based traversal
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; // Start with root node
    
    bool hit_anything = false;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = nodes[node_idx];
        
        float t_min = 0.001f;
        float t_max = hit_info.t;
        
        if (!intersect_aabb(ray, node.min_bounds, node.max_bounds, t_min, t_max)) {
            continue; // No intersection with this node's bounding box
        }
        
        if (node.is_leaf) {
            // Test all triangles in this leaf
            int prim_count = node.right_child;
            int prim_idx = node.left_child;
            
            for (int i = 0; i < prim_count; i++) {
                const Triangle& tri = triangles[prim_idx + i];
                float t;
                float2 bary;
                
                if (intersect_triangle(ray, tri, t, bary) && t < hit_info.t) {
                    hit_info.t = t;
                    hit_info.triangle_idx = prim_idx + i;
                    hit_info.bary = bary;
                    hit_anything = true;
                }
            }
        } else {
            // Add children to stack
            // Add the further child first (depth-first traversal optimization)
            float3 center = (node.min_bounds + node.max_bounds) * 0.5f;
            bool hit_left_first = (ray.origin.x <= center.x);
            
            if (hit_left_first) {
                stack[stack_ptr++] = node.right_child;
                stack[stack_ptr++] = node.left_child;
            } else {
                stack[stack_ptr++] = node.left_child;
                stack[stack_ptr++] = node.right_child;
            }
        }
    }
    
    return hit_anything;
}
```

## Hybrid Rendering Techniques

Hybrid rendering combines raytracing with traditional rasterization to balance quality and performance:

### Deferred Raytracing

Deferred raytracing uses rasterization for primary visibility and raytracing for specific effects:

```cpp
// Example: Deferred raytracing pipeline
void hybrid_render_frame() {
    // 1. Rasterize scene to G-buffer (positions, normals, materials)
    rasterize_gbuffer();
    
    // 2. Use raytracing for reflections
    compute_raytraced_reflections();
    
    // 3. Use raytracing for shadows
    compute_raytraced_shadows();
    
    // 4. Combine results in a final lighting pass
    compute_final_lighting();
}

// Example: Reflection ray generation kernel
__global__ void reflection_ray_kernel(float4* positions, float4* normals, float4* materials,
                                    float4* reflection_result, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    
    float4 position = positions[idx];
    if (position.w == 0.0f) return; // Sky pixel
    
    float4 normal = normals[idx];
    float4 material = materials[idx];
    
    float reflectivity = material.w;
    if (reflectivity <= 0.0f) {
        reflection_result[idx] = make_float4(0.0f);
        return;
    }
    
    // Create reflection ray
    float3 view_dir = normalize(make_float3(-position.x, -position.y, -position.z));
    float3 reflect_dir = reflect(-view_dir, make_float3(normal));
    
    Ray ray;
    ray.origin = make_float3(position) + 0.001f * make_float3(normal);
    ray.direction = reflect_dir;
    
    // Trace reflection ray through acceleration structure
    float3 color = trace_scene(ray);
    
    reflection_result[idx] = make_float4(color, reflectivity);
}
```

### Ray Traced Ambient Occlusion

Ray traced ambient occlusion (RTAO) improves the realism of indirect lighting:

```cuda
// Example: Ray traced ambient occlusion kernel
__global__ void rtao_kernel(float4* positions, float4* normals, float* ao_result,
                          int width, int height, int samples_per_pixel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    
    float4 position = positions[idx];
    if (position.w == 0.0f) return; // Sky pixel
    
    float4 normal = normals[idx];
    float3 n = make_float3(normal);
    
    // Create tangent space basis
    float3 tangent, bitangent;
    create_orthonormal_basis(n, tangent, bitangent);
    
    // Trace ambient occlusion rays
    float occlusion = 0.0f;
    float max_distance = 1.0f;
    
    // Use a fixed seed for reproducibility
    unsigned int seed = (y * width + x) * 719393;
    
    for (int i = 0; i < samples_per_pixel; i++) {
        // Generate random direction in hemisphere
        float r1 = random_float(seed);
        float r2 = random_float(seed);
        float3 sample_dir = sample_hemisphere_cosine(r1, r2);
        
        // Transform to world space
        float3 world_dir = sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * n;
        
        // Create and trace ray
        Ray ray;
        ray.origin = make_float3(position) + 0.001f * n;
        ray.direction = world_dir;
        
        // Check for occlusion
        bool occluded = trace_occlusion(ray, max_distance);
        if (occluded) {
            occlusion += 1.0f;
        }
    }
    
    // Normalize result
    ao_result[idx] = 1.0f - (occlusion / samples_per_pixel);
}
```

## Conclusion

Raytracing on GPUs represents a significant advancement in real-time rendering, enabling effects that were previously only possible in offline rendering. With dedicated hardware acceleration, developers can now incorporate physically accurate lighting, reflections, shadows, and global illumination into interactive applications.

Key takeaways from this article include:

1. **Raytracing Fundamentals**: Understanding the basic principles of ray casting, intersection testing, and shading
2. **Hardware Acceleration**: Leveraging RT cores and APIs like DXR and OptiX for real-time performance
3. **Acceleration Structures**: Using BVHs to efficiently organize scene geometry
4. **Hybrid Techniques**: Combining rasterization and raytracing for optimal performance and quality

In our next article, we'll explore GPU computing for scientific simulations, focusing on how GPUs accelerate complex physical simulations across various scientific domains.

## Exercises for Practice

1. **Basic Raytracer**: Implement the CUDA raytracer from this article and extend it to support textured materials.

2. **BVH Construction**: Implement a simple BVH construction algorithm on the CPU and use it to accelerate your raytracer.

3. **DXR Integration**: If you have an RTX GPU, integrate DirectX Raytracing into a simple rendering application.

4. **Hybrid Renderer**: Create a hybrid renderer that uses rasterization for primary visibility and raytracing for reflections.

5. **Path Tracer**: Extend the basic raytracer to a path tracer by implementing Monte Carlo integration for global illumination.

## Further Resources

- [NVIDIA RTX Developer Resources](https://developer.nvidia.com/rtx)
- [DirectX Raytracing Specification](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)
- [NVIDIA OptiX Documentation](https://developer.nvidia.com/optix)
- [Ray Tracing Gems (Free Book)](http://www.realtimerendering.com/raytracinggems/)
- [Physically Based Rendering: From Theory to Implementation](http://www.pbr-book.org/)