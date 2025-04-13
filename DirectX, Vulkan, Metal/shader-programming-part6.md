# Part 6: Shader Programming Across Modern Graphics APIs

## Introduction
Shader programming is a crucial skill for modern graphics development. This guide explores shader languages across DirectX (HLSL), Vulkan (GLSL/SPIR-V), and Metal (MSL), highlighting their similarities and differences while providing practical examples.

## Shader Language Comparison

### HLSL (High-Level Shading Language)
```hlsl
struct VSInput {
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
};

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
};

PSInput VSMain(VSInput input) {
    PSInput output;
    output.position = mul(float4(input.position, 1.0f), worldViewProj);
    output.texCoord = input.texCoord;
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET {
    return diffuseTexture.Sample(samplerState, input.texCoord);
}
```

### GLSL (OpenGL/Vulkan Shading Language)
```glsl
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 outTexCoord;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 worldViewProj;
} ubo;

void main() {
    gl_Position = ubo.worldViewProj * vec4(inPosition, 1.0);
    outTexCoord = inTexCoord;
}
```

### MSL (Metal Shading Language)
```metal
struct VertexInput {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOutput {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOutput vertexShader(
    VertexInput input [[stage_in]],
    constant matrix_float4x4& worldViewProj [[buffer(0)]]
) {
    VertexOutput output;
    output.position = worldViewProj * float4(input.position, 1.0);
    output.texCoord = input.texCoord;
    return output;
}
```

## Cross-API Shader Development

### Common Patterns and Best Practices
1. **Resource Binding**
   - DirectX: Descriptor tables and root signatures
   - Vulkan: Descriptor sets and bindings
   - Metal: Argument buffers

2. **Uniform Data Management**
```cpp
// DirectX
struct SceneConstants {
    XMMATRIX worldViewProj;
    XMFLOAT4 lightPosition;
    XMFLOAT4 cameraPosition;
};

// Vulkan/GLSL
layout(std140, set = 0, binding = 0) uniform SceneData {
    mat4 worldViewProj;
    vec4 lightPosition;
    vec4 cameraPosition;
} sceneData;

// Metal
struct SceneData {
    matrix_float4x4 worldViewProj;
    vector_float4 lightPosition;
    vector_float4 cameraPosition;
};
```

## Compute Shaders

### DirectX Compute Shader
```hlsl
[numthreads(8, 8, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID) {
    float4 color = inputTexture[DTid.xy];
    outputTexture[DTid.xy] = float4(1.0 - color.rgb, color.a);
}
```

### Vulkan Compute Shader
```glsl
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D inputTexture;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D outputTexture;

void main() {
    ivec2 texCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = texelFetch(inputTexture, texCoord, 0);
    imageStore(outputTexture, texCoord, vec4(1.0 - color.rgb, color.a));
}
```

### Metal Compute Shader
```metal
kernel void computeInvert(
    texture2d<float, access::read> inputTexture [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 color = inputTexture.read(gid);
    outputTexture.write(float4(1.0 - color.rgb, color.a), gid);
}
```

## Advanced Shader Techniques

### Physically Based Rendering (PBR)
```hlsl
float3 CalculatePBR(float3 albedo, float metallic, float roughness, float3 N, float3 V, float3 L) {
    float3 H = normalize(V + L);
    
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);
    
    // Calculate specular and diffuse components
    // ... PBR calculations ...
    
    return finalColor;
}
```

### Shadow Mapping
```glsl
float CalculateShadow(vec4 fragPosLightSpace, sampler2D shadowMap) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    
    float bias = 0.005;
    return currentDepth - bias > closestDepth ? 0.0 : 1.0;
}
```

## Performance Optimization

### Shader Optimization Techniques
1. **Arithmetic Optimization**
   - Use MAD (Multiply-Add) operations
   - Minimize dynamic branching
   - Utilize built-in functions

2. **Memory Access Patterns**
   - Coalesce texture reads
   - Use appropriate texture formats
   - Minimize divergent memory access

3. **Resource Management**
   - Use constant buffers efficiently
   - Optimize texture sampling
   - Manage register pressure

## Debug and Profiling

### Debugging Tools
- DirectX: PIX
- Vulkan: RenderDoc
- Metal: Metal Frame Capture

### Common Issues and Solutions
1. **Compilation Errors**
   - Syntax differences between APIs
   - Resource binding mismatches
   - Type conversion issues

2. **Runtime Issues**
   - Undefined behavior
   - Performance bottlenecks
   - Memory access violations

## Next Steps
- Explore ray tracing shaders
- Learn about mesh and amplification shaders
- Study advanced rendering techniques
- Practice cross-platform shader development

## Resources
- [Microsoft HLSL Documentation](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl)
- [Vulkan GLSL Reference](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html)
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)