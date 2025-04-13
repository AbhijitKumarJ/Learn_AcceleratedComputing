# OpenGL from the Ground Up: Part 6 - Lighting and Materials

## Introduction to Lighting in OpenGL

Before diving into implementation details, let's understand the fundamental concepts of lighting in computer graphics:

### Light Properties
1. **Color**: Represented as RGB values
2. **Intensity**: The strength of the light
3. **Position/Direction**: Where the light is coming from
4. **Attenuation**: How light fades over distance

### Surface Properties
1. **Material**: How a surface reacts to light
2. **Normal vectors**: Direction perpendicular to the surface
3. **Roughness/Shininess**: How scattered or focused reflections are

## The Phong Lighting Model in Detail

The Phong model breaks lighting into three distinct components that we calculate separately and then combine:

### 1. Ambient Lighting
Represents indirect light bouncing around the scene. This is a simple approximation of global illumination.

```glsl
// Ambient lighting calculation
float ambientStrength = 0.1;  // Usually small to avoid washing out the scene
vec3 ambient = ambientStrength * lightColor;
// This creates a baseline illumination so objects are never completely black
```

### 2. Diffuse Lighting
The main directional lighting that gives objects their basic shading. Based on the angle between the light and the surface normal.

```glsl
// Diffuse lighting calculation explained
vec3 norm = normalize(Normal);        // Ensure normal is unit length
vec3 lightDir = normalize(lightPos - FragPos);  // Direction from fragment to light
float diff = max(dot(norm, lightDir), 0.0);     // Cosine of angle between vectors
// max() ensures we don't get negative lighting
vec3 diffuse = diff * lightColor;     // Stronger light when surface faces light directly
```

### 3. Specular Lighting
Creates bright highlights on shiny surfaces, simulating direct reflection of light.

```glsl
// Specular lighting calculation with commentary
float specularStrength = 0.5;         // Adjust based on material shininess
vec3 viewDir = normalize(viewPos - FragPos);  // Direction from fragment to camera
vec3 reflectDir = reflect(-lightDir, norm);   // Light reflection vector
// The shininess power (32 here) determines how focused the highlight is
float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
vec3 specular = specularStrength * spec * lightColor;
```

## Complete Lighting Implementation

Here's the full shader implementation with detailed comments:

```glsl
// Vertex Shader
#version 330 core
layout (location = 0) in vec3 aPos;    // Vertex position attribute
layout (location = 1) in vec3 aNormal; // Normal vector attribute

// Output to fragment shader
out vec3 FragPos;   // World space fragment position
out vec3 Normal;    // World space normal vector

// Transformation matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    // Calculate fragment position in world space
    FragPos = vec3(model * vec4(aPos, 1.0));
    
    // Transform normal vector to world space
    // Using transpose(inverse(model)) to handle non-uniform scaling correctly
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    // Final vertex position in clip space
    gl_Position = projection * view * vec4(FragPos, 1.0);
}

// Fragment Shader
#version 330 core
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

// Light and material properties
struct Material {
    vec3 ambient;    // Color under ambient lighting
    vec3 diffuse;    // Main color of the material
    vec3 specular;   // Color of specular highlights
    float shininess; // Focus of specular highlights
};

struct Light {
    vec3 position;   // Light position in world space
    vec3 ambient;    // Ambient light intensity
    vec3 diffuse;    // Diffuse light intensity
    vec3 specular;   // Specular light intensity
};

uniform Material material;
uniform Light light;
uniform vec3 viewPos;  // Camera position for specular calculation

void main() {
    // 1. Ambient lighting
    vec3 ambient = light.ambient * material.ambient;

    // 2. Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * (diff * material.diffuse);

    // 3. Specular lighting
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = light.specular * (spec * material.specular);

    // Combine all lighting components
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
```

## Material System Implementation

Here's how to set up materials in your C++ code:

```cpp
// Material properties structure
struct Material {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
};

// Example materials
Material emerald = {
    glm::vec3(0.0215f, 0.1745f, 0.0215f),   // ambient
    glm::vec3(0.07568f, 0.61424f, 0.07568f), // diffuse
    glm::vec3(0.633f, 0.727811f, 0.633f),    // specular
    0.6f * 128.0f                            // shininess
};

Material gold = {
    glm::vec3(0.24725f, 0.1995f, 0.0745f),
    glm::vec3(0.75164f, 0.60648f, 0.22648f),
    glm::vec3(0.628281f, 0.555802f, 0.366065f),
    0.4f * 128.0f
};

// Setting material uniforms
shader.use();
shader.setVec3("material.ambient", material.ambient);
shader.setVec3("material.diffuse", material.diffuse);
shader.setVec3("material.specular", material.specular);
shader.setFloat("material.shininess", material.shininess);
```

## Multiple Light Sources

### Types of Lights

1. Directional Light
```glsl
struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    vec3 ambient = light.ambient * material.diffuse;
    vec3 diffuse = light.diffuse * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;
    
    return (ambient + diffuse + specular);
}
```

2. Point Light
```glsl
struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    
    float constant;
    float linear;
    float quadratic;
};

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                              light.quadratic * distance * distance);
    
    vec3 ambient = light.ambient * material.diffuse;
    vec3 diffuse = light.diffuse * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;
    
    return (ambient + diffuse + specular) * attenuation;
}
```

3. Spotlight
```glsl
struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    
    float constant;
    float linear;
    float quadratic;
};

vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                              light.quadratic * distance * distance);
    
    vec3 ambient = light.ambient * material.diffuse;
    vec3 diffuse = light.diffuse * diff * material.diffuse;
    vec3 specular = light.specular * spec * material.specular;
    
    return (ambient + diffuse + specular) * attenuation * intensity;
}
```

## Normal Mapping

### Implementing Normal Mapping

```glsl
// Vertex Shader with Tangent Space
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out VS_OUT {
    vec3 FragPos;
    vec2 TexCoords;
    mat3 TBN;
} vs_out;

void main() {
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));   
    vs_out.TexCoords = aTexCoords;
    
    vec3 T = normalize(vec3(model * vec4(aTangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(aNormal, 0.0)));
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    
    vs_out.TBN = mat3(T, B, N);
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}

// Fragment Shader with Normal Mapping
uniform sampler2D normalMap;

void main() {
    vec3 normal = texture(normalMap, TexCoords).rgb;
    normal = normal * 2.0 - 1.0;
    normal = normalize(vs_out.TBN * normal);
    
    // Use normal in lighting calculations
    // ...
}
```

### Parallax Mapping

```glsl
vec2 ParallaxMapping(vec2 texCoords, vec3 viewDir) {
    float height = texture(heightMap, texCoords).r;    
    vec2 p = viewDir.xy / viewDir.z * (height * heightScale);
    return texCoords - p;
}

// In fragment shader
vec3 viewDir = normalize(vs_out.TangentViewPos - vs_out.TangentFragPos);
vec2 texCoords = ParallaxMapping(vs_out.TexCoords, viewDir);
if(texCoords.x > 1.0 || texCoords.y > 1.0 || texCoords.x < 0.0 || texCoords.y < 0.0)
    discard;
```

## Advanced Material Effects

### PBR (Physically Based Rendering)

```glsl
// PBR Fragment Shader
#version 330 core
out vec4 FragColor;

uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];

uniform vec3 camPos;

const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(camPos - FragPos);
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);
    
    vec3 Lo = vec3(0.0);
    
    // Calculate lighting contribution from each light
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(lightPositions[i] - FragPos);
        vec3 H = normalize(V + L);
        
        float distance = length(lightPositions[i] - FragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;
        
        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);
        float G   = GeometrySmith(N, V, L, roughness);
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 nominator    = NDF * G * F;
        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        vec3 specular     = nominator / max(denominator, 0.001);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        float NdotL = max(dot(N, L), 0.0);
        
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }
    
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
    
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2)); 
    
    FragColor = vec4(color, 1.0);
}
```

## Practice Exercises

1. Implement a deferred rendering system:
```cpp
// First pass - G-Buffer generation
void firstPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Render scene to G-Buffer
}

// Second pass - Lighting
void secondPass() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    // Process G-Buffer and apply lighting
}
```

2. Create a custom material system:
```cpp
class Material {
public:
    void setProperty(const std::string& name, const glm::vec3& value) {
        properties[name] = value;
    }
    
    void bind(Shader& shader) {
        for(const auto& [name, value] : properties) {
            shader.setVec3(name.c_str(), value);
        }
    }
    
private:
    std::unordered_map<std::string, glm::vec3> properties;
};
```

## Next Steps

In Part 7, we'll explore camera systems and view transformations, including:
- View matrix calculations
- Different camera types
- Camera movement and controls
- Projection matrices

Remember to experiment with different lighting models and material combinations to achieve the desired visual effects!
