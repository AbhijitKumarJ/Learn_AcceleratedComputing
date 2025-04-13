# OpenGL from the Ground Up: Part 5 - Texturing

Welcome to Part 5 of our OpenGL tutorial series! In this installment, we'll do a deep dive into texture mapping, a fundamental technique for adding detail and realism to 3D objects.

## Texture Mapping Fundamentals

### What are Textures?

Textures are 2D or 3D data arrays that we map onto geometry to add visual detail and other properties. While most commonly used for color information (diffuse textures), modern graphics applications use textures for various purposes:

1. **Color/Diffuse Maps**
   - Store surface color information
   - Usually RGB or RGBA format
   - Most common texture type

2. **Normal Maps**
   - Store surface normal information in RGB channels
   - Red = X direction
   - Green = Y direction
   - Blue = Z direction
   - Used for adding detail without extra geometry

3. **Specular Maps**
   - Define shininess and reflectivity
   - Usually grayscale
   - Bright areas = more reflective
   - Dark areas = less reflective

4. **Height/Displacement Maps**
   - Grayscale images representing surface height
   - White = highest points
   - Black = lowest points
   - Used for parallax mapping or actual geometry displacement

5. **Ambient Occlusion Maps**
   - Pre-calculated ambient lighting information
   - Grayscale textures
   - Dark areas = more occlusion
   - Light areas = less occlusion

### Texture Coordinates (UV Coordinates)

UV coordinates (also called texture coordinates) are 2D coordinates that map points on a texture to points on a 3D model. They range from 0.0 to 1.0:

- U coordinate: horizontal axis (0 = left, 1 = right)
- V coordinate: vertical axis (0 = bottom, 1 = top)

Here's how to define vertices with texture coordinates:

```cpp
// Vertex data structure
struct Vertex {
    glm::vec3 position;    // 3D position
    glm::vec2 texCoord;    // UV coordinates
};

// Vertex data with texture coordinates for a quad
float vertices[] = {
    // positions          // texture coords
     0.5f,  0.5f, 0.0f,   1.0f, 1.0f,   // top right
     0.5f, -0.5f, 0.0f,   1.0f, 0.0f,   // bottom right
    -0.5f, -0.5f, 0.0f,   0.0f, 0.0f,   // bottom left
    -0.5f,  0.5f, 0.0f,   0.0f, 1.0f    // top left 
};

// Index data for drawing with elements
unsigned int indices[] = {
    0, 1, 3,  // first triangle
    1, 2, 3   // second triangle
};
```

## Loading and Binding Textures

### Loading Texture Images

We'll use the stb_image library for loading texture files. Here's a comprehensive texture loading function with error handling and support for different formats:

```cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct TextureProperties {
    unsigned int id;
    int width;
    int height;
    int channels;
    GLenum format;
    bool success;
};

TextureProperties loadTexture(const char* path, bool generateMipmap = true) {
    TextureProperties texture = {};
    
    // Generate texture ID
    glGenTextures(1, &texture.id);
    glBindTexture(GL_TEXTURE_2D, texture.id);
    
    // Set default texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Load image data
    stbi_set_flip_vertically_on_load(true); // Flip texture vertically
    unsigned char* data = stbi_load(path, 
                                  &texture.width, 
                                  &texture.height, 
                                  &texture.channels, 
                                  0);
    
    if (data) {
        // Determine format based on number of channels
        switch(texture.channels) {
            case 1:
                texture.format = GL_RED;
                break;
            case 3:
                texture.format = GL_RGB;
                break;
            case 4:
                texture.format = GL_RGBA;
                break;
            default:
                std::cerr << "Unexpected number of channels: " << texture.channels << std::endl;
                stbi_image_free(data);
                return texture;
        }
        
        // Upload texture data to GPU
        glTexImage2D(GL_TEXTURE_2D, 
                    0, 
                    texture.format, 
                    texture.width, 
                    texture.height, 
                    0, 
                    texture.format, 
                    GL_UNSIGNED_BYTE, 
                    data);
        
        if (generateMipmap) {
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        
        texture.success = true;
    } else {
        std::cerr << "Failed to load texture: " << path << std::endl;
        std::cerr << "STB Error: " << stbi_failure_reason() << std::endl;
    }
    
    stbi_image_free(data);
    return texture;
}
```

### Using Multiple Textures

Modern OpenGL applications often use multiple textures simultaneously. Here's how to manage and use multiple textures:

```cpp
class TextureManager {
private:
    struct TextureUnit {
        unsigned int id;
        std::string name;
        int unit;
    };
    std::vector<TextureUnit> textures;
    
public:
    void addTexture(const char* path, const char* uniformName, int textureUnit) {
        TextureUnit tex;
        auto props = loadTexture(path);
        if (props.success) {
            tex.id = props.id;
            tex.name = uniformName;
            tex.unit = textureUnit;
            textures.push_back(tex);
        }
    }
    
    void bindAll(Shader& shader) {
        for (const auto& tex : textures) {
            glActiveTexture(GL_TEXTURE0 + tex.unit);
            glBindTexture(GL_TEXTURE_2D, tex.id);
            shader.setInt(tex.name, tex.unit);
        }
    }
};

// Usage example:
TextureManager texManager;
texManager.addTexture("diffuse.png", "material.diffuse", 0);
texManager.addTexture("specular.png", "material.specular", 1);
texManager.addTexture("normal.png", "material.normal", 2);

// In render loop
shader.use();
texManager.bindAll(shader);
```

## Texture Parameters and Filtering

### Texture Wrapping

```cpp
// Set wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

// Available wrapping options:
// GL_REPEAT: Default, repeats the texture
// GL_MIRRORED_REPEAT: Repeats and mirrors
// GL_CLAMP_TO_EDGE: Stretches edge pixels
// GL_CLAMP_TO_BORDER: Uses border color
```

### Texture Filtering

```cpp
// Set filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

// Available filtering options:
// GL_NEAREST: Nearest neighbor filtering
// GL_LINEAR: Linear filtering
// GL_NEAREST_MIPMAP_NEAREST: Nearest neighbor with mipmaps
// GL_LINEAR_MIPMAP_LINEAR: Trilinear filtering
```

## Advanced Texture Techniques

### Mipmapping

```cpp
// Generate mipmaps automatically
glGenerateMipmap(GL_TEXTURE_2D);

// Custom mipmap levels
for (unsigned int level = 0; level < maxMipmapLevel; ++level) {
    glTexImage2D(GL_TEXTURE_2D, level, GL_RGB, width >> level, height >> level,
                 0, GL_RGB, GL_UNSIGNED_BYTE, mipmapData[level]);
}
```

### Anisotropic Filtering

```cpp
// Enable anisotropic filtering
float maxAnisotropy;
glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);
glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisotropy);
```

### Texture Arrays

```cpp
// Creating and using texture arrays
GLuint textureArray;
glGenTextures(1, &textureArray);
glBindTexture(GL_TEXTURE_2D_ARRAY, textureArray);

glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, width, height, numLayers,
             0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

// In shader
uniform sampler2DArray textureArray;
vec4 color = texture(textureArray, vec3(TexCoord, layer));
```

## UV Mapping Concepts

### Basic UV Mapping

```cpp
// Vertex shader
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}

// Fragment shader
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ourTexture;

void main() {
    FragColor = texture(ourTexture, TexCoord);
}
```

### UV Transformations

```glsl
// Scaling UVs
vec2 scaledUV = TexCoord * 2.0; // Repeat texture twice

// Rotating UVs
float angle = time * 0.1;
mat2 rotationMatrix = mat2(cos(angle), -sin(angle),
                          sin(angle),  cos(angle));
vec2 rotatedUV = rotationMatrix * (TexCoord - 0.5) + 0.5;

// Scrolling UVs
vec2 scrolledUV = TexCoord + vec2(time * 0.1, 0);
```

## Multi-texturing Techniques

### Basic Multi-texturing

```glsl
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform sampler2D specularMap;
uniform sampler2D normalMap;

void main() {
    vec4 diffuse = texture(diffuseMap, TexCoord);
    vec4 specular = texture(specularMap, TexCoord);
    vec3 normal = texture(normalMap, TexCoord).rgb * 2.0 - 1.0;
    
    // Combine textures for final output
    // ... lighting calculations ...
}
```

### Blend Mapping

```glsl
uniform sampler2D grassTexture;
uniform sampler2D rockTexture;
uniform sampler2D blendMap;

void main() {
    vec4 blendMapColor = texture(blendMap, TexCoord);
    vec4 grassColor = texture(grassTexture, TexCoord * 10.0);
    vec4 rockColor = texture(rockTexture, TexCoord * 10.0);
    
    FragColor = mix(grassColor, rockColor, blendMapColor.r);
}
```

## Common Texture Mapping Issues and Solutions

### Texture Seams

```cpp
// Prevent texture seams with proper UV unwrapping
// Use texture atlas or adjust UV coordinates at seams
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
```

### Mipmap Artifacts

```cpp
// Reduce mipmap artifacts with proper filtering
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.4f);
```

## Best Practices and Optimization

1. **Memory Management**
   - Use appropriate texture formats (e.g., DXT compression for non-critical textures)
   - Generate mipmaps for textures used at varying distances
   - Release texture resources when no longer needed

2. **Performance Tips**
   - Batch textures with similar properties
   - Use texture arrays for similar textures
   - Consider using texture atlases for sprite-based applications
   - Implement proper texture streaming for large open worlds

3. **Quality Considerations**
   - Use appropriate filtering modes based on content
   - Consider anisotropic filtering for textures viewed at angles
   - Implement proper UV unwrapping to minimize stretching

## UV Transformations and Effects

Here's how to implement various UV effects in the shader:

```glsl
// Vertex shader
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 uvTransform;  // UV transformation matrix

void main() {
    // Transform position
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    
    // Transform UV coordinates
    vec3 transformedUV = uvTransform * vec3(aTexCoord, 1.0);
    TexCoord = transformedUV.xy;
}

// Fragment shader
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform sampler2D normalMap;
uniform float time;        // For animated effects

void main() {
    // Basic texture lookup
    vec4 diffuseColor = texture(diffuseMap, TexCoord);
    
    // UV animation example (scrolling)
    vec2 scrollingUV = TexCoord + vec2(time * 0.1, 0.0);
    vec4 scrollingColor = texture(diffuseMap, scrollingUV);
    
    // UV scaling example
    vec2 scaledUV = TexCoord * 2.0; // Repeat texture 2x
    vec4 scaledColor = texture(diffuseMap, scaledUV);
    
    // UV rotation example
    float angle = time;
    mat2 rotationMatrix = mat2(
        cos(angle), -sin(angle),
        sin(angle),  cos(angle)
    );
    vec2 rotatedUV = (TexCoord - 0.5) * rotationMatrix + 0.5;
    vec4 rotatedColor = texture(diffuseMap, rotatedUV);
    
    // Combine effects
    FragColor = mix(diffuseColor, scrollingColor, 0.5);
}
```

## Texture Arrays and Atlases

Texture arrays allow efficient handling of multiple similar textures:

```cpp
class TextureArray {
private:
    GLuint textureID;
    int layerCount;
    
public:
    TextureArray(int width, int height, int layers) {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D_ARRAY, textureID);
        
        // Allocate storage for array
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8,
                    width, height, layers,
                    0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                    
        // Set parameters
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
        layerCount = layers;
    }
    
    void addLayer(const char* path, int layer) {
        int width, height, channels;
        unsigned char* data = stbi_load(path, &width, &height, &channels, 4);
        
        if (data && layer < layerCount) {
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                           0, 0, layer,
                           width, height, 1,
                           GL_RGBA, GL_UNSIGNED_BYTE, data);
        }
        
        stbi_image_free(data);
    }
};

// Usage in shader
#version 330 core
uniform sampler2DArray textureArray;
in float layer;  // Layer index passed from vertex shader
in vec2 TexCoord;

void main() {
    vec4 color = texture(textureArray, vec3(TexCoord, layer));
    // ... rest of shader code
}
```

## Next Steps

In the next part, we'll explore advanced texture techniques including:
- Normal mapping and parallax mapping
- Environment mapping (cubemaps)
- Shadow mapping
- Deferred texturing
- Procedural texturing

Remember to experiment with the provided code examples and modify them to understand how different parameters affect the final result!
