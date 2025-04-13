# OpenGL from the Ground Up: Part 8 - Advanced Techniques

Welcome to Part 8 of our OpenGL tutorial series! In this installment, we'll dive deep into advanced rendering techniques that will elevate your graphics applications to the next level.

## 1. Framebuffers and Render Targets

### Understanding Framebuffers
A framebuffer is an object that contains a collection of buffers that can be used as the destination for rendering. Instead of drawing directly to the screen (default framebuffer), we can render to a custom framebuffer, enabling:
- Post-processing effects
- Dynamic reflections
- Shadow mapping
- Deferred rendering
- Screen-space effects

### Detailed Framebuffer Implementation

```cpp
class Framebuffer {
private:
    GLuint fbo;         // Framebuffer object handle
    GLuint textureId;   // Color attachment texture
    GLuint rbo;         // Renderbuffer object (depth+stencil)
    int width, height;

public:
    Framebuffer(int width, int height) : width(width), height(height) {
        // Step 1: Generate and bind framebuffer
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Step 2: Create color attachment texture
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        
        // Allocate storage for the texture
        glTexImage2D(
            GL_TEXTURE_2D,          // Target
            0,                      // Mipmap level
            GL_RGB,                 // Internal format
            width, height,          // Width and height
            0,                      // Border (must be 0)
            GL_RGB,                 // Format
            GL_UNSIGNED_BYTE,       // Data type
            NULL                    // No data (we'll render to it)
        );

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        // Attach texture to framebuffer
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,         // Target
            GL_COLOR_ATTACHMENT0,   // Attachment point
            GL_TEXTURE_2D,          // Texture target
            textureId,              // Texture ID
            0                       // Mipmap level
        );

        // Step 3: Create renderbuffer for depth and stencil
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        
        // Allocate storage for depth and stencil
        glRenderbufferStorage(
            GL_RENDERBUFFER,
            GL_DEPTH24_STENCIL8,    // Combined depth (24 bits) and stencil (8 bits)
            width, height
        );

        // Attach renderbuffer to framebuffer
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_STENCIL_ATTACHMENT,
            GL_RENDERBUFFER,
            rbo
        );

        // Step 4: Verify framebuffer completeness
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            throw std::runtime_error("Framebuffer is not complete! Check attachment types and formats.");
        }

        // Reset to default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Utility methods
    void bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, width, height);  // Important: set viewport to framebuffer size
    }

    void unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    GLuint getTextureId() const { return textureId; }
    
    // Clean up resources
    ~Framebuffer() {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &textureId);
        glDeleteRenderbuffers(1, &rbo);
    }
};
```

### Post-Processing Pipeline

Here's how to implement a complete post-processing pipeline:

```cpp
class PostProcessor {
private:
    Framebuffer* fbo;
    Shader* postShader;
    GLuint quadVAO;
    
    void setupQuad() {
        float quadVertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };
        
        GLuint quadVBO;
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }

public:
    PostProcessor(int width, int height) {
        fbo = new Framebuffer(width, height);
        setupQuad();
        
        // Load post-processing shaders
        postShader = new Shader("post_vs.glsl", "post_fs.glsl");
    }
    
    void beginRender() {
        fbo->bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    
    void endRender() {
        fbo->unbind();
        
        // Render post-processed result to screen
        postShader->use();
        glDisable(GL_DEPTH_TEST);
        
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fbo->getTextureId());
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        glEnable(GL_DEPTH_TEST);
    }
};
```

### Post-Processing Shaders

Here's a collection of common post-processing effects:

```glsl
// Vertex Shader (post_vs.glsl)
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    TexCoords = aTexCoords;
}

// Fragment Shader (post_fs.glsl)
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D screenTexture;
uniform int effectType;  // 0=none, 1=grayscale, 2=blur, 3=edge

// Utility functions
vec3 grayscale(vec3 color) {
    float average = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return vec3(average);
}

vec3 gaussianBlur(sampler2D tex, vec2 uv) {
    float offset = 1.0 / 300.0;
    vec2 offsets[9] = vec2[](
        vec2(-offset,  offset),  vec2(0.0,  offset),  vec2(offset,  offset),
        vec2(-offset,  0.0),     vec2(0.0,  0.0),     vec2(offset,  0.0),
        vec2(-offset, -offset),  vec2(0.0, -offset),  vec2(offset, -offset)
    );

    float kernel[9] = float[](
        1.0/16, 2.0/16, 1.0/16,
        2.0/16, 4.0/16, 2.0/16,
        1.0/16, 2.0/16, 1.0/16
    );

    vec3 color = vec3(0.0);
    for(int i = 0; i < 9; i++)
        color += texture(tex, uv + offsets[i]).rgb * kernel[i];
    
    return color;
}

vec3 edgeDetection(sampler2D tex, vec2 uv) {
    float offset = 1.0 / 300.0;
    
    vec2 offsets[9] = vec2[](
        vec2(-offset,  offset),  vec2(0.0,  offset),  vec2(offset,  offset),
        vec2(-offset,  0.0),     vec2(0.0,  0.0),     vec2(offset,  0.0),
        vec2(-offset, -offset),  vec2(0.0, -offset),  vec2(offset, -offset)
    );

    float kernel[9] = float[](
        1,  1,  1,
        1, -8,  1,
        1,  1,  1
    );

    vec3 color = vec3(0.0);
    for(int i = 0; i < 9; i++)
        color += texture(tex, uv + offsets[i]).rgb * kernel[i];
    
    return vec3(1.0 - length(color));
}

void main() {
    vec3 color = texture(screenTexture, TexCoords).rgb;
    
    switch(effectType) {
        case 1:
            color = grayscale(color);
            break;
        case 2:
            color = gaussianBlur(screenTexture, TexCoords);
            break;
        case 3:
            color = edgeDetection(screenTexture, TexCoords);
            break;
    }
    
    FragColor = vec4(color, 1.0);
}
```

## Usage Example

Here's how to use these components in your application:

```cpp
class Application {
private:
    PostProcessor* postProcessor;
    Scene* scene;
    
public:
    void init(int width, int height) {
        postProcessor = new PostProcessor(width, height);
        scene = new Scene();
    }
    
    void render() {
        // Begin off-screen rendering
        postProcessor->beginRender();
        
        // Render your scene normally
        scene->render();
        
        // End off-screen rendering and apply post-processing
        postProcessor->endRender();
    }
};
```

## Best Practices and Optimization Tips

1. **Memory Management**
   - Delete framebuffers and associated resources when no longer needed
   - Consider using a framebuffer pool for multiple effects
   - Use appropriate texture formats (e.g., GL_RGB vs GL_RGBA)

2. **Performance Considerations**
   - Minimize framebuffer switches
   - Use appropriate texture filtering modes
   - Consider using multiple render targets (MRT) for complex effects
   - Profile your post-processing effects and optimize expensive operations

3. **Common Pitfalls**
   - Always check framebuffer completeness
   - Remember to reset viewport when switching framebuffers
   - Handle resolution changes properly
   - Be careful with depth testing during post-processing

## Next Steps

In the next part, we'll explore advanced topics including:
- Shadow mapping
- Deferred rendering
- Screen-space ambient occlusion (SSAO)
- High dynamic range (HDR) rendering
- Bloom effects

Remember to experiment with different post-processing effects and combine them to create unique visual styles for your applications!
