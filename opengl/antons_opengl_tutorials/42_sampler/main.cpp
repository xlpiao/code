#define GLFW_INCLUDE_GLCOREARB
#define GLFW_NO_GLU
#define DEBUG 1
#include "GLFW/glfw3.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

//two triangles put together to make a square
const float positions[] = {
    -1, -1, 0,
    -1, 1, 0,
    1, -1, 0,
    1, -1, 0,
    -1, 1, 0,
    1, 1, 0
};

//4 boxes texture, rbrb
const GLubyte texture[] = {
    255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,
    255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,
    255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,
    255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,
    0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,
    0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,
    0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,
    0,0,255,255,  0,0,255,255,  0,0,255,255,  0,0,255,255,  255,0,0,255,  255,0,0,255,  255,0,0,255,  255,0,0,255
};

//simple shader output = input. texture coordinates are the xy, should make 4 copies of texture.
const std::string vertexStr(
        "#version 150\n"
        "in vec3 pos;\n"
        "out vec2 texCoords;\n"
        "void main(){\n"
        "   texCoords=vec2(pos.x, pos.y);\n"
        "   gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);\n"
        "}\n"
    );

//Gets the values from the texture.
const std::string fragmentStr(
    "#version 150\n"
    "out vec4 outputColor;\n"
    "in vec2 texCoords;\n"
    "uniform sampler2D texSampler;\n"
    "void main(){\n"
        "outputColor = texture(texSampler, texCoords);\n"
    "}\n"
);

bool shaderStatus(GLuint &shader);
bool programStatus(GLuint &program);
int main(int arg_co, char** args){

        /*
         *Creating and initlizing a window on mac.
         *
         **/
        GLFWwindow* window;

        /* Initialize the library */
        if (!glfwInit())
            return -1;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        /* Create a windowed mode window and its OpenGL context */
        window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            return -1;
        }

        /* Make the window's context current */
        glfwMakeContextCurrent(window);

        printf("GLSL version %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
        printf("GL Version: %s\n", glGetString(GL_VERSION));


        /*
         * Create the program.
         */
        const char* vertexCStr = vertexStr.c_str();
        GLuint vertexShader = glCreateShader( GL_VERTEX_SHADER );
        glShaderSource(vertexShader, 1, &vertexCStr, NULL);
        glCompileShader( vertexShader );

        if(!shaderStatus(vertexShader)){
            exit(1);
        }

        const char* fragmentCStr = fragmentStr.c_str();
        GLuint fragmentShader = glCreateShader( GL_FRAGMENT_SHADER );
        glShaderSource(fragmentShader, 1, &fragmentCStr, NULL);
        glCompileShader( fragmentShader );

        if(!shaderStatus(fragmentShader)){
            exit(1);
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        if(!programStatus(program)){
            exit(1);
        }

        /*
         * Set up the buffers.
        */
        glUseProgram(program);

        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint positionBufferObject;
        glGenBuffers(1, &positionBufferObject);
        glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float)*6*3,positions, GL_STREAM_DRAW);

        GLuint posIndex = glGetAttribLocation(program, "pos");
        glEnableVertexAttribArray(posIndex);
        glVertexAttribPointer(posIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glUseProgram(0);

        /*
         * Set up texture buffer
         **/
        glUseProgram(program);

        GLuint texBufferdObject;
        glGenTextures(1, &texBufferdObject);
        glBindTexture(GL_TEXTURE_2D, texBufferdObject);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, &texture[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        GLuint sampler=0;
        glGenSamplers(1, &sampler);
        glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        // glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

        // glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        const int texUnit=0;
        GLuint samplerUniform = glGetUniformLocation(program, "texSampler");
        glUniform1i(samplerUniform, texUnit);
        glUseProgram(0);

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window))
        {
            /* Render here */
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClearDepth(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(program);
            glBindVertexArray(vao);

            glActiveTexture(GL_TEXTURE0 + texUnit);
            glBindTexture(GL_TEXTURE_2D, texBufferdObject);
            glBindSampler(texUnit, sampler);

            glDrawArrays(GL_TRIANGLES, 0, 6);

            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glUseProgram(0);

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }


        glfwTerminate();
        return 0;

}

/**
 * @brief shaderStatus
 * For checking if the shader compiled and printing any error messages.
 *
 * @param shader a shader that was compiled or attempted.
 * @return true if the shader didn't fail.
 */
bool shaderStatus(GLuint &shader){
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        const char *strShaderType = "shader";

        fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, strInfoLog);
        delete[] strInfoLog;
        return false;
    }


    return true;
}

/**
 * @brief programStatus
 *
 * Checks if the program linked ok.
 *
 * @param program
 * @return
 */
bool programStatus(GLuint &program){
    GLint status;
    glGetProgramiv (program, GL_LINK_STATUS, &status);
    if(status==GL_FALSE){
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        return false;
    }
    return true;
}

