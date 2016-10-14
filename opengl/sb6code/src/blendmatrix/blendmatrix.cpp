/*
 * Copyright � 2012-2013 Graham Sellers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <sb6.h>
#include <vmath.h>

class blendmatrix_app : public sb6::application
{
    void init()
    {
        static const char title[] = "OpenGL SuperBible - Blending Functions";

        sb6::application::init();

        memcpy(info.title, title, sizeof(title));
    }

    virtual void startup()
    {
        static const char * vs_source[] =
        {
            "#version 410 core                                                  \n"
            "                                                                   \n"
            "in vec4 position;                                                  \n"
            "                                                                   \n"
            "out VS_OUT                                                         \n"
            "{                                                                  \n"
            "    vec4 color0;                                                   \n"
            "    vec4 color1;                                                   \n"
            "} vs_out;                                                          \n"
            "                                                                   \n"
            "uniform mat4 mv_matrix;                                            \n"
            "uniform mat4 proj_matrix;                                          \n"
            "                                                                   \n"
            "void main(void)                                                    \n"
            "{                                                                  \n"
            "    gl_Position = proj_matrix * mv_matrix * position;              \n"
            "    vs_out.color0 = position * 2.0 + vec4(0.5, 0.5, 0.5, 0.0);     \n"
            "    vs_out.color1 = vec4(0.5, 0.5, 0.5, 0.0) - position * 2.0;     \n"
            "}                                                                  \n"
        };

        static const char * fs_source[] =
        {
            "#version 410 core                                                  \n"
            "                                                                   \n"
            "layout (location = 0, index = 0) out vec4 color0;                  \n"
            "layout (location = 0, index = 1) out vec4 color1;                  \n"
            "                                                                   \n"
            "in VS_OUT                                                          \n"
            "{                                                                  \n"
            "    vec4 color0;                                                   \n"
            "    vec4 color1;                                                   \n"
            "} fs_in;                                                           \n"
            "                                                                   \n"
            "void main(void)                                                    \n"
            "{                                                                  \n"
            "    color0 = vec4(fs_in.color0.xyz, 1.0);                          \n"
            "    color1 = vec4(fs_in.color0.xyz, 1.0);                          \n"
            "}                                                                  \n"
        };

        program = glCreateProgram();
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, fs_source, NULL);
        glCompileShader(fs);

        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, vs_source, NULL);
        glCompileShader(vs);

        glAttachShader(program, vs);
        glAttachShader(program, fs);

        glLinkProgram(program);

        mv_location = glGetUniformLocation(program, "mv_matrix");
        proj_location = glGetUniformLocation(program, "proj_matrix");

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        static const GLushort vertex_indices[] =
        {
            0, 1, 2,
            2, 1, 3,
            2, 3, 4,
            4, 3, 5,
            4, 5, 6,
            6, 5, 7,
            6, 7, 0,
            0, 7, 1,
            6, 0, 2,
            2, 4, 6,
            7, 5, 3,
            7, 3, 1
        };

        static const GLfloat vertex_positions[] =
        {
            -0.25f, -0.25f, -0.25f,
            -0.25f,  0.25f, -0.25f,
             0.25f, -0.25f, -0.25f,
             0.25f,  0.25f, -0.25f,
             0.25f, -0.25f,  0.25f,
             0.25f,  0.25f,  0.25f,
            -0.25f, -0.25f,  0.25f,
            -0.25f,  0.25f,  0.25f,
        };

        glGenBuffers(1, &position_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, position_buffer);
        glBufferData(GL_ARRAY_BUFFER,
                     sizeof(vertex_positions),
                     vertex_positions,
                     GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);

        glGenBuffers(1, &index_buffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     sizeof(vertex_indices),
                     vertex_indices,
                     GL_STATIC_DRAW);

        glEnable(GL_CULL_FACE);
        // glFrontFace(GL_CW);

        //glEnable(GL_DEPTH_TEST);
        //glDepthFunc(GL_LEQUAL);
    }

    virtual void render(double currentTime)
    {
        int i, j;
        static const GLfloat orange[] = { 0.6f, 0.4f, 0.1f, 1.0f };
        static const GLfloat one = 1.0f;

        static const GLenum blend_func[] =
        {
            GL_ZERO,
            GL_ONE,
            GL_SRC_COLOR,
            GL_ONE_MINUS_SRC_COLOR,
            GL_DST_COLOR,
            GL_ONE_MINUS_DST_COLOR,
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_DST_ALPHA,
            GL_ONE_MINUS_DST_ALPHA,
            GL_CONSTANT_COLOR,
            GL_ONE_MINUS_CONSTANT_COLOR,
            GL_CONSTANT_ALPHA,
            GL_ONE_MINUS_CONSTANT_ALPHA,
            GL_SRC_ALPHA_SATURATE,
            GL_SRC1_COLOR,
            GL_ONE_MINUS_SRC1_COLOR,
            GL_SRC1_ALPHA,
            GL_ONE_MINUS_SRC1_ALPHA
        };
        static const int num_blend_funcs = sizeof(blend_func) / sizeof(blend_func[0]);
        static const float x_scale = 20.0f / float(num_blend_funcs);
        static const float y_scale = 16.0f / float(num_blend_funcs);
        const float t = (float)currentTime;

        glViewport(0, 0, info.windowWidth, info.windowHeight);
        glClearBufferfv(GL_COLOR, 0, orange);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glUseProgram(program);

        vmath::mat4 proj_matrix = vmath::perspective(50.0f,
                                                     (float)info.windowWidth / (float)info.windowHeight,
                                                     0.1f,
                                                     1000.0f);
        glUniformMatrix4fv(proj_location, 1, GL_FALSE, proj_matrix);

        glEnable(GL_BLEND);
        glBlendColor(0.2f, 0.5f, 0.7f, 0.5f);
        for (j = 0; j < num_blend_funcs; j++)
        {
            for (i = 0; i < num_blend_funcs; i++)
            {
                vmath::mat4 mv_matrix = 
                    vmath::translate(9.5f - x_scale * float(i),
                                     7.5f - y_scale * float(j),
                                     -18.0f) *
                    vmath::rotate(t * -45.0f, 0.0f, 1.0f, 0.0f) *
                    vmath::rotate(t * -21.0f, 1.0f, 0.0f, 0.0f);
                glUniformMatrix4fv(mv_location, 1, GL_FALSE, mv_matrix);
                glBlendFunc(blend_func[i], blend_func[j]);
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
            }
        }
    }

    virtual void shutdown()
    {
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
        glDeleteBuffers(1, &position_buffer);
    }

private:
    GLuint          program;
    GLuint          vao;
    GLuint          position_buffer;
    GLuint          index_buffer;
    GLint           mv_location;
    GLint           proj_location;
};

DECLARE_MAIN(blendmatrix_app)
