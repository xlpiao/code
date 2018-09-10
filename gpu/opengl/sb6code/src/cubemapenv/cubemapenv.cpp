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

#include <object.h>
#include <shader.h>
#include <sb6ktx.h>

class cubemapenv_app : public sb6::application
{
public:
    cubemapenv_app()
        : envmap_index(0),
          render_prog(0)
    {
    }

protected:
    void init()
    {
        static const char title[] = "OpenGL SuperBible - Cubic Environment Map";

        sb6::application::init();

        memcpy(info.title, title, sizeof(title));
    }

    virtual void startup()
    {
        envmaps[0] = sb6::ktx::file::load("media/textures/envmaps/mountaincube.ktx");
        tex_envmap = envmaps[envmap_index];

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

        object.load("media/objects/dragon.sbm");

        load_shaders();

        glGenVertexArrays(1, &skybox_vao);
        glBindVertexArray(skybox_vao);

        glDepthFunc(GL_LEQUAL);
    }

    virtual void render(double currentTime)
    {
        static const GLfloat gray[] = { 0.2f, 0.2f, 0.2f, 1.0f };
        static const GLfloat ones[] = { 1.0f };
        const float t = (float)currentTime * 0.1f;

        vmath::mat4 proj_matrix = vmath::perspective(60.0f, (float)info.windowWidth / (float)info.windowHeight, 0.1f, 1000.0f);
        vmath::mat4 view_matrix = vmath::lookat(vmath::vec3(15.0f * sinf(t), 0.0f, 15.0f * cosf(t)),
                                                vmath::vec3(0.0f, 0.0f, 0.0f),
                                                vmath::vec3(0.0f, 1.0f, 0.0f));
        vmath::mat4 mv_matrix = view_matrix *
                                vmath::rotate(t, 1.0f, 0.0f, 0.0f) *
                                vmath::rotate(t * 130.1f, 0.0f, 1.0f, 0.0f) *
                                vmath::translate(0.0f, -4.0f, 0.0f);

        glClearBufferfv(GL_COLOR, 0, gray);
        glClearBufferfv(GL_DEPTH, 0, ones);
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex_envmap);

        glViewport(0, 0, info.windowWidth, info.windowHeight);

        glUseProgram(skybox_prog);
        glBindVertexArray(skybox_vao);

        glUniformMatrix4fv(uniforms.skybox.view_matrix, 1, GL_FALSE, view_matrix);

        glDisable(GL_DEPTH_TEST);

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glUseProgram(render_prog);

        glUniformMatrix4fv(uniforms.render.mv_matrix, 1, GL_FALSE, mv_matrix);
        glUniformMatrix4fv(uniforms.render.proj_matrix, 1, GL_FALSE, proj_matrix);

        glEnable(GL_DEPTH_TEST);

        object.render();
    }

    virtual void shutdown()
    {
        glDeleteProgram(render_prog);
        glDeleteTextures(3, envmaps);
    }

    void load_shaders()
    {
        if (render_prog)
            glDeleteProgram(render_prog);

        GLuint vs, fs;

        vs = sb6::shader::load("media/shaders/cubemapenv/render.vs.glsl", GL_VERTEX_SHADER);
        fs = sb6::shader::load("media/shaders/cubemapenv/render.fs.glsl", GL_FRAGMENT_SHADER);

        render_prog = glCreateProgram();
        glAttachShader(render_prog, vs);
        glAttachShader(render_prog, fs);
        glLinkProgram(render_prog);

        glDeleteShader(vs);
        glDeleteShader(fs);

        uniforms.render.mv_matrix = glGetUniformLocation(render_prog, "mv_matrix");
        uniforms.render.proj_matrix = glGetUniformLocation(render_prog, "proj_matrix");

        vs = sb6::shader::load("media/shaders/cubemapenv/skybox.vs.glsl", GL_VERTEX_SHADER);
        fs = sb6::shader::load("media/shaders/cubemapenv/skybox.fs.glsl", GL_FRAGMENT_SHADER);

        skybox_prog = glCreateProgram();
        glAttachShader(skybox_prog, vs);
        glAttachShader(skybox_prog, fs);
        glLinkProgram(skybox_prog);

        glDeleteShader(vs);
        glDeleteShader(fs);

        uniforms.skybox.view_matrix = glGetUniformLocation(skybox_prog, "view_matrix");
    }

    virtual void onKey(int key, int action)
    {
        if (action)
        {
            switch (key)
            {
                case 'R': load_shaders();
                    break;
                case 'E':
                    envmap_index = (envmap_index + 1) % 3;
                    tex_envmap = envmaps[envmap_index];
                    break;
            }
        }
    }

protected:
    GLuint          render_prog;
    GLuint          skybox_prog;

    GLuint          tex_envmap;
    GLuint          envmaps[3];
    int             envmap_index;

    struct
    {
        struct
        {
            GLint       mv_matrix;
            GLint       proj_matrix;
        } render;
        struct
        {
            GLint       view_matrix;
        } skybox;
    } uniforms;

    sb6::object     object;

    GLuint          skybox_vao;
};

DECLARE_MAIN(cubemapenv_app)
