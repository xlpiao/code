//CODE REFERENCE:  https://www.khronos.org/registry/egl/sdk/docs/man/html/eglIntro.xhtml

/************************************************************************
STEP REFS: http://www.qnx.com/developers/docs/660/index.jsp?topic=%2Fcom.qnx.doc.screen%2Ftopic%2Fmanual%2Fcscreen_tutorial_opengl.html
 1) establish a connection to and initialize the display
 2) choose an appropriate EGL configuration
 3) create an OpenGL ES rendering context
 4) create a native context
 5) create a native window
 6) set the appropriate properties for your native window
 7) create an EGL on-screen rendering surface
 8) create a main application loop to:
 9) process events in the native context
10) render using OpenGL ES 1.X
11) release resources
************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h> 

#include <GL/gl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <gbm.h>


// #include <GLES/egl.h>
// #include <GLES/gl.h>

static EGLint const attribute_list[] = {
        EGL_RED_SIZE, 1,
        EGL_GREEN_SIZE, 1,
        EGL_BLUE_SIZE, 1,
        EGL_NONE
};
int main(int argc, char ** argv)
{
    EGLDisplay display;
    EGLConfig config;
    EGLContext context;
    EGLSurface surface;
    EGLint num_config;
    struct gbm_device * gbm;
    struct gbm_surface * gs;

    int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    gbm = gbm_create_device( fd );


    display = eglGetDisplay(gbm);
    eglInitialize(display, NULL, NULL);
    eglChooseConfig(display, attribute_list, &config, 1, &num_config);
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL);
    gs = gbm_surface_create(gbm, 300, 400,
                            GBM_BO_FORMAT_ARGB8888, GBM_BO_USE_SCANOUT|
                            GBM_BO_USE_RENDERING);
    surface = eglCreateWindowSurface(display, config, gs, NULL);
    eglMakeCurrent(display, surface, surface, context);
    glClearColor(1.0, 1.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();
    eglSwapBuffers(display, surface);
    sleep(10);
    return EXIT_SUCCESS;
}