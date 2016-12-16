#define GLFW_INCLUDE_GLCOREARB
// #define GLFW_NO_GLU
#define DEBUG 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h> /* include GLEW and new version of GL on Windows */
#include <GLFW/glfw3.h> /* GLFW helper library */

#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glut.h>
#endif

#define CHECK_GL() do { GLenum e; while ( (e = glGetError()) != GL_NO_ERROR ) { fprintf(stderr, __FILE__ ":%d glGetError() returns '%d'\n", __LINE__, e ); } } while(0)

void test_texturestorage()
{
     for (int num_layers = 250; num_layers<=260; ++num_layers) {

	  // Creates GL_TEXTURE_2D_ARRAY with num_layers
	  GLuint texture;
	  glGenTextures(1, &texture);
	  glBindTexture( GL_TEXTURE_2D_ARRAY, texture );
	  glTexStorage3D( GL_TEXTURE_2D_ARRAY, 
			  1,
			  GL_RGB8,
			  64, 64, num_layers);
	  CHECK_GL();

	  // Retrieves number of layers
	  GLint value;
	  glGetTexParameterIiv(GL_TEXTURE_2D_ARRAY,
			       GL_TEXTURE_VIEW_NUM_LAYERS,
			       &value);
	  printf("GL_TEXTURE_VIEW_NUM_LAYERS: %d  num_layers: %d\n", 
		 value, num_layers);


	  glBindTexture( GL_TEXTURE_2D_ARRAY,0);
	  glDeleteTextures(1, &texture);
     }
}

void test_textureview(int num_layers)
{
     // Creates texture 2d array
     GLuint originaltexture;
     glGenTextures(1, &originaltexture);
     glBindTexture( GL_TEXTURE_2D_ARRAY, originaltexture );
     glTexStorage3D( GL_TEXTURE_2D_ARRAY, 
		     1,
		     GL_RGB8,
		     64, 64, num_layers);
     CHECK_GL();

     // Creates an texture view for the last slice of originaltexture
     GLuint textureview;
     glGenTextures(1, &textureview);
     glTextureView(textureview, GL_TEXTURE_2D,
		   originaltexture,
		   GL_RGB8,
		   0,1,
		   (num_layers-1),1);
     CHECK_GL();

     glDeleteTextures(1, &originaltexture);
     glDeleteTextures(1, &textureview);
}

int
main(int argc, char *argv[])
{
     GLFWwindow* window;
     // Initialize GLUT stuff
     // glutInit(&argc, argv);
     // glutInitDisplayMode( GLUT_RGBA );
     // glutInitWindowSize( 100,100 );
     // glutCreateWindow( argv[ 0 ] );

     // Initialize glew stuff
     glewInit();

     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

     window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
     glfwMakeContextCurrent(window);

     printf("GLSL version %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
     printf("GL Version: %s\n", glGetString(GL_VERSION));

     glActiveTexture(GL_TEXTURE1);

     test_texturestorage();

     // This works fine.
     test_textureview(255);

     // This doesn't work in my environment.
     test_textureview(256);

     glfwSwapBuffers(window);
     glfwTerminate();
     return 0;
}
