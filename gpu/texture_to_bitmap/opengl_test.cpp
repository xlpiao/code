#include <cstdlib> 
#include <cstdio> 
#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glut.h>
#endif

#include "bitmap_image.hpp"
using namespace std;
 

//Draws the 3D scene
void drawScene()
{
    //Clear information from last draw
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
    glMatrixMode(GL_MODELVIEW); //Switch to the drawing perspective
    glLoadIdentity(); //Reset the drawing perspective    

    glEnable(GL_DEPTH_TEST);
 
    static unsigned char texture[3 * 400 * 400];
    static unsigned int texture_id;
 
    // Texture setting
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 400, 400, 0, GL_RGB,    GL_UNSIGNED_BYTE, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
 
    glLoadIdentity();
    glTranslatef(0, 0, -10);    
 
    /* Define a view-port adapted to the texture */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(20, 1, 5, 15);
    glViewport(0, 0, 400, 400);
    glMatrixMode(GL_MODELVIEW);
 
    /* Render to buffer */
    glClearColor(1, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
    
    glColor3f(1.0,0.0,0.0);
 
    glBegin(GL_TRIANGLES);
    
    glVertex3f( 0.0f, 1.0f, 0.0f);             
 
    glVertex3f(-1.0f,-1.0f, 0.0f);             
 
    glVertex3f( 1.0f,-1.0f, 0.0f);             
 
    glEnd();  
    ///glFlush();
    // Image Writing    
    unsigned char* imageData = (unsigned char *)malloc((int)(400*400*(3)));
    glReadPixels(0, 0, 400, 400, GL_RGB, GL_UNSIGNED_BYTE, imageData);
    //write 
    bitmap_image image(400,400);
    image_drawer draw(image);
 
    for (unsigned int i = 0; i < image.width(); ++i)
    {
        for (unsigned int j = 0; j < image.height(); ++j)
        {
            unsigned char* r;
            unsigned char* g;
            unsigned char* b;
            r = ++imageData;
            g = ++imageData;
            b = ++imageData;
            image.set_pixel(i,j,*(r),*(g),*(b));        
        }
    }
 
    image.save_image("bmp/gl_trangle_image.bmp");
    ///glutSwapBuffers(); 
}
 
int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(400, 400); //Set the window size
    glutCreateWindow("Test");    
    drawScene();
    return 0; //This line is never reached
}
