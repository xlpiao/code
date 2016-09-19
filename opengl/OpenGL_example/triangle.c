#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glut.h>
#endif

/* display function - code from:
http://fly.cc.fer.hr/~unreal/theredbook/chapter01.html
This is the actual usage of the OpenGL library. 
The following code is the same for any platform */
void renderFunction()
{
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0.0, 1.0, 0.0);
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glBegin(GL_POLYGON);
    glVertex2f(0.0, 0.5);
    glVertex2f(-0.5, -0.5);
    glVertex2f(0.5, -0.5);
    glEnd();
    glFlush();
}

/* Main method - main entry point of application
the freeglut library does the window creation work for us, 
regardless of the platform. */
int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    // glutInitDisplayMode (GLUT_3_2_CORE_PROFILE | GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    glutCreateWindow("OpenGL - First window demo");


    glutDisplayFunc(renderFunction);

    glutMainLoop();    
    return 0;
}

