/* ============================================================================
**
** Demonstration of view-ports
** Copyright (C) 2005  Julien Guertault
**
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License
** as published by the Free Software Foundation; either version 2
** of the License, or (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
**
** ========================================================================= */

#include <math.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

static int left_click = GLUT_UP;
static int right_click = GLUT_UP;
static int xold;
static int yold;
static int width;
static int height;
static int wh;
static int hw;
static float rotate_x = 30;
static float rotate_y = 15;
static float alpha = 0;
static float beta = 0;

/*
** Just a teapot and its frame
*/
void Teapot(void) {
    /* Axis */
    glBegin(GL_LINES);
    glColor3f(1, 0, 0);
    glVertex3f(-1, -1, -1);
    glVertex3f(1, -1, -1);
    glColor3f(0, 1, 0);
    glVertex3f(-1, -1, -1);
    glVertex3f(-1, 1, -1);
    glColor3f(0, 0, 1);
    glVertex3f(-1, -1, -1);
    glVertex3f(-1, -1, 1);
    glEnd();

    glRotatef(beta, 1, 0, 0);
    glRotatef(alpha, 0, 1, 0);
    glColor3f(1, 1, 1);

    /* Teapot itself */
    glutWireTeapot(0.5);
}

/*
** Function called to update rendering
*/
void DisplayFunc(void) {
    glEnable(GL_SCISSOR_TEST);
    glScissor((float)width / 4, (float)height / 4, (float)width, (float)height);

    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.5, 0.5, 0.0, 0);

    glLoadIdentity();

    /* Perspective view */
    glViewport(0, 0, width / 2, height / 2);
    glPushMatrix();
    glTranslatef(0, 0, -10);
    glRotatef(rotate_y, 1, 0, 0);
    glRotatef(rotate_x, 0, 1, 0);
    Teapot();
    glPopMatrix();

    /* Orthogonal projection */
    glMatrixMode(GL_PROJECTION);
    /*
    ** Note how the projection matrix is pushed, to save the perspective
    ** computed in ReshapeFunc
    */
    glPushMatrix();

    glLoadIdentity();
    if (height > width)
        glOrtho(-1.2, 1.2, -1.2 * hw, 1.2 * hw, -1.2, 1.2);
    else
        glOrtho(-1.2 * wh, 1.2 * wh, -1.2, 1.2, -1.2, 1.2);
    glMatrixMode(GL_MODELVIEW);

    /* Right view */
    glViewport(0, height / 2 + 1, width / 2 + 1, height / 2);
    glPushMatrix();
    glRotatef(90, 0, -1, 0);
    Teapot();
    glPopMatrix();

    /* Face view */
    glViewport(width / 2 + 1, height / 2 + 1, width / 2, height / 2);
    glPushMatrix();
    Teapot();
    glPopMatrix();

    /* Top view */
    glViewport(width / 2 + 1, 0, width / 2, height / 2);
    glPushMatrix();
    glRotatef(90, 1, 0, 0);
    Teapot();
    glPopMatrix();

    /* Projection matrix is restaured */
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    /* End */
    glFlush();
    glutSwapBuffers();
}

/*
** Function called when the window is created or resized
*/
void ReshapeFunc(int new_width, int new_height) {
    width = new_width;
    height = new_height;

    hw = height / (float)width;
    wh = width / (float)height;

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    gluPerspective(20 * sqrtf(1 + hw * hw), wh, 8, 12);

    glMatrixMode(GL_MODELVIEW);
    glutPostRedisplay();
}

/*
** Function called when a key is hit
*/
void KeyboardFunc(unsigned char key, int x, int y) {
    xold = x; /* Has no effect: just to avoid a warning */
    yold = y;
    if ('q' == key || 'Q' == key || 27 == key)
        exit(0);
}

/*
** Function called when a mouse button is hit
*/
void MouseFunc(int button, int state, int x, int y) {
    if (GLUT_LEFT_BUTTON == button)
        left_click = state;
    if (GLUT_RIGHT_BUTTON == button)
        right_click = state;
    xold = x;
    yold = y;
}

/*
** Function called when the mouse is moved
*/
void MotionFunc(int x, int y) {
    if (GLUT_DOWN == left_click) {
        rotate_y = rotate_y + (y - yold) / 5.f;
        rotate_x = rotate_x + (x - xold) / 5.f;
        if (rotate_y > 90)
            rotate_y = 90;
        if (rotate_y < -90)
            rotate_y = -90;
        glutPostRedisplay();
    }
    if (GLUT_DOWN == right_click) {
        beta = beta + (y - yold) / 2.f;
        alpha = alpha + (x - xold) / 2.f;
        glutPostRedisplay();
    }
    xold = x;
    yold = y;
}

int main(int argc, char **argv) {
    /* Creation of the window */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(500, 500);
    glutCreateWindow("Viewport");

    /* OpenGL settings */
    glClearColor(0, 0, 0, 0);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /* Declaration of the callbacks */
    glutDisplayFunc(&DisplayFunc);
    glutReshapeFunc(&ReshapeFunc);
    glutKeyboardFunc(&KeyboardFunc);
    glutMouseFunc(&MouseFunc);
    glutMotionFunc(&MotionFunc);

    /* Loop */
    glutMainLoop();

    /* Never reached */
    return 0;
}

/* ========================================================================= */
