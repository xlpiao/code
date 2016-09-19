/* ============================================================================
**
** Demonstration of display lists
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

#include	<stdlib.h>

#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glut.h>
#endif

static int	rotate_list_id = 0;
static int	rotate_teapot_list_id = 0;
static int	teapot_list_id = 0;
static int	left_click = GLUT_UP;
static int	right_click = GLUT_UP;
static int	xold;
static int	yold;
static float	rotate_x = -30;
static float	rotate_y = 15;
static float	alpha = 0;
static float	beta = 10;

/*
** The instructions to rotate each teapot, according to the mouse
*/
void		compile_rotate_list(void)
{
  glNewList(rotate_list_id, GL_COMPILE);
  glRotatef(rotate_y, 1, 0, 0);
  glRotatef(rotate_x, 0, 1, 0);
  glEndList();
}

/*
** The instructions to rotate each teapot, according to the mouse
*/
void		compile_rotate_teapot_list(void)
{
  glNewList(rotate_teapot_list_id, GL_COMPILE);
  glRotatef(beta, 1, 0, 0);
  glRotatef(alpha, 0, 1, 0);
  glEndList();
}

/*
** The instructions to render a teapot
*/
void		compile_teapot_list(void)
{
  glNewList(teapot_list_id, GL_COMPILE);
  glPushMatrix();

  /* Nested call to another display list */
  glCallList(rotate_teapot_list_id);

  glutWireTeapot(0.3);

  glPopMatrix();
  glEndList();
}

/*
** Function called to update rendering
*/
void		DisplayFunc(void)
{
  int i;
  int j;
  int k;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glTranslatef(0, 0, -10.);

  glCallList(rotate_list_id);

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 3; ++k)
	{
	  glPushMatrix();
	  glTranslatef(i - 1, j - 1, k - 1);
	  glColor3f(i / 2., j / 2., k / 2.);

	  glCallList(teapot_list_id);

	  glPopMatrix();
	}

  /* End */
  glFlush();
  glutSwapBuffers();
}

/*
** Function called when the window is created or resized
*/
void		ReshapeFunc(int width, int height)
{
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity();
  gluPerspective(20, width / (float) height, 5, 15);
  glViewport(0, 0, width, height);

  glMatrixMode(GL_MODELVIEW);
  glutPostRedisplay();
}

/*
** Function called when a key is hit
*/
void		KeyboardFunc(unsigned char key, int x, int y)
{
  xold = x; /* Has no effect: just to avoid a warning */
  yold = y;
  if ('q' == key || 'Q' == key || 27 == key)
    {
      glDeleteLists(rotate_list_id, 1);
      glDeleteLists(rotate_teapot_list_id, 1);
      glDeleteLists(teapot_list_id, 1);
      exit(0);
    }
}

/*
** Function called when a mouse button is hit
*/
void		MouseFunc(int button, int state, int x, int y)
{
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
void		MotionFunc(int x, int y)
{
  if (GLUT_DOWN == left_click)
    {
      rotate_y = rotate_y + (y - yold) / 5.f;
      rotate_x = rotate_x + (x - xold) / 5.f;
      if (rotate_y > 90)
	rotate_y = 90;
      if (rotate_y < -90)
	rotate_y = -90;

      /* Update the rotations */
      compile_rotate_list();

      glutPostRedisplay();
    }
  if (GLUT_DOWN == right_click)
    {
      beta = beta + (y - yold) / 2.f;
      alpha = alpha + (x - xold) / 2.f;

      /* Update the rotations */
      compile_rotate_teapot_list();

      glutPostRedisplay();
    }
  xold = x;
  yold = y;
}


int		main(int argc, char **argv)
{
  /* Creation of the window */
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(500, 500);
  glutCreateWindow("Display lists");

  /* OpenGL settings */
  glClearColor(0, 0, 0, 0);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glEnable(GL_DEPTH_TEST);

  /* Compilation of the instructions lists */
  rotate_list_id = glGenLists(1);
  rotate_teapot_list_id = glGenLists(1);
  teapot_list_id = glGenLists(1);
  compile_rotate_list();
  compile_rotate_teapot_list();
  compile_teapot_list();

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
