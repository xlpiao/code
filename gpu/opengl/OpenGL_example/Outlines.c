/* ============================================================================
**
** Demonstration of outline rendering
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

static int	left_click = GLUT_UP;
static int	xold;
static int	yold;
static float	rotation_x = -30;
static float	rotation_y = 15;


/*
** Function called to update rendering
*/
void		DisplayFunc (void)
{
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity ();
  glTranslatef (0, 0, -10.);

  glPushMatrix ();
  glTranslatef (0, 0, -2);
  glBegin (GL_QUADS);
  glColor3f (1, 0, 0);
  glVertex3f (0, 0, 0);
  glVertex3f (1, 0, 0);
  glVertex3f (1, 1, 0);
  glVertex3f (0, 1, 0);

  glColor3f (1, 1, 0);
  glVertex3f (-2, -2, 0);
  glVertex3f ( 0, -2, 0);
  glVertex3f ( 0, 0, 0);
  glVertex3f (-2, 0, 0);

  glColor3f (0, 0, 1);
  glVertex3f (0, -0.5, 0);
  glVertex3f (2, -0.5, 0);
  glVertex3f (2, 0, 0);
  glVertex3f (0, 0, 0);

  glEnd ();
  glPopMatrix ();

  /* Transparent teapot body */
  glPushMatrix ();
  glRotatef (rotation_y, 1, 0, 0);
  glRotatef (rotation_x, 0, 1, 0);
  glColor4f (0, 0, 0, 0);
  glCullFace (GL_FRONT);
  glEnable (GL_BLEND);
  glutSolidTeapot (1);
  glDisable (GL_BLEND);
  glPopMatrix ();

  /* Black teapot outline */
  glPushMatrix ();
  glTranslatef (0, 0, 0.1); /* Tiny z shift */
  glRotatef (rotation_y, 1, 0, 0);
  glRotatef (rotation_x, 0, 1, 0);
  glColor3f (0, 0, 0);
  glCullFace (GL_BACK);
  glutSolidTeapot (1);
  glPopMatrix ();

  glFlush ();
  glutSwapBuffers ();
}


/*
** Function called when the window is created or resized
*/
void		ReshapeFunc (int width, int height)
{
  glMatrixMode(GL_PROJECTION);

  glLoadIdentity ();
  gluPerspective (20, width / (float) height, 5, 15);
  glViewport (0, 0, width, height);

  glMatrixMode(GL_MODELVIEW);
  glutPostRedisplay();
}

/*
** Function called when a key is hit
*/
void		KeyboardFunc (unsigned char key, int x, int y)
{
  xold = x; /* Has no effect: just to avoid a warning */
  yold = y;
  if ('q' == key || 'Q' == key || 27 == key)
      exit (0);
}

/*
** Function called when a mouse button is hit
*/
void		MouseFunc (int button, int state, int x, int y)
{
  if (GLUT_LEFT_BUTTON == button)
    left_click = state;
  xold = x;
  yold = y;
}

/*
** Function called when the mouse is moved
*/
void		MotionFunc (int x, int y)
{
  if (GLUT_DOWN == left_click)
    {
      rotation_y = rotation_y + (y - yold) / 5.0;
      rotation_x = rotation_x + (x - xold) / 5.0;
      if (rotation_y > 90)
	rotation_y = 90;
      if (rotation_y < -90)
	rotation_y = -90;

      glutPostRedisplay ();
    }
  xold = x;
  yold = y;
}


int		main (int argc, char **argv)
{
  /* Creation of the window */
  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize (500, 500);
  glutCreateWindow ("Outlines");

  /* OpenGL settings */
  glClearColor (1, 1, 1, 0);
  glEnable (GL_DEPTH_TEST);
  glBlendFunc (GL_ZERO, GL_ONE);
  glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
  glEnable (GL_CULL_FACE);

  /* Declaration of the callbacks */
  glutDisplayFunc (&DisplayFunc);
  glutReshapeFunc (&ReshapeFunc);
  glutKeyboardFunc (&KeyboardFunc);
  glutMouseFunc (&MouseFunc);
  glutMotionFunc (&MotionFunc);

  /* Loop */
  glutMainLoop ();

  /* Never reached */
  return 0;
}

/* ========================================================================= */
