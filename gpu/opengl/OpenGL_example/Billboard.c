/* ============================================================================
**
** Demonstration of billboards
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

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <jerror.h>

#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glut.h>
#endif

static GLuint	texture;

static int	left_click = GLUT_UP;
static int	right_click = GLUT_UP;
static int	xold;
static int	yold;
static float	rotate_x = 30;
static float	rotate_y = 15;
static float	rotate_z = -5;


/*
** Function to load a Jpeg file.
*/
int		load_texture (const char * filename,
			      unsigned char * dest,
			      const int format,
			      const unsigned int size)
{
  FILE *fd;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  unsigned char * line;

  cinfo.err = jpeg_std_error (&jerr);
  jpeg_create_decompress (&cinfo);

  if (0 == (fd = fopen(filename, "rb")))
    return 1;

  jpeg_stdio_src (&cinfo, fd);
  jpeg_read_header (&cinfo, TRUE);
  if ((cinfo.image_width != size) || (cinfo.image_height != size))
    return 1;

  if (GL_RGB == format)
    {
      if (cinfo.out_color_space == JCS_GRAYSCALE)
	return 1;
    }
  else
    if (cinfo.out_color_space != JCS_GRAYSCALE)
      return 1;

  jpeg_start_decompress (&cinfo);

  while (cinfo.output_scanline < cinfo.output_height)
    {
      line = dest +
	(GL_RGB == format ? 3 * size : size) * cinfo.output_scanline;
      jpeg_read_scanlines (&cinfo, &line, 1);
    }
  jpeg_finish_decompress (&cinfo);
  jpeg_destroy_decompress (&cinfo);
  return 0;
}

/*
** Just a square
*/
void		Square (void)
{
  const float u[3] = {0.2, 0, 0};
  const float v[3] = {0, 0.2, 0};

  glBegin (GL_QUADS);
  glTexCoord2i(0, 0); glVertex3f ( u[0] + v[0],  u[1] + v[1],  u[2] + v[2]);
  glTexCoord2i(1, 0); glVertex3f (-u[0] + v[0], -u[1] + v[1], -u[2] + v[2]);
  glTexCoord2i(1, 1); glVertex3f (-u[0] - v[0], -u[1] - v[1], -u[2] - v[2]);
  glTexCoord2i(0, 1); glVertex3f ( u[0] - v[0],  u[1] - v[1],  u[2] - v[2]);
  glEnd ();
}

/*
** Axis aligned billboard (trees)
*/
void		axis_billboard (void)
{
  glPushMatrix ();
  glRotatef (rotate_x, 0, -1, 0);
  Square ();
  glPopMatrix ();
}

/*
** World aligned billboard (particules)
*/
void		world_billboard (void)
{
  glPushMatrix ();
  glRotatef (rotate_x, 0, -1, 0);
  glRotatef (rotate_y, -1, 0, 0);
  Square ();
  glPopMatrix ();
}

/*
** Screen aligned billboard (text)
*/
void		screen_billboard (void)
{
  glPushMatrix ();
  glRotatef (rotate_x, 0, -1, 0);
  glRotatef (rotate_y, -1, 0, 0);
  glRotatef (rotate_z, 0, 0, -1);
  Square ();
  glPopMatrix ();
}

/*
** Function called to update rendering
*/
void		DisplayFunc (void)
{
  int i;
  int j;
  int k;

  glClear (GL_COLOR_BUFFER_BIT);
  glLoadIdentity ();
  glTranslatef (0, 0, -10.);

  glRotatef (rotate_z, 0, 0, 1);
  glRotatef (rotate_y, 1, 0, 0);
  glRotatef (rotate_x, 0, 1, 0);

  /* A cube made of 9 billboards */
  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      for (k = 0; k < 3; ++k)
	{
	  glPushMatrix ();
	  glTranslatef (i - 1, j - 1, k - 1);
	  glColor3f (0.2 + i * 0.4, 0.2 + j * 0.4, 0.2 + k * 0.4);

	  /* The bottom rows are axis aligned billboards */
	  if (0 == j)
	    axis_billboard ();

	  /* The middle rows are world aligned billboards */
	  if (1 == j)
	    world_billboard ();

	  /* The top rows are screen aligned billboards */
	  if (2 == j)
	    screen_billboard ();

	  glPopMatrix ();
	}

  /* End */
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
  if (GLUT_RIGHT_BUTTON == button)
    right_click = state;
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
      rotate_y = rotate_y + (y - yold) / 5.0;
      rotate_x = rotate_x + (x - xold) / 5.0;
      if (rotate_y > 90)
	rotate_y = 90;
      if (rotate_y < -90)
	rotate_y = -90;
      glutPostRedisplay ();
    }
  if (GLUT_DOWN == right_click)
    {
      rotate_z = rotate_z + (x - xold) / 5.0;
      glutPostRedisplay ();
    }
  xold = x;
  yold = y;
}


int		main (int narg, char **args)
{
  unsigned char texture_data[64 * 64];

  /* Creation of the window */
  glutInit (&narg, args);
  glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize (500, 500);
  glutCreateWindow ("Billboard");

  /* OpenGL settings */
  glClearColor (0, 0, 0, 0);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE);
  glEnable (GL_TEXTURE_2D);
  glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  /* Texture loading  */
  if (load_texture ("square.jpg", texture_data, GL_ALPHA, 64) != 0)
    return 1;
  glGenTextures (1, &texture);
  glBindTexture (GL_TEXTURE_2D, texture);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_ALPHA, 64, 64, 0,
		GL_ALPHA, GL_UNSIGNED_BYTE, texture_data);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

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
