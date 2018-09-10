/* ============================================================================
**
** Demonstration of texture gen
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

#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
    #include <GLUT/glut.h>
#else
    #include <GL/gl.h>
    #include <GL/glu.h>
    #include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <jerror.h>

#define		NUM_TEXTURES 10

typedef		struct
{
  char *	name;
  int		format;
  unsigned int	size;
}		texture_info_t;

static GLuint	textures[NUM_TEXTURES];

static int	left_click = GLUT_UP;
static int	right_click = GLUT_UP;
static int	xold;
static int	yold;
static float	rotate_x = 30;
static float	rotate_y = 15;
static float	translate_x = 0;
static float	translate_y = 0;
static float	plane_xy[3] = {1, 0, 0};
static float	plane_yz[3] = {0, 0, 1};


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
** Just a teapot
*/
void		Teapot (void)
{
  glTranslatef (translate_x, 0, translate_y);
  glRotatef (-60, 0, 1, 0);
  glutSolidTeapot (0.4);
}

/*
** Function called to update rendering
*/
void		DisplayFunc (void)
{
  glMatrixMode (GL_TEXTURE);
  glLoadIdentity ();
  glScalef (1, -1, 1);
  glMatrixMode (GL_MODELVIEW);

  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity ();
  glTranslatef (0, 0, -10.);

  glRotatef (rotate_y, 1, 0, 0);
  glRotatef (rotate_x, 0, 1, 0);

  glEnable (GL_TEXTURE_GEN_S);
  glEnable (GL_TEXTURE_GEN_T);
  glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  /* Marble teapot */
  glBindTexture (GL_TEXTURE_2D, textures[0]);
  glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  glTexGenfv (GL_S, GL_EYE_PLANE, plane_xy);
  glTexGenfv (GL_T, GL_EYE_PLANE, plane_yz);
  glPushMatrix ();
  glTranslatef (-1, -1, 0);
  Teapot ();
  glPopMatrix ();

  /* Chess teapot, texture funny projection */
  glBindTexture (GL_TEXTURE_2D, textures[1]);
  glPushMatrix ();
  glLoadIdentity ();
  glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
  glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
  glTexGenfv (GL_S, GL_EYE_PLANE, plane_yz);
  glTexGenfv (GL_T, GL_EYE_PLANE, plane_yz);
  glPopMatrix ();
  glPushMatrix ();
  glTranslatef (0, -1, 0);
  Teapot ();
  glPopMatrix ();

  /*/ Chess teapot, texture projected vertically */
  glBindTexture (GL_TEXTURE_2D, textures[1]);
  glTexGenfv (GL_S, GL_EYE_PLANE, plane_xy);
  glTexGenfv (GL_T, GL_EYE_PLANE, plane_yz);
  glPushMatrix ();
  glTranslatef (1, -1, 0);
  Teapot ();
  glPopMatrix ();

  /* Chrome teapot */
  glBindTexture (GL_TEXTURE_2D, textures[2]);
  glTexGeni (GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
  glTexGeni (GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
  glPushMatrix ();
  glTranslatef (-1, 0, 0);
  Teapot ();
  glPopMatrix ();

  /* Mercedes teapot */
  glBindTexture (GL_TEXTURE_2D, textures[3]);
  glPushMatrix ();
  glTranslatef (0, 0, 0);
  Teapot ();
  glPopMatrix ();

  /* Satin teapot */
  glBindTexture (GL_TEXTURE_2D, textures[4]);
  glPushMatrix ();
  glTranslatef (1, 0, 0);
  Teapot ();
  glPopMatrix ();

  /* Outlined teapot */
  glBindTexture (GL_TEXTURE_2D, textures[5]);
  glPushMatrix ();
  glTranslatef (0, 1, 0);
  Teapot ();
  glPopMatrix ();

  /* Golden teapot */
  glBindTexture (GL_TEXTURE_2D, textures[6]);
  glPushMatrix ();
  glTranslatef (-1, 1, 0);
  Teapot ();
  glPopMatrix ();

  /* Final Fantasy ghost teapot */
  glBindTexture (GL_TEXTURE_2D, textures[7]);
  glColor3f (1, 0.7, 0);
  glPushMatrix ();
  glTranslatef (1, 1, 0);
  Teapot ();
  glPopMatrix ();

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
      translate_x = translate_x + (x - xold) / 100.0;
      translate_y = translate_y + (y - yold) / 100.0;
      glutPostRedisplay ();
    }
  xold = x;
  yold = y;
}


int		main (int argc, char **argv)
{
  unsigned char texture[NUM_TEXTURES][3 * 64 * 64];
  unsigned int i;
  static texture_info_t	textures_info[] =
    {
      { "marble.jpg", GL_RGB, 64},
      { "chess.jpg", GL_LUMINANCE, 64},
      { "chrome.jpg", GL_RGB, 64},
      { "mercedes.jpg", GL_RGB, 64},
      { "satin.jpg", GL_RGB, 64},
      { "outline.jpg", GL_LUMINANCE, 64},
      { "gold.jpg", GL_RGB, 64},
      { "glass.jpg", GL_ALPHA, 64},
      { 0, 0, 0}
    };

  /* Creation of the window */
  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize (500, 500);
  glutCreateWindow ("Texture gen");

  /* OpenGL settings */
  glClearColor (0, 0, 0, 0);
  glEnable (GL_DEPTH_TEST);
  glEnable (GL_CULL_FACE);
  glCullFace (GL_FRONT);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glEnable (GL_TEXTURE_2D);

  /* Texture loading  */
  glGenTextures (NUM_TEXTURES, textures);
  for (i = 0; textures_info[i].name != 0; ++i)
    {
      if (load_texture (textures_info[i].name,
			texture[i],
			textures_info[i].format,
			textures_info[i].size) != 0)
	return 1;

      glBindTexture (GL_TEXTURE_2D, textures[i]);
      gluBuild2DMipmaps (GL_TEXTURE_2D, textures_info[i].format,
			 textures_info[i].size, textures_info[i].size,
			 textures_info[i].format,
			 GL_UNSIGNED_BYTE,  texture[i]);

      glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }

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
