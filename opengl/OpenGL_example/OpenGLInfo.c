/* ============================================================================
**
** Demonstration of spinning cube
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


/*
** Function called to update rendering, just one time in this case
*/
void		DisplayFunc (void)
{
  int lights = 0;
  int clipping_planes = 0;
  int model_stack = 0;
  int projection_stack = 0;
  int texture_stack = 0;
  int bpp = 0;
  int max_tex3d = 0;
  int max_tex2d = 0;
  int name_stack = 0;
  int list_stack = 0;
  int max_poly = 0;
  int attrib_stack = 0;
  int buffers = 0;

  unsigned char stencil = 0;
  unsigned char accum = 0;
  unsigned char rgba = 0;
  unsigned char index = 0;
  unsigned char double_buffer = 0;
  unsigned char stereo = 0;

  int convolution_width = 0;
  int convolution_height = 0;
  int max_index = 0;
  int max_vertex = 0;
  int texture_units = 0;

  glGetIntegerv (GL_MAX_LIGHTS, &lights);
  glGetIntegerv (GL_MAX_CLIP_PLANES, &clipping_planes);
  glGetIntegerv (GL_MAX_MODELVIEW_STACK_DEPTH, &model_stack);
  glGetIntegerv (GL_MAX_PROJECTION_STACK_DEPTH, &projection_stack);
  glGetIntegerv (GL_MAX_TEXTURE_STACK_DEPTH, &texture_stack);
  glGetIntegerv (GL_SUBPIXEL_BITS, &bpp);
//  glGetIntegerv (GL_MAX_3D_TEXTURE_SIZE, &max_tex3d);
  glGetIntegerv (GL_MAX_TEXTURE_SIZE, &max_tex2d);
  glGetIntegerv (GL_MAX_NAME_STACK_DEPTH, &name_stack);
  glGetIntegerv (GL_MAX_LIST_NESTING, &list_stack);
  glGetIntegerv (GL_MAX_EVAL_ORDER, &max_poly);
  glGetIntegerv (GL_MAX_ATTRIB_STACK_DEPTH, &attrib_stack);
  glGetIntegerv (GL_AUX_BUFFERS, &buffers);
  glGetBooleanv (GL_STENCIL, &stencil);
  glGetBooleanv (GL_ACCUM, &accum);
  glGetBooleanv (GL_RGBA_MODE, &rgba);
  glGetBooleanv (GL_INDEX_MODE, &index);
  glGetBooleanv (GL_DOUBLEBUFFER, &double_buffer);
  glGetBooleanv (GL_STEREO, &stereo);
//  glGetIntegerv (GL_MAX_CONVOLUTION_WIDTH, &convolution_width);
//  glGetIntegerv (GL_MAX_CONVOLUTION_HEIGHT, &convolution_height);
//  glGetIntegerv (GL_MAX_ELEMENTS_INDICES, &max_index);
//  glGetIntegerv (GL_MAX_ELEMENTS_VERTICES, &max_vertex);
//  glGetIntegerv (GL_MAX_TEXTURE_UNITS_ARB, &texture_units);

  printf ("OpenGL driver informations:\n");
  printf ("===========================\n\n");
  printf ("OpenGL version: %s\n", glGetString (GL_VERSION));
  printf ("Software implementation: %s\n", glGetString (GL_VENDOR));
  printf ("Renderer: %s\n", glGetString (GL_RENDERER));
  printf ("Supported extensions: %s\n\n", glGetString (GL_EXTENSIONS));

  printf ("Double buffering: %s\n", (double_buffer ? "yes" : "no"));
  printf ("Stereo: %s\n", (stereo ? "yes" : "no"));
  printf ("Auxiliary buffer(s): %d\n", buffers);
  printf ("Stencil buffer: %s\n", (stencil ? "yes" : "no"));
  printf ("Accumulation buffer: %s\n\n", (accum ? "yes" : "no"));

  printf ("Bits per pixel: %d\n", bpp);
  printf ("RGBA color: %s\n", (rgba ? "yes" : "no"));
  printf ("Index color palette: %s\n\n", (index ? "yes" : "no"));

  printf ("Model stack size: %d\n", model_stack);
  printf ("Projection stack size: %d\n", projection_stack);
  printf ("Texture stack size: %d\n", texture_stack);
  printf ("Name stack size: %d\n", name_stack);
  printf ("List stack size: %d\n", list_stack);
  printf ("Attributes stack size: %d\n\n", attrib_stack);

/*   printf ("Maximum 3D texture size: %d\n", max_tex3d); */
  printf ("Maximum 2D texture size: %d\n", max_tex2d);
/*   printf ("Maximum texture units: %d\n\n", texture_units); */

  printf ("Maximum lights: %d\n", lights);
  printf ("Maximum clipping planes: %d\n", clipping_planes);
  printf ("Maximum evaluators equation order: %d\n", max_poly);
/*   printf ("Maximum convolution: %dx%d\n", */
/* 	  convolution_width, convolution_height); */
/*   printf ("Maximum recommended index elements: %d\n", max_index); */
/*   printf ("Maximum recommended vertex elements: %d\n", max_vertex); */

  /* That's all! */
  exit (0);
}


int		main (int argc, char **argv)
{
  /* Creation of the window */
  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize (500, 500);
  glutCreateWindow ("OpenGL info");

  /* Declaration of the callback */
  glutDisplayFunc (&DisplayFunc);

  /* Loop */
  glutMainLoop ();

  /* Never reached */
  return 0;
}

/* ========================================================================= */
