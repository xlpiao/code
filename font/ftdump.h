/****************************************************************************/
/*                                                                          */
/*  The FreeType project -- a free and portable quality TrueType renderer.  */
/*                                                                          */
/*  Copyright 1996-2017                                                     */
/*  D. Turner, R.Wilhelm, and W. Lemberg                                    */
/*                                                                          */
/****************************************************************************/

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_SFNT_NAMES_H
#include FT_TRUETYPE_IDS_H
#include FT_TRUETYPE_TABLES_H
#include FT_TRUETYPE_TAGS_H
#include FT_MULTIPLE_MASTERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FT_Error error;

static int verbose = 0;

static void Print_Name(FT_Face face) {
  const char* ps_name;

  printf("font name entries\n");

  /* XXX: Foundry?  Copyright?  Version? ... */

  printf("   family:     %s\n", face->family_name);
  printf("   style:      %s\n", face->style_name);

  ps_name = FT_Get_Postscript_Name(face);
  if (ps_name == NULL)
    ps_name = "UNAVAILABLE";

  printf("   postscript: %s\n", ps_name);
}

static void Print_Sfnt_Tables(FT_Face face) {
  FT_ULong num_tables, i;
  FT_ULong tag, length;
  FT_Byte buffer[4];

  FT_Sfnt_Table_Info(face, 0, NULL, &num_tables);

  printf("font tables (%lu)\n", num_tables);

  for (i = 0; i < num_tables; i++) {
    FT_Sfnt_Table_Info(face, (FT_UInt)i, &tag, &length);

    if (length >= 4) {
      length = 4;
      FT_Load_Sfnt_Table(face, tag, 0, buffer, &length);
    } else
      continue;

    printf("  %2lu: %c%c%c%c %02X%02X%02X%02X...\n", i, (FT_Char)(tag >> 24),
           (FT_Char)(tag >> 16), (FT_Char)(tag >> 8), (FT_Char)(tag),
           (FT_UInt)buffer[0], (FT_UInt)buffer[1], (FT_UInt)buffer[2],
           (FT_UInt)buffer[3]);
  }
}

static void Print_Fixed(FT_Face face) {
  int i;

  /* num_fixed_size */
  printf("fixed size\n");

  /* available size */
  for (i = 0; i < face->num_fixed_sizes; i++) {
    FT_Bitmap_Size* bsize = face->available_sizes + i;

    printf("   %3d: height %d, width %d\n", i, bsize->height, bsize->width);
    printf("        size %.3f, x_ppem %.3f, y_ppem %.3f\n", bsize->size / 64.0,
           bsize->x_ppem / 64.0, bsize->y_ppem / 64.0);
  }
}

static void Print_Charmaps(FT_Face face) {
  int i, active = -1;

  if (face->charmap)
    active = FT_Get_Charmap_Index(face->charmap);

  /* CharMaps */
  printf("charmaps (%d)\n", face->num_charmaps);

  for (i = 0; i < face->num_charmaps; i++) {
    FT_Long format = FT_Get_CMap_Format(face->charmaps[i]);
    FT_ULong lang_id = FT_Get_CMap_Language_ID(face->charmaps[i]);

    if (format >= 0)
      printf("  %2d: format %2ld, platform %u, encoding %2u", i, format,
             face->charmaps[i]->platform_id, face->charmaps[i]->encoding_id);
    else
      printf("  %2d: synthetic, platform %u, encoding %2u", i,
             face->charmaps[i]->platform_id, face->charmaps[i]->encoding_id);

    if (lang_id == 0xFFFFFFFFUL)
      printf("   (Unicode Variation Sequences)");
    else
      printf("   language %lu", lang_id);

    if (i == active)
      printf(" (active)");

    printf("\n");

    if (verbose) {
      FT_ULong charcode;
      FT_UInt gindex;
      FT_String buf[32];

      FT_Set_Charmap(face, face->charmaps[i]);

      charcode = FT_Get_First_Char(face, &gindex);
      while (gindex) {
        if (FT_HAS_GLYPH_NAMES(face))
          FT_Get_Glyph_Name(face, gindex, buf, 32);
        else
          buf[0] = '\0';

        printf("      0x%04lx => %d %s\n", charcode, gindex, buf);
        charcode = FT_Get_Next_Char(face, charcode, &gindex);
      }
      printf("\n");
    }
  }
}

static void Print_Bytecode(FT_Byte* buffer, FT_UShort length, char* tag) {
  FT_UShort i;
  int j = 0; /* status counter */

  for (i = 0; i < length; i++) {
    if ((i & 15) == 0)
      printf("\n%s:%04hx ", tag, i);

    if (j == 0) {
      printf(" %02x", (FT_UInt)buffer[i]);

      if (buffer[i] == 0x40)
        j = -1;
      else if (buffer[i] == 0x41)
        j = -2;
      else if (0xB0 <= buffer[i] && buffer[i] <= 0xB7)
        j = buffer[i] - 0xAF;
      else if (0xB8 <= buffer[i] && buffer[i] <= 0xBF)
        j = 2 * (buffer[i] - 0xB7);
    } else {
      printf("_%02x", (FT_UInt)buffer[i]);

      if (j == -1)
        j = buffer[i];
      else if (j == -2)
        j = 2 * buffer[i];
      else
        j--;
    }
  }
  printf("\n");
}

static void Print_Programs(FT_Face face) {
  FT_ULong length = 0;
  FT_UShort i;
  FT_Byte* buffer = NULL;
  FT_Byte* offset = NULL;

  TT_Header* head;
  TT_MaxProfile* maxp;

  error = FT_Load_Sfnt_Table(face, TTAG_fpgm, 0, NULL, &length);
  if (error || length == 0)
    goto Prep;

  buffer = (FT_Byte*)malloc(length);
  if (buffer == NULL)
    goto Exit;

  error = FT_Load_Sfnt_Table(face, TTAG_fpgm, 0, buffer, &length);
  if (error)
    goto Exit;

  printf("font program");
  Print_Bytecode(buffer, (FT_UShort)length, (char*)"fpgm");

Prep:
  length = 0;

  error = FT_Load_Sfnt_Table(face, TTAG_prep, 0, NULL, &length);
  if (error || length == 0)
    goto Glyf;

  buffer = (FT_Byte*)realloc(buffer, length);
  if (buffer == NULL)
    goto Exit;

  error = FT_Load_Sfnt_Table(face, TTAG_prep, 0, buffer, &length);
  if (error)
    goto Exit;

  printf("\ncontrol value program");
  Print_Bytecode(buffer, (FT_UShort)length, (char*)"prep");

Glyf:
  length = 0;

  error = FT_Load_Sfnt_Table(face, TTAG_glyf, 0, NULL, &length);
  if (error || length == 0)
    goto Exit;

  buffer = (FT_Byte*)realloc(buffer, length);
  if (buffer == NULL)
    goto Exit;

  error = FT_Load_Sfnt_Table(face, TTAG_glyf, 0, buffer, &length);
  if (error)
    goto Exit;

  length = 0;

  error = FT_Load_Sfnt_Table(face, TTAG_loca, 0, NULL, &length);
  if (error || length == 0)
    goto Exit;

  offset = (FT_Byte*)malloc(length);
  if (offset == NULL)
    goto Exit;

  error = FT_Load_Sfnt_Table(face, TTAG_loca, 0, offset, &length);
  if (error)
    goto Exit;

  head = (TT_Header*)FT_Get_Sfnt_Table(face, FT_SFNT_HEAD);
  maxp = (TT_MaxProfile*)FT_Get_Sfnt_Table(face, FT_SFNT_MAXP);

  for (i = 0; i < maxp->numGlyphs; i++) {
    FT_UInt32 loc;
    FT_UInt16 len;
    char tag[5];

    if (head->Index_To_Loc_Format)
      loc = (FT_UInt32)offset[4 * i] << 24 |
            (FT_UInt32)offset[4 * i + 1] << 16 |
            (FT_UInt32)offset[4 * i + 2] << 8 | (FT_UInt32)offset[4 * i + 3];
    else
      loc = (FT_UInt32)offset[2 * i] << 9 | (FT_UInt32)offset[2 * i + 1] << 1;

    len = (FT_UInt16)(buffer[loc] << 8 | buffer[loc + 1]);

    loc += 10;

    if ((FT_Int16)len < 0) /* composite */
    {
      FT_UShort flags;

      do {
        flags = (FT_UInt16)(buffer[loc] << 8 | buffer[loc + 1]);

        loc += 4;

        loc += flags & FT_SUBGLYPH_FLAG_ARGS_ARE_WORDS ? 4 : 2;

        loc += flags & FT_SUBGLYPH_FLAG_SCALE
                   ? 2
                   : flags & FT_SUBGLYPH_FLAG_XY_SCALE
                         ? 4
                         : flags & FT_SUBGLYPH_FLAG_2X2 ? 8 : 0;
      } while (flags & 0x20); /* more components */

      if ((flags & 0x100) == 0)
        continue;
    } else
      loc += 2 * len;

    len = (FT_UInt16)(buffer[loc] << 8 | buffer[loc + 1]);

    if (len == 0)
      continue;

    loc += 2;

    sprintf(tag, "%04hx", i);
    printf("\nglyf program %hd (%.4s)", i, tag);
    Print_Bytecode(buffer + loc, len, tag);
  }

Exit:
  free(buffer);
  free(offset);
}
