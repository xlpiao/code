#include <pango/pangoft2.h>

#include "ftdump.h"

typedef struct FT_FaceRec_* FT_Face;

int main(void) {
  char text[] = "a가나bb@fff다라마바사";
  int len = sizeof(text) / sizeof(char) - 1;

  PangoFontMap* font_map = pango_ft2_font_map_new();
  PangoContext* context = pango_font_map_create_context(font_map);

  PangoFontDescription* desc = pango_font_description_from_string("Sans 12");
  pango_context_set_font_description(context, desc);
  pango_font_description_free(desc);

  //// pango itemize
  PangoAttrList* attr_list = pango_attr_list_new();
  GList* p_items = pango_itemize(context, text, 0, len, attr_list, NULL);

  for (GList* it = p_items; it != NULL; it = it->next) {
    PangoItem* p_item = it->data;

    printf("chars: %d\n", p_item->num_chars);
    printf("Length: %d\n", p_item->length);
    printf("Offset: %d\n", p_item->offset);
    printf("Lang: %s\n", pango_language_to_string(p_item->analysis.language));

    PangoFontDescription* font_desc =
        pango_font_describe(p_item->analysis.font);
    char* font_name = (char*)(pango_font_description_get_family(font_desc));
    int font_style = (int)(pango_font_description_get_style(font_desc));
    int font_size = PANGO_PIXELS(pango_font_description_get_size(font_desc));
    printf("name: %s, style: %d, size: %d\n", font_name, font_style, font_size);

    //// check: pango_ft2_font_get_face is deprecated
    // FT_Face face = pango_ft2_font_get_face(p_item->analysis.font);
    //// check: pango_fc_font_lock_face is deprecated after 1.44
    FT_Face face =
        pango_fc_font_lock_face(PANGO_FC_FONT(p_item->analysis.font));
    if (face) {
      // Print_Charmaps(face);
      Print_Fixed(face);
      Print_Name(face);
      Print_Sfnt_Tables(face);
      // Print_Programs(face);
    } else {
      pango_fc_font_unlock_face(PANGO_FC_FONT(p_item->analysis.font));
    }

    //// pango shape
    PangoGlyphString* glyphs = pango_glyph_string_new();
    pango_shape(text, len, &(p_item->analysis), glyphs);

    //// draw glyph
    // draw text with drawing api ...
  }

  return 0;
}
