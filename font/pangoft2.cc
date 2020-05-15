#include <iostream>

#include <pango/pangoft2.h>

#include "ftdump.h"

typedef struct FT_FaceRec_* FT_Face;

int main(void) {
  std::string text = "a가나bb@fff다라마바사";
  std::cout << "text: " << text << std::endl;

  PangoFontMap* font_map = pango_ft2_font_map_new();
  PangoContext* context = pango_font_map_create_context(font_map);

  PangoFontDescription* desc = pango_font_description_from_string("Sans 12");
  pango_context_set_font_description(context, desc);
  pango_font_description_free(desc);

  //// pango itemize
  PangoAttrList* attr_list = pango_attr_list_new();
  GList* p_items =
      pango_itemize(context, text.c_str(), 0, text.size(), attr_list, NULL);

  for (GList* it = p_items; it != NULL; it = it->next) {
    PangoItem* p_item = (PangoItem*)it->data;

    std::cout << "chars: " << p_item->num_chars << std::endl;
    std::cout << "Length: " << p_item->length << std::endl;
    std::cout << "Offset: " << p_item->offset << std::endl;
    std::cout << "Lang: " << pango_language_to_string(p_item->analysis.language)
              << std::endl;

    PangoFontDescription* font_desc =
        pango_font_describe(p_item->analysis.font);
    char* font_name = (char*)(pango_font_description_get_family(font_desc));
    int font_style = (int)(pango_font_description_get_style(font_desc));
    int font_size = PANGO_PIXELS(pango_font_description_get_size(font_desc));
    std::cout << "name: " << font_name << ", style: " << font_style
              << ", size: " << font_size << std::endl;

    //// check: pango_ft2_font_get_face is deprecated
    // FT_Face face = pango_ft2_font_get_face(p_item->analysis.font);
    //// check: pango_fc_font_lock_face is deprecated after 1.44
    FT_Face face =
        pango_fc_font_lock_face(PANGO_FC_FONT(p_item->analysis.font));
    if (face) {
      // Print_Charmaps(face);
      // Print_Fixed(face);
      // Print_Name(face);
      // Print_Sfnt_Tables(face);
      // Print_Programs(face);
    } else {
      pango_fc_font_unlock_face(PANGO_FC_FONT(p_item->analysis.font));
    }

    //// pango shape
    auto item_text = text.substr(p_item->offset, p_item->length);
    std::cout << "item text: " << item_text << std::endl;
    PangoGlyphString* glyph_string = pango_glyph_string_new();
    pango_shape(item_text.c_str(), item_text.size(), &(p_item->analysis),
                glyph_string);

    for (int index = 0; index < glyph_string->num_glyphs; ++index) {
      PangoGlyphInfo* glyph_info = &glyph_string->glyphs[index];

      int glyph_id = glyph_info->glyph;

      int x_advance = PANGO_PIXELS(glyph_info->geometry.width);

      PangoFontMetrics* metrics = pango_font_get_metrics(
          p_item->analysis.font, p_item->analysis.language);
      int y_advance = PANGO_PIXELS(pango_font_metrics_get_ascent(metrics) +
                                   pango_font_metrics_get_descent(metrics));

      int x_offset = glyph_info->geometry.x_offset;
      int y_offset = glyph_info->geometry.y_offset;

      std::cout << "glyph_id: " << glyph_id << ", x_advance: " << x_advance
                << ", y_advance: " << y_advance << ", x_offset: " << x_offset
                << ", y_offset: " << y_offset << std::endl;
    }
    std::cout << std::endl;

    //// draw glyph
    // draw text with drawing api ...
  }

  return 0;
}
