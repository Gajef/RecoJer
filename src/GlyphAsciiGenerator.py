from PathsProvider import PathsProvider
from gardiner2unicode import GardinerToUnicodeMap, UnicodeGlyphGenerator


class GlyphAsciiGenerator:
    def __init__(self):
        self.paths = PathsProvider()
        self.mapper = GardinerToUnicodeMap()
        self.generators = self._create_generators()

    def _create_generators(self, ):
        fonts_path = self.paths.FONTS
        fonts_paths_list = self.paths.get_files_by_extension_in_order(fonts_path, "ttf")

        generators = []
        for font in fonts_paths_list:
            generators.append(UnicodeGlyphGenerator(path_to_font=f"{fonts_path}/{font}"))

        return generators

    def generate_all_single_glyphs(self):
        for generator in self.generators:
            for gardiner_glyph in self.mapper.gardiner2unicode.keys():
                print(generator, gardiner_glyph)

    def generate_pictures_glyphdataset_like(self):
        pass

    def generate_pictures_pyramid_like(self):
        pass