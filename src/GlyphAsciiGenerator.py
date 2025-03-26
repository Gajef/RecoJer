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
        generator_index = 0
        for generator in self.generators:
            for gardiner, hexcode in self.mapper.gardiner2unicode.items():
                generator.generate_image(self.mapper.to_unicode_char(gardiner) , save_path_png=f"{self.paths.FONTS}/results/{gardiner}_{generator_index}.png")
            print(generator_index)
            generator_index += 1
            

    def generate_pictures_glyphdataset_like(self):
        pass

    def generate_pictures_pyramid_like(self):
        pass