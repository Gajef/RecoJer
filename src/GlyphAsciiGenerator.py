from random import randint

import cv2
import numpy as np
import os

from PIL import Image

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
            print(generator.font)
            for gardiner in self.mapper.gardiner2unicode.keys():
                generator.generate_image(self.mapper.to_unicode_char(gardiner),
                                         save_path_png=f"{self.paths.FONTS_RESULTS}/{gardiner}_{generator_index}.png")
            print(generator_index)
            generator_index += 1

    def generate_pictures_pt(self, glyph_paths_stack, borders=150 ,columns=12):
        pass

    def generate_picture_pyramid_text(self, glyphs_paths_stack):
        height, width = (2800, 1700)
        margin = 150
        n_columns = np.random.randint(1, 13)
        blocks = self.distribute_columns_to_blocks(n_columns)
        n_blocks = len(blocks)

        real_widht = width - (margin * 2)
        glyphs_width = n_columns * 100 + (n_blocks - 1) * 100
        leftover_margin = int((real_widht - glyphs_width)/ 2)

        plotting_index = (440, margin + leftover_margin)

        glyph_img = Image.open("../data/fonts/results/A1_0.png")

        withe_canvas = np.full((height, width), 255, dtype=np.uint8)
        white_canvas = Image.fromarray(withe_canvas)

        white_canvas.paste(glyph_img, plotting_index)
        glyph_img.save("A2_0.png")
        white_canvas.save("image.png")

    def distribute_columns_to_blocks(self, n_columns):
        blocks = []
        columns_remaining = n_columns
        size = 0
        while columns_remaining > 0 and size <= 1400:
            columns_in_block = np.random.randint(1, np.min([columns_remaining+1 , 6]))
            blocks.append(columns_in_block)
            columns_remaining -= columns_in_block
            size = np.sum(blocks) * 100 + (len(blocks) - 1) * 100
        return blocks

    def generate_pictures_glyphdataset(self):
        pass