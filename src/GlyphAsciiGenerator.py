import string

import cv2
import numpy as np
import os

from PIL import Image, ImageDraw
from PIL import ImageFont

from PathsProvider import PathsProvider
from gardiner2unicode import GardinerToUnicodeMap, UnicodeGlyphGenerator
from DatasetAugmenter import DatasetAugmenter
from YOLOFilesGenerator import YOLOFilesGenerator

class GlyphAsciiGenerator:
    def __init__(self):
        self.paths = PathsProvider()
        self.mapper = GardinerToUnicodeMap()
        self.generators = self._create_generators()
        self.augmenter = DatasetAugmenter()
        self.files_generator = YOLOFilesGenerator("ascii")
        self.glyph_class_dictionary = self.__create_dictionary()


    def _create_generators(self, ):
        fonts_path = self.paths.GLYPH_FONTS
        fonts_paths_list = self.paths.get_files_by_extension_in_order(fonts_path, "ttf")

        generators = []
        for font in fonts_paths_list:
            generators.append(UnicodeGlyphGenerator(path_to_font=f"{fonts_path}/{font}"))

        return generators

    def generate_all_single_glyphs(self):
        generator_index = 0
        for generator in self.generators:
            for gardiner in self.mapper.gardiner2unicode.keys():
                generator.generate_image(self.mapper.to_unicode_char(gardiner),
                                         save_path_png=f"{self.paths.ASCII_GLYPHS}/{gardiner}_{generator_index}.png")
            generator_index += 1

    def generate_picture_pyramid_text(self, glyphs_paths_list):
        img_height, img_width = (2800, 1700)
        margin = 150
        column_width = 100

        real_width = img_width - (margin * 2)
        real_height = img_height - (margin * 1.5)

        complete_picture, plotting_column_positions, plotting_index, white_canvas, annotations, picture_id = self.initialize_picture(
            column_width, img_height, img_width, margin, real_width)

        while len(glyphs_paths_list) > 0:
            next_image, next_image_path, is_glyph = self.choose_next_image(glyphs_paths_list)

            (next_image_width, next_image_height) = next_image.size
            if next_image_width > 85:
                resize_factor = 85 / next_image_width
                next_image = next_image.resize((int(next_image_width * resize_factor), int(next_image_height * resize_factor)))
                (next_image_width, next_image_height) = next_image.size

            white_canvas.paste(next_image, (plotting_index[0] + (column_width - next_image_width)//2, plotting_index[1]))
            if is_glyph:
                glyph_gardiner_id = next_image_path.split("/")[-1].split("_")[0]
                glyph_class = self.files_generator.glyph_class_dictionary.get(glyph_gardiner_id)
                normalized_x = (plotting_index[0] + (column_width - next_image_width)//2 + (next_image_width/2)) / img_width
                normalized_y = (plotting_index[1] + (next_image_height/2)) / img_height
                normalized_width = next_image_width / img_width
                normalized_height = next_image_height / img_height
                annotations += [f"{glyph_class} {normalized_x:.6f} {normalized_y:.6f} {normalized_width:.6f} {normalized_height:.6f}"]
            next_image.close()

            if plotting_index[1] + next_image_height <= real_height:
                plotting_index = (plotting_index[0], plotting_index[1] + next_image_height)
            elif plotting_index[1] + next_image_height > real_height and len(plotting_column_positions) != 0:
                plotting_index = (plotting_index[0] + plotting_column_positions.pop(), 440)
            else:
                complete_picture = True

            if complete_picture:
                white_canvas.save(f"{self.paths.ASCII_AUGMENTATION}/pt_{picture_id}.png")
                white_canvas.close()
                self.files_generator.generate_txt(annotations, f"pt_{picture_id}.txt")
                complete_picture, plotting_column_positions, plotting_index, white_canvas, annotations, picture_id = self.initialize_picture(
                    column_width, img_height, img_width, margin, real_width)

        picture_id = self._generate_image_id()
        white_canvas.save(f"{self.paths.ASCII_AUGMENTATION}/pt_{picture_id}.png")
        white_canvas.close()
        self.files_generator.generate_txt(annotations, f"pt_{picture_id}.txt")

    def initialize_picture(self, column_width, img_height, img_width, margin, real_width):
        withe_canvas = np.full((img_height, img_width), 255, dtype=np.uint8)
        white_canvas = Image.fromarray(withe_canvas)

        n_columns = np.random.randint(1, 13)
        blocks = self.distribute_columns_to_blocks(n_columns)
        n_blocks = len(blocks)
        complete_picture = False
        glyphs_width = n_columns * column_width + (n_blocks - 1) * column_width
        leftover_margin = int((real_width - glyphs_width) / 2)

        self.plot_annotations(white_canvas, (margin + leftover_margin, 440), blocks, margin,
                              real_width)

        plotting_index = (margin + leftover_margin, 440)
        plotting_column_positions = self.compute_column_positions(blocks, column_width)

        annotations = []

        picture_id = self._generate_image_id()

        return complete_picture, plotting_column_positions, plotting_index, white_canvas, annotations, picture_id

    def choose_next_image(self, glyphs_paths_list):
        is_glyph = False
        options = ["glyph", "blank", "line", "h_text", "v_text", "erased", "dots"]
        choice = np.random.choice(options, 1, p=[0.6, 0.25, 0.05, 0.0, 0.0, 0.05, 0.05])[0]
        if choice == "glyph":
            next_image_path = glyphs_paths_list.pop(0)
            next_image = Image.open(next_image_path)
            next_image = self.augmenter.random_transformation(np.array(next_image))
            is_glyph = True
        elif choice == "blank" or choice == "line" or choice == "erased" or choice == "dots":
            next_image_path = f"{self.paths.EXTRA_ASCII_IMAGES}/{choice}.png"
            next_image = Image.open(next_image_path)
        elif choice == "h_text":
            next_image = ...
        elif choice == "v_text":
            next_image = ...
        else:
            next_image_path = f"{self.paths.EXTRA_ASCII_IMAGES}/{choice}.png"
            next_image = Image.open(next_image_path)
        return next_image, next_image_path, is_glyph


    def plot_annotations(self, picture, init_pos, blocks, margin, real_width):
        picture_draw = ImageDraw.Draw(picture)
        font = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 50)
        font_columns = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 30)
        letters = list(string.ascii_uppercase)

        plotting_index_top = (init_pos[0], init_pos[1] -185)

        # Nombre de la pagina (PT nnn / (PT nnn))
        picture_draw.text((real_width//2 + 80, margin), f"PT {np.random.randint(1, 800)}", font=font)

        for block in blocks:
            width = 100 * block
            # Cuadrado estrecho
            plotting_index_bottom = (plotting_index_top[0] + width , plotting_index_top[1] + 16)
            picture_draw.rectangle((plotting_index_top, plotting_index_bottom), outline="black", width=3)

            # Cuadrado grande con letras
            plotting_index_top = (plotting_index_top[0], plotting_index_top[1] + 14)
            plotting_index_bottom = (plotting_index_top[0] + width , plotting_index_top[1] + 75)
            picture_draw.rectangle((plotting_index_top, plotting_index_bottom), outline="black", width=3)
            for column in range(0, block):
                plotting_text = (40 + plotting_index_top[0] + column * 100, plotting_index_top[1] + 25)
                picture_draw.text(plotting_text, f"{np.random.choice(letters)}", font=font_columns)

            # Cuadrado estrecho
            plotting_index_top = (plotting_index_top[0], plotting_index_top[1] + 73)
            plotting_index_bottom = (plotting_index_top[0] + width , plotting_index_top[1] + 16)
            picture_draw.rectangle((plotting_index_top, plotting_index_bottom), outline="black", width=3)

            # Cuadrado envolviendo todos los jeroglificos
            plotting_index_top = (plotting_index_top[0], plotting_index_top[1] + 14)
            plotting_index_bottom = (plotting_index_top[0] + width , plotting_index_top[1] + 2350)
            picture_draw.rectangle((plotting_index_top, plotting_index_bottom), outline="black", width=3)

            plotting_index_top = (plotting_index_bottom[0] + 100, init_pos[1] - 185)

        return picture_draw

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

    def compute_column_positions(self, blocks, column_width):
        column_positions = []
        for block_columns in blocks:
            for n_columns in range(0, block_columns - 1):
                column_positions = [column_width] + column_positions
            column_positions = [column_width * 2] + column_positions

        column_positions = column_positions[1:]
        return column_positions

    def __create_dictionary(self):
        classes_number = np.arange(0, len(self.mapper.gardiner2unicode.keys()))
        dictionary = {}
        for name, number in zip(self.mapper.gardiner2unicode.keys(), classes_number):
            dictionary[name] = number

        return dictionary

    def _generate_image_id(self):
        characters = string.ascii_uppercase + string.digits
        image_id =  ''.join(np.random.choice(list(characters)) for _ in range(6))
        return image_id

