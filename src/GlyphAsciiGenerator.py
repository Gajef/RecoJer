import os
import random
import string

import numpy as np

from PIL import Image, ImageDraw
from PIL import ImageFont

from PTBookImporter import paths
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
                self.classify_glyph(f"{self.paths.ASCII_GLYPHS}/{gardiner}_{generator_index}.png")
            generator_index += 1

        # Remove wrong glyphs
        os.remove(f"{self.paths.ASCII_GLYPHS}/Square/D52_3.png")
        os.remove(f"{self.paths.ASCII_GLYPHS}/Square/D52A_3.png")
        os.remove(f"{self.paths.ASCII_GLYPHS}/Square/D53_3.png")

    def classify_glyph(self, path):
        glyph = Image.open(path).convert('L')
        glyph = self.augmenter.trim_borders(glyph)
        x, y = glyph.size
        ratio = x/y
        if ratio > 1.3:
            dst_folder = "Wide"
        elif ratio < 0.7:
            dst_folder = "Tall"
        else:
            dst_folder = "Square"

        glyph_name = path.split("/")[-1]
        dst_path = os.path.join(self.paths.ASCII_GLYPHS, dst_folder, glyph_name)
        os.replace(path, dst_path)
        glyph.close()

    def generate_train_val_pt_pictures(self, source, n_train, n_val, replace=True, dst = paths.ASCII_AUGMENTATION):
        # path = self.path.ASCII_GLYPHS
        if replace:
            self.remove_previous_pictures_and_labels(dst)
        self.generate_pt_pictures(source, n_train, "/train", dst = dst)
        self.generate_pt_pictures(source, n_val, "/val", dst = dst)

    def generate_pt_pictures(self, source, n, path, dst):
        rng = np.random.default_rng()
        stone_img_paths = self.paths.get_files_path_by_extension_in_order_recursive(self.paths.HIERO_DATASET, "png")
        stone_img_paths += self.paths.get_files_path_by_extension_in_order_recursive(self.paths.HIERO_DATASET, "jpg")
        for i in range(n):
            img_paths = self.paths.get_files_path_by_extension_in_order_recursive(source, "png")
            rng.shuffle(img_paths)
            self.generate_picture_pyramid_text(img_paths, stone_img_paths, path, dst)

    def remove_previous_pictures_and_labels(self, dataset_path):
        img_train_paths = self.paths.get_files_path_by_extension_in_order(dataset_path + "/images/train", "png")
        img_val_paths = self.paths.get_files_path_by_extension_in_order(dataset_path + "/images/val", "png")
        txt_train_paths = self.paths.get_files_path_by_extension_in_order(dataset_path + "/labels/train", "txt")
        txt_val_paths = self.paths.get_files_path_by_extension_in_order(dataset_path + "/labels/val", "txt")

        for path in img_train_paths + img_val_paths + txt_train_paths + txt_val_paths:
            os.remove(path)

    def generate_picture_pyramid_text(self, glyphs_paths_list, stone_paths_list, path, dst = paths.ASCII_AUGMENTATION):
        img_shape = (2800, 1700)
        img_height, img_width = img_shape
        margin = 150
        column_width = 100

        real_width = img_width - (margin * 2)
        real_height = img_height - (margin * 1.5)

        complete_picture, plotting_column_positions, plotting_index, white_canvas, annotations, picture_id = self.initialize_picture(
            column_width, img_height, img_width, margin, real_width)

        while len(glyphs_paths_list) > 0:
            glyphs_annotations, next_image_height = self.paste_next_glyphs(glyphs_paths_list,
                                                                           stone_paths_list,
                                                                           white_canvas,
                                                                           plotting_index,
                                                                           column_width,
                                                                           img_shape)

            annotations += glyphs_annotations

            if plotting_index[1] + next_image_height <= real_height:
                glyph_separation = random.randint(1, 10)
                plotting_index = (plotting_index[0], plotting_index[1] + next_image_height + glyph_separation)
            elif plotting_index[1] + next_image_height > real_height and len(plotting_column_positions) != 0:
                plotting_index = (plotting_index[0] + plotting_column_positions.pop(), 440)
            else:
                complete_picture = True

            if complete_picture:
                white_canvas.save(f"{dst}/images{path}/pt_{picture_id}.png")
                white_canvas.close()
                self.files_generator.generate_txt(annotations, path, f"pt_{picture_id}.txt", dst = dst)
                complete_picture, plotting_column_positions, plotting_index, white_canvas, annotations, picture_id = self.initialize_picture(
                    column_width, img_height, img_width, margin, real_width)

        picture_id = self._generate_image_id()
        white_canvas.save(f"{dst}/images{path}/pt_{picture_id}.png")
        white_canvas.close()
        self.files_generator.generate_txt(annotations, path, f"pt_{picture_id}.txt", dst = dst)

    def paste_next_glyphs(self, glyphs_paths_list, stone_paths_list, white_canvas, plotting_index, column_width, img_shape):
        next_image, is_glyph, image_annotations = self.choose_next_image(glyphs_paths_list, stone_paths_list, plotting_index, column_width, img_shape)

        (next_image_width, next_image_height) = next_image.size
        white_canvas.paste(next_image, (plotting_index[0] + (column_width - next_image_width) // 2, plotting_index[1]))
        next_image.close()

        return image_annotations, next_image_height

    def initialize_picture(self, column_width, img_height, img_width, margin, real_width):
        withe_canvas = np.full((img_height, img_width), 255, dtype=np.uint8)
        white_canvas = Image.fromarray(withe_canvas)
        white_canvas_rgb = np.stack([white_canvas, white_canvas, white_canvas], axis=2)
        white_canvas_rgb = Image.fromarray(white_canvas_rgb)


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

    def choose_next_image(self, glyphs_paths_list, stone_paths_list, plotting_index, column_width, img_shape):
        is_glyph = False
        options = ["glyph", "blank", "line", "h_text", "v_text", "erased", "dots", "stone"]
        choice = np.random.choice(options, p=[0.45, 0.225, 0.05, 0.025, 0.0, 0.05, 0.05, 0.15])
        annotations = ""
        if choice == "glyph" or choice == "stone":
            next_image, annotations = self.choose_next_glyphs(glyphs_paths_list, stone_paths_list, plotting_index, column_width, img_shape, is_stone = (choice == "stone"))
            is_glyph = True
        elif choice == "blank" or choice == "line" or choice == "erased" or choice == "dots":
            if choice == "erased":
                choice = np.random.choice(["erased", "erased1", "erased2", "erased3", "erased4"])
            elif choice == "dots":
                choice = np.random.choice(["dots", "dots1", "dots2", "dots3"])
            next_image_path = f"{self.paths.EXTRA_ASCII_IMAGES}/{choice}.png"
            next_image = self.resize_to_width(Image.open(next_image_path))
            # if choice != "blank": Annotations will only show for glyphs
            #    annotations = self.generate_annotations(plotting_index, column_width, img_shape, next_image.size, next_image_path, next_image, (0, 0), 0)
        elif choice == "h_text":
            next_image = self._generate_random_text_location_img("horizontal")
        elif choice == "v_text":
            next_image = ...
        else:
            next_image_path = f"{self.paths.EXTRA_ASCII_IMAGES}/{choice}.png"
            next_image = Image.open(next_image_path)

        return next_image, is_glyph, annotations

    def choose_next_glyphs(self, glyphs_paths_list, stone_paths_list, plotting_index, column_width, img_shape, is_stone = False):
        annotations = []
        if not is_stone:
            transformation_code = np.random.randint(0, 1023)
            next_image_path = glyphs_paths_list.pop(0)
            next_image = Image.open(next_image_path).convert("L")
            next_image = self.augmenter.random_transformation_from_code(np.array(next_image), transformation_code)
        else:
            next_image_path = np.random.choice(stone_paths_list)
            next_image = Image.open(next_image_path)
        custom_class = 1

        do_stack = False if is_stone else np.random.choice([False, True])

        if do_stack:
            glyph_type = next_image_path.split("/")[-2]
            extra_glyphs_paths = self.paths.get_files_path_by_extension_in_order(f"{self.paths.ASCII_GLYPHS}/{glyph_type}", "png")
            extra_glyph_path = np.random.choice(extra_glyphs_paths)
            extra_glyph = Image.open(extra_glyph_path).convert("L")
            extra_glyph = self.augmenter.random_transformation_from_code(np.array(extra_glyph), transformation_code)

            was_rotated = transformation_code & 2

            glyph_separation = np.random.randint(1, 10)

            if (glyph_type == "Tall" and not was_rotated) or (glyph_type == "Wide" and was_rotated):
                # Apilar horizontalmente
                composition_width = next_image.size[0] + glyph_separation + extra_glyph.size[0]
                if composition_width > 60:
                    resize_factor = (60 - glyph_separation) / composition_width
                    next_image = next_image.resize((int(next_image.size[0] * resize_factor), int(next_image.size[1] * resize_factor)))
                    extra_glyph = extra_glyph.resize((int(extra_glyph.size[0] * resize_factor), int(extra_glyph.size[1] * resize_factor)))
                    composition_width = next_image.size[0] + glyph_separation + extra_glyph.size[0]
                composition =  Image.new('L', (composition_width, np.max([next_image.size[1], extra_glyph.size[1]])), color = 255)
                composition_height = composition.size[1]
                composition.paste(next_image, (0, composition_height//2 - next_image.size[1]//2))
                composition.paste(extra_glyph, (glyph_separation + next_image.size[0], int(composition_height/2 - extra_glyph.size[1]/2)))

                # Anotaciones
                annotations += self.generate_annotations(plotting_index, column_width, img_shape, composition.size, next_image_path, next_image, (0, 0), custom_class)
                annotations += self.generate_annotations(plotting_index, column_width, img_shape, composition.size, extra_glyph_path, extra_glyph, (next_image.size[0] + glyph_separation, 0), custom_class)


            elif (glyph_type == "Wide" and not was_rotated) or (glyph_type == "Tall" and was_rotated):
                # Apilar verticalmente
                next_image = self.resize_to_width(next_image)
                extra_glyph = self.resize_to_width(extra_glyph)
                composition =  Image.new('L', (np.max([next_image.size[0], extra_glyph.size[0]]), next_image.size[1] + glyph_separation + extra_glyph.size[1]), color = 255)
                composition_width, composition_height = composition.size
                composition.paste(next_image, (composition_width//2 - next_image.size[0]//2, 0))
                composition.paste(extra_glyph, (0, next_image.size[1] + glyph_separation))
                annotations += self.generate_annotations(plotting_index, column_width, img_shape, next_image.size, next_image_path, next_image, (0, 0), custom_class)
                annotations += self.generate_annotations(plotting_index, column_width, img_shape, (composition.size[0], extra_glyph.size[1]), extra_glyph_path, extra_glyph, (0, next_image.size[1] + glyph_separation), custom_class)
            else:
                # Se deja como está previsiblemente (TODO: o se aplica mixup)
                composition = self.resize_to_width(next_image)
                annotations += self.generate_annotations(plotting_index, column_width, img_shape, composition.size, next_image_path, composition, (0, 0), custom_class)

        else:
            composition = self.resize_to_width(next_image)
            annotations += self.generate_annotations(plotting_index, column_width, img_shape, composition.size, next_image_path, composition, (0, 0), custom_class)

        return composition, annotations

    def resize_to_width(self, image, width=60):
        (image_width, image_height) = image.size
        if image_width > width:
            resize_factor = width / image_width
            image = image.resize(
                (int(image_width * resize_factor), int(image_height * resize_factor)))

        return image

    def generate_annotations(self, plotting_index, column_width, img_shape, composition_size, pasted_image_path, pasted_image, paste_position, custom_class = None):
        (img_height, img_width) = img_shape
        (pasted_image_width, pasted_image_height) = pasted_image.size
        (paste_x, paste_y) = paste_position
        (composition_width, composition_height) = composition_size
        glyph_gardiner_id = pasted_image_path.split("/")[-1].split("_")[0]
        glyph_class = custom_class if custom_class is not None else self.files_generator.glyph_class_dictionary.get(glyph_gardiner_id)
        normalized_x = (plotting_index[0] + ((column_width - composition_width) // 2) + (pasted_image_width / 2) + paste_x) / img_width
        normalized_y = (plotting_index[1] + (composition_height / 2) + paste_y) / img_height
        normalized_width = pasted_image_width / img_width
        normalized_height = pasted_image_height / img_height
        annotation = [f"{glyph_class} {normalized_x:.6f} {normalized_y:.6f} {normalized_width:.6f} {normalized_height:.6f}"]

        return annotation

    def plot_annotations(self, picture, init_pos, blocks, margin, real_width):
        picture_draw = ImageDraw.Draw(picture)
        font = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 50)
        font_columns = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 30)
        font_small = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 20)
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
                # Texto grande
                plotting_text = (40 + plotting_index_top[0] + column * 100, plotting_index_top[1] + 25)
                picture_draw.text(plotting_text, f"{np.random.choice(letters)}", font=font_columns)
                # Texto abajo pequeño
                plotting_text = (plotting_text[0]+13, plotting_text[1] + 85)
                picture_draw.text(plotting_text, f"{self._generate_random_text_location()}", font=font_small, anchor="ma")


            # Cuadrado estrecho
            plotting_index_top = (plotting_index_top[0], plotting_index_top[1] + 73)
            plotting_index_bottom = (plotting_index_top[0] + width , plotting_index_top[1] + 16)
            picture_draw.rectangle((plotting_index_top, plotting_index_bottom), outline="black", width=3)

            # Cuadrado envolviendo todos los jeroglificos
            plotting_index_top = (plotting_index_top[0], plotting_index_top[1] + 14)
            plotting_index_bottom = (plotting_index_top[0] + width , plotting_index_top[1] + 2350)
            picture_draw.rectangle((plotting_index_top, plotting_index_bottom), outline="black", width=3)

            self._plot_border_text(plotting_index_top, picture_draw)

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

    def _generate_random_text_location(self):
        room = np.random.choice(["B", "Bs", "P", "A", "SP", "C", "Cs", "Cm", "Cn", "V", "APs", "APn", "fr."])
        wall = np.random.choice(["E", "Eg", "Eh", "iB", "iE", "iL", "iN", "iS", "iW", "N", "Ne", "Nw", "Nwh", "S", "Se", "Sw", "Swh",
                    "W", "Wg", "Wg"])
        section = np.random.choice(["A", "B", "C", "D", "E", "F", ""], p = [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.64])
        register = np.random.choice(["i", "ii", "iii", "iv", "v", ""], p = [0.16, 0.16, 0.16, 0.16, 0.16, 0.2])

        location = f"{room}/{wall} {section}\n{register} {np.random.randint(500)}"

        return location

    def _generate_random_text_location_img(self, orientation):
        if orientation == "horizontal":
            location_text = self._generate_random_text_location()
            font = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 20)
            text_image = Image.fromarray(np.full((200, 200), 255).astype(np.uint8)).convert('L')
            image_draw = ImageDraw.Draw(text_image)
            image_draw.text((0, 0), location_text, font=font)
            text_image = self.augmenter.trim_borders(text_image)

            return text_image

    def _plot_border_text(self, plot_index, picture_draw):
        font_big = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 30)
        font_small = ImageFont.truetype(self.paths.FONTS + "/Tekton Pro Regular.otf", 20)

        # Anotación superior
        top_text = f"{np.random.randint(1, 10)}"
        picture_draw.text((plot_index[0] - 30, plot_index[1] + 90), top_text, font=font_big)

        bottom_text = f"{np.random.randint(1, 1500)}{np.random.choice(list(string.ascii_lowercase))}"
        picture_draw.text((plot_index[0] - 10, plot_index[1] + 135), bottom_text, font=font_small, anchor="rs")

        # Anotación media
        extra_space = np.random.randint(250, 750)
        top_text = f"{np.random.randint(1, 10)}"
        picture_draw.text((plot_index[0] - 30, plot_index[1] + 90 + extra_space), top_text, font=font_big)

        bottom_text = f"{np.random.randint(1, 1500)}{np.random.choice(list(string.ascii_lowercase))}"
        picture_draw.text((plot_index[0] - 10, plot_index[1] + 135 + extra_space), bottom_text, font=font_small, anchor="rs")

        # Anotación inferior
        extra_space = np.random.randint(1000, 2100)
        top_text = f"{np.random.randint(1, 10)}"
        picture_draw.text((plot_index[0] - 30, plot_index[1] + 90 + extra_space), top_text, font=font_big)

        bottom_text = f"{np.random.randint(1, 1500)}{np.random.choice(list(string.ascii_lowercase))}"
        picture_draw.text((plot_index[0] - 10, plot_index[1] + 135 + extra_space), bottom_text, font=font_small, anchor="rs")