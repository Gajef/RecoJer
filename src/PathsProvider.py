import os.path

import natsort.natsort
class PathsProvider:
    def __init__(self):
        self.DATASET = "../data/GlyphDataset/Dataset"
        self.AUTOMATED = self.DATASET + "/Automated"
        self.MANUAL = self.DATASET + "/Manual"
        self.PICTURES = self.DATASET + "/Pictures"
        self.AUTOMATED_LOCATIONS = self.AUTOMATED + "/Locations"
        self.MANUAL_LOCATIONS = self.MANUAL + "/Locations"
        self.AUGMENTED_DATASET = "../data/GlyphDataset/AugmentedDataset"
        self.AUGMENTATION_MASKS_BIG = "../data/GlyphDataset/AugmentationMasks/Big"
        self.AUGMENTATION_MASKS_SMALL = "../data/GlyphDataset/AugmentationMasks/Small"
        self.FONTS = "../data/AsciiDataset/Fonts"
        self.GLYPH_FONTS = self.FONTS + "/GlyphFonts"
        self.ASCII_DATASET =  "../data/AsciiDataset"
        self.ASCII_GLYPHS = self.ASCII_DATASET + "/AsciiGlyphs"
        self.ASCII_AUGMENTATION = self.ASCII_DATASET + "/AsciiAugmentation"
        self.EXTRA_ASCII_IMAGES = self.ASCII_DATASET + "/ExtraImages"


    # Returns the path from a hieroglyphic image based on the picture, number and gardiner code.
    def get_path_by_picture_number_gardiner(self, picture, number, gardiner, mode="Preprocessed"):
        if mode == "Preprocessed" or mode == "Raw":
            path = f"{self.MANUAL}/{mode}/{int(picture)}/{picture}{number}_{gardiner}.png"
            if not os.path.exists(path):
                path = None
        else:
            path = None

        return path

    def get_picture_path_by_number(self, number, mode="Preprocessed"):
        if mode == "Preprocessed" or mode == "Raw":
            path = f"{self.PICTURES}/egyptianTexts{number}.jpg"
            if not os.path.exists(path):
                path = None
        else:
            path = None

        return path

    def get_files_and_folders_in_order(self, path):
        files_and_folders_paths = os.listdir(path)
        return natsort.natsorted(files_and_folders_paths)

    def get_files_in_order(self, path):
        files_and_folders_paths = self.get_files_and_folders_in_order(path)
        files_paths = list(filter(lambda file_folder_path: os.path.isfile(path + "/" + file_folder_path), files_and_folders_paths))

        return files_paths

    def get_files_by_extension_in_order(self, path, extension):
        files_paths = self.get_files_in_order(path)
        filtered_paths =  list(filter(lambda file_path: file_path.split(".")[1] == extension, files_paths))
        return filtered_paths

    def get_files_path_by_extension_in_order(self, path, extension):
        files_paths = self.get_files_by_extension_in_order(path, extension)
        files_paths = list(map(lambda file_path: path + "/" + file_path, files_paths))
        return files_paths