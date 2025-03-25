import os
import random

import cv2
import numpy as np

from GlyphdatasetPreprocessor import GlyphdatasetPreprocessor
from PathsProvider import PathsProvider
from Picture import Picture
from YOLOFilesGenerator import YOLOFilesGenerator


class DatasetAugmenter:
    def __init__(self):
        self.paths = PathsProvider()
        self.preprocessor = GlyphdatasetPreprocessor()
        self.files_generator = YOLOFilesGenerator()


    def full_augmentation(self):
        pictures_path = self.paths.PICTURES
        pictures_path = self.paths.get_files_by_extension_in_order(pictures_path, "jpg")
        for picture_path in pictures_path:
            self.augmentation(picture_path)

    # Returns n images by doing all the preprocessing
    def augmentation(self, picture_name, verbose=False):
        image = cv2.cvtColor(cv2.imread(f"{self.paths.PICTURES}/{picture_name}"), cv2.COLOR_RGB2GRAY)
        picture = Picture(image, picture_name.split(".")[0])

        # 2 extra brightness pictures
        higher_brightness = self.preprocessor.brightness(image, 1.5)
        hb_picture = Picture(higher_brightness, f"{picture_name.split('.')[0]}_hb")
        lower_brightness = self.preprocessor.brightness(image, 0.5)
        lb_picture = Picture(lower_brightness, f"{picture_name.split('.')[0]}_lb")
        self.train_test_split_and_save_pictures([picture, hb_picture, lb_picture])

        # 3 binarized imgs
        auto_binarized_img = self.preprocessor.binarize(image)
        bin_picture = Picture(auto_binarized_img, f"{picture_name.split('.')[0]}_bin")
        low_binarized_img = self.preprocessor.binarize(image, th=140)
        lbin_picture = Picture(low_binarized_img, f"{picture_name.split('.')[0]}_low")
        high_binarized_img = self.preprocessor.binarize(image, th=180)
        hbin_picture = Picture(high_binarized_img, f"{picture_name.split('.')[0]}_hin")
        self.train_test_split_and_save_pictures([bin_picture, lbin_picture, hbin_picture])

        # 1 borders_imgs
        auto_borders = self.preprocessor.ext_border(auto_binarized_img)
        ab_picture = Picture(auto_borders, f"{picture_name.split('.')[0]}_ab")
        self.train_test_split_and_save_pictures([ab_picture])

        # 2 shadow_imgs
        left_shadow_img = self.preprocessor.shadow(image, "left")
        lf_picture = Picture(left_shadow_img, f"{picture_name.split('.')[0]}_left")
        right_shadow_img = self.preprocessor.shadow(image, "right")
        rg_picture = Picture(right_shadow_img, f"{picture_name.split('.')[0]}_rg")
        self.train_test_split_and_save_pictures([lf_picture, rg_picture])

        if verbose:
            pass
        return None #[binarized_img, borders]

    def train_test_split_and_save_pictures(self, pictures):
        dst = self.paths.AUGMENTED_DATASET
        source = self.paths.MANUAL_LOCATIONS

        rng = np.random.default_rng()
        rng.shuffle(pictures)

        length = len(pictures)

        if length > 1:
            train = pictures[: int(length * 0.67)]
            test = pictures[int(length * 0.67):]
        else:
            slicer = random.randint(0, 1)
            train = pictures[:slicer]
            test = pictures[slicer:]


        for train_picture in train:
            train_image = train_picture.image
            train_name = train_picture.name

            location_number = train_name.split(".")[0].split("_")[0].split("Texts")[1]
            cv2.imwrite(f"{dst}/images/train/{train_name}.jpg", train_image)
            self.files_generator.translate_txt(f"{source}/{location_number}.txt", f"{dst}/labels/train/{train_name}.txt")

        for test_img in test:
            test_image = test_img.image
            test_name = test_img.name
            location_number = test_name.split(".")[0].split("_")[0].split("Texts")[1]
            cv2.imwrite(f"{dst}/images/val/{test_name}.jpg", test_image)
            self.files_generator.translate_txt(f"{source}/{location_number}.txt", f"{dst}/labels/val/{test_name}.txt")

    # Generate folders for train and test
    def generate_folders(self, folder_name):
        dst = self.paths.AUGMENTED_DATASET

        # Images files
        if not os.path.exists(f"{dst}/{folder_name}"):
            os.makedirs(f"{dst}/{folder_name}")

        if not os.path.exists(f"{dst}/{folder_name}/train"):
            os.makedirs(f"{dst}/{folder_name}/train")

        if not os.path.exists(f"{dst}/{folder_name}/val"):
            os.makedirs(f"{dst}/{folder_name}/val")
