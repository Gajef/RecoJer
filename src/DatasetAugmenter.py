import os
import random

import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps

from GlyphdatasetPreprocessor import GlyphdatasetPreprocessor
from PathsProvider import PathsProvider
from Picture import Picture
from YOLOFilesGenerator import YOLOFilesGenerator


class DatasetAugmenter:
    def __init__(self):
        self.paths = PathsProvider()
        self.preprocessor = GlyphdatasetPreprocessor()
        self.files_generator = YOLOFilesGenerator("glyphdataset")


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

    def trim_borders(self, image):
        inverted = ImageOps.invert(image)
        bbox = inverted.getbbox()
        trimmed_image = inverted.crop(bbox)
        result = ImageOps.invert(trimmed_image)
        return result

    def random_quality_loss(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            image_height, image_width = image.shape
            image_resized = cv2.resize(image, (int(image_width * 0.75), int(image_height * 0.75)))
            image_height, image_width = image_resized.shape
            result = cv2.resize(image, (int(image_width * 1.25), int(image_height * 1.25)))
        else:
            result = image

        return result

    def random_rotation(self, image):
        choice = np.random.choice([0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE], 1)
        if choice == 0:
            rotation = image
        else:
            rotation = cv2.rotate(image, choice[0])

        return rotation

    def random_dilation(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            dilation = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=1)
        else:
            dilation = image

        return dilation

    def random_erosion(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            erosion = cv2.erode(image, np.ones((3, 3), np.uint8), iterations=1)
        else:
            erosion = image
        return erosion

    def random_resize(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            image_width, image_height = image.shape
            scale_factor_w = np.random.uniform(0.85, 1.00)
            scale_factor_h = np.random.uniform(0.85, 1.00)
            image_resized = cv2.resize(image, (int(image_height * scale_factor_w), int(image_width * scale_factor_h)))
        else:
            image_resized = image

        return image_resized

    def random_radioed_resize(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            image_width, image_height = image.shape
            scale_factor = np.random.uniform(0.50, 1.00)
            image_resized = cv2.resize(image, (int(image_height * scale_factor), int(image_width * scale_factor)))
        else:
            image_resized = image

        return image_resized

    def random_little_rotation(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            inverted = ImageOps.invert(Image.fromarray(image))
            inverted = np.array(inverted)
            (h, w) = inverted.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, np.random.uniform(-5, 5), 1.0)
            rotated = cv2.warpAffine(inverted, rotation_matrix, (w, h))
            rotated = np.array(ImageOps.invert(Image.fromarray(rotated)))

        else:
            rotated = image

        return rotated

    # def random_skew(self, image):
    #     choice = np.random.choice([0, 1, ], 1)
    #     pass

    def random_filling(self, image):
        choice = np.random.choice([False, True], 1)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if choice:
            filling = cv2.floodFill(thresh, None, (0, 0), 255)[2]
            filling[filling == 1] = 255
        else:
            filling = image
        return filling

    def random_threshold(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = image
        return thresh

    def random_gaussian(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            kernelW = np.random.choice([3, 5, 7], 1)
            kernelH = np.random.choice([3, 5, 7], 1)
            blur = cv2.GaussianBlur(image, (kernelH[0], kernelW[0]), 0)
        else:
            blur = image

        return blur

    def random_flip(self, image):
        choice = np.random.choice([False, True], 1)
        if choice:
            flipped = cv2.flip(image, 1)
        else:
            flipped = image

        return flipped

    def random_transformation(self, image):
        flip = self.random_flip(image)
        rotation = self.random_rotation(flip)
        erosion = self.random_erosion(rotation)
        quality_loss = self.random_quality_loss(erosion)
        little_rotation = self.random_little_rotation(quality_loss)
        resize = self.random_resize(little_rotation)
        resize = self.random_radioed_resize(resize)
        choice = np.random.choice([0, 1], 1)
        if choice == 0:
            thresh = self.random_filling(resize)
        else:
            thresh = self.random_threshold(resize)
        blurred = self.random_gaussian(thresh)
        final_image = self.trim_borders(Image.fromarray(blurred.astype('uint8'), 'L'))
        return final_image