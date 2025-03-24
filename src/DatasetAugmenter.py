import os
import cv2
import numpy as np

from GlyphdatasetPreprocessor import GlyphdatasetPreprocessor
from PathsProvider import PathsProvider
from Picture import Picture


class DatasetAugmentator:
    def __init__(self):
        self.paths = PathsProvider()
        self.preprocessor = GlyphdatasetPreprocessor()


    def full_augmentation(self):
        pictures_path = self.paths.PICTURES

    # Returns n images by doing all the preprocessing
    def augmentation(self, picture_name, verbose=False):
        image = cv2.imread(f"{self.paths.PICTURES}/{picture_name}")
        picture = Picture(image, picture_name)

        # 2 extra brightness pictures
        higher_brightness = self.preprocessor.brightness(image, 1.5)
        hb_picture = Picture(higher_brightness, f"{picture_name}_hb.png")
        lower_brightness = self.preprocessor.brightness(image, 0.5)
        lb_picture = Picture(lower_brightness, f"{picture_name}_lb.png")
        self.train_test_split_and_save_pictures([picture, hb_picture, lb_picture])

        # 3 binarized imgs
        auto_binarized_img = self.preprocessor.binarize(image)
        low_binarized_img = self.preprocessor.binarize(image, th=140)
        high_binarized_img = self.preprocessor.binarize(image, th=180)

        # 3 borders_imgs
        auto_borders = self.preprocessor.ext_border(auto_binarized_img)
        low_borders = self.preprocessor.ext_border(low_binarized_img)
        high_borders = self.preprocessor.ext_border(high_binarized_img)

        # 3 shadow_imgs

        if verbose:
            pass
        return None #[binarized_img, borders]

    def train_test_split_and_save_pictures(self, pictures):
        dst = self.paths.AUGMENTED_DATASET

        rng = np.random.default_rng()
        rng.shuffle(pictures)

        length = len(pictures)
        train = pictures[: int(length * 0.67)]
        test = pictures[int(length * 0.67):]

        for train_picture in train:
            train_image = train_picture.image
            train_name = train_picture.name
            cv2.imwrite(train_image, f"{dst}/images/train/{train_name}.png")
            
        for test_img in test:
            test_image = test_img.image
            test_name = test_img.name
            cv2.imwrite(test_image, f"{dst}/images/test/{test_name}.png")
            
    # Changes original txt line into yolo line
    def line_parser(self):
        pass
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
