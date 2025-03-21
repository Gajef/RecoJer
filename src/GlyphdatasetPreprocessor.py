from PathsProvider import PathsProvider
import cv2
import numpy as np
from matplotlib import pyplot as plt

class GlyphdatasetPreprocessor:
    def __init__(self):
        self.paths = PathsProvider()

    # Returns n images by doing all the preprocessing
    def full_augmentation(self, hieroglyphic, verbose=False):
        binarized_img = self.binarize(hieroglyphic)
        borders = self.ext_border(binarized_img)
        if verbose:
            pass
        return [binarized_img, borders]

    # Binarizes an image
    def binarize(self, img, verbose = False):
        print(img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binarized = cv2.erode(th, (3, 3), iterations=1)
        if verbose:
            plt.figure(figsize=(15, 11))
            plt.imshow(binarized, cmap='gray')
            plt.show()
        return binarized

    # Gets the exterior borders of a binarized image
    def ext_border(self, binarized_img, verbose = False):
        canny_borders = cv2.Canny(binarized_img, 100, 200)
        gray_img = (canny_borders * 0.75).astype(np.uint8)
        borders = np.invert(gray_img)
        if verbose:
            plt.imshow(borders)
            plt.show()
        #ext_image = cv2.erode(binarized_img,(3,3), iterations=2)
        return borders
