from PathsProvider import PathsProvider
import cv2
import numpy as np
from matplotlib import pyplot as plt

class GlyphdatasetPreprocessor:
    def __init__(self):
        self.paths = PathsProvider()

    def binarize(self, img, verbose = False, th = None):
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        if th is None:
            #Binariza de manera automática con otsu
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
            # Binariza con threshold
            _, th = cv2.threshold(blur, th, 255, cv2.THRESH_BINARY)

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

        return borders

    # Inverts an image
    def invert(self, img, verbose = False):
        inverted = cv2.bitwise_not(img)
        if verbose:
            plt.imshow(inverted)
            plt.show()
        return inverted

    # Cast a shadow on the hieroglyphics
    def shadow(self, gray_img, location="right",  verbose = False):
        shadow = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
        if location == "right":
            # X derivate on right borders
            shadow[shadow < 0] = 0
        else:
            # X derivate on left borders
            shadow[shadow > 0] = 0

        shadow = cv2.convertScaleAbs(shadow)
        shadow = self.invert(shadow)

        shadow = self.stack_images(gray_img, shadow)

        if verbose:
            plt.figure(figsize=(15, 11))
            plt.imshow(shadow, cmap='gray')
            plt.show()
        return shadow

    # Add two pictures together by adding the dark pixels of the second image
    def stack_images(self, base_gray, top_img, erode_iterations=4,  verbose = False):
        # base_gray = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)
        top_img = cv2.GaussianBlur(top_img, (5,5), 0)
        _, top_img = cv2.threshold(top_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top_img = cv2.GaussianBlur(top_img, (3,3), 0)
        top_img = cv2.erode(top_img, (10, 10), iterations=erode_iterations)
        alfa = (255 - top_img).astype(np.float64)
        alfa /= 255
        stacked = (base_gray * (1 - alfa) + top_img * alfa).astype(np.uint8)
        if verbose:
            plt.figure(figsize=(15, 11))
            plt.imshow(stacked, cmap='gray')
            plt.show()
        return stacked

    def remove_image_parts(self, binarized_img, verbose=False):
        masks_paths = self.paths.get_files_path_by_extention_in_order(self.paths.AUGMENTATION_MASKS_SMALL, "png")
        mask_path = masks_paths[np.random.randint(0, len(masks_paths))]
        mask = cv2.imread(mask_path)
        mask_binarized = self.binarize(mask)

        binarized_img[mask_binarized == 0] = 0
        borders_img = self.ext_border(binarized_img)

        if verbose:
            plt.figure(figsize=(15, 11))
            plt.imshow(borders_img, cmap='gray')
            plt.show()

        return borders_img

    def brightness(self, image, factor, verbose=False):
        brightness = (image*factor).astype(np.uint16)

        if verbose:
            plt.figure(figsize=(15, 11))
            plt.imshow(brightness, cmap='gray')
            plt.show()

        return brightness