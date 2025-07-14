import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import shutil
from skimage.feature import hog
from PathsProvider import PathsProvider

class HogClassifier:
    def __init__(self):
        self.paths = PathsProvider()

    def find_glyphs(self, image_path, method = 0):
        glyph_list = []
        detect_bboxes = []

        if method == 0: # FIND CONTOURS
            rgb_im = cv2.imread(image_path)
            gray = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2GRAY)
            # _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, 9)
            contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 120 > w > 10 and 120 > h > 10 and not (h < 20 and w < 20):
                    cv2.rectangle(rgb_im, (x, y), (x + w, y + h),
                                  (np.random.randint(255), np.random.randint(255), np.random.randint(255)), 2)
                    glyph = gray[y:y + h, x:x + w]
                    resize_glyph = cv2.resize(glyph, (32, 32))
                    glyph_list.append(resize_glyph)
                    detect_bboxes.append([x, y, x + w, y + h])

            with open(f"{self.paths.RESULTS}/contours/labels/{os.path.basename(image_path).split('.')[0]}.txt", mode = 'w') as file:
                detect_bboxes_str = [[str(coord) for coord in bbox] for bbox in detect_bboxes]
                for bbox in detect_bboxes_str:
                    file.write(",".join(bbox) + "\n")

        return glyph_list, detect_bboxes

    def compute_hog(self, glyph_list, method = 1):
        """
        Computes HoG for al glyphs on a list.

        :param method: 0 for CV2, >=1 for scikit-image
        :param glyph_list: List of glyphs.
        :return: HoG list.
        """
        hog_list = []

        if method == 0:
            hog_descriptor = cv2.HOGDescriptor(_winSize=(32, 32), _blockSize=(8, 8), _blockStride=(4, 4), _cellSize=(4, 4), _nbins=7)

            for glyph in glyph_list:
                descriptor = hog_descriptor.compute(glyph, winStride=(4, 4), padding=(0, 0))
                hog_list.append(descriptor)

        else:
            for glyph in glyph_list:
                descriptor, _ = hog(glyph, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                    visualize=True, transform_sqrt=True)
                hog_list.append(descriptor)

        return hog_list

    def group_by_hog_clustering(self, glyph_list, hog_list):
        """
        Given a list of glyphs and their hog descriptions, groups by clustering with kmeans.

        :param glyph_list: List of glyphs.
        :param hog_list: List of hog descriptors in the same order of the glyphs.
        :return: grouped ?
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(hog_list)

        kmeans_model = KMeans(n_clusters=220, random_state=42, n_init=10, max_iter=1000)
        kmeans_model.fit(X_scaled)

        hog_classification = kmeans_model.labels_

        base_folder = f"{self.paths.RESULTS}/hog_clustering/classification"

        # Borro las carpetas de antes
        for folder in os.listdir(base_folder):
            if os.path.isdir(f"{base_folder}/{folder}"):
                shutil.rmtree(f"{base_folder}/{folder}")

        # Guardo x carpetas segun la clasificacion del cluster
        for hog_class, glyph in list(zip(hog_classification, glyph_list)):
            new_dir_path = f"{base_folder}/{hog_class}"
            if not os.path.isdir(new_dir_path):
                os.makedirs(new_dir_path)
            cv2.imwrite(f"{new_dir_path}/{np.random.randint(5000)}.png", glyph)

    def group_by_hog_distance(self, glyph_list, hog_list):
        pass

    def classify_folder(self, path):
        pass
