import os.path
class PathsProvider:
    def __init__(self):
        self.DATASET = "../data/GlyphDataset/Dataset"
        self.AUTOMATED = self.DATASET + "/Automated"
        self.MANUAL = self.DATASET + "/Manual"
        self.PICTURES = self.DATASET + "/Pictures"
        self.AUTOMATED_LOCATIONS = self.AUTOMATED + "/Locations"
        self.MANUAL_LOCATIONS = self.MANUAL + "/Locations"

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
