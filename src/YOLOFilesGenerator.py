import gardiner2unicode
import numpy as np
from GlyphdatasetAnalyzer import GlyphdatasetAnalyzer
from PathsProvider import PathsProvider
from gardiner2unicode import GardinerToUnicodeMap, UnicodeGlyphGenerator

class YOLOFilesGenerator:
    def __init__(self, source, n_occurrences=50):
        self.source = source
        self.analyzer = GlyphdatasetAnalyzer()
        self.paths = PathsProvider()
        self.complete_dataframe = self.analyzer.generate_dataframe(self.paths.MANUAL_LOCATIONS)
        self.gardiner_ids, self.selected_dataframe = self.__select_classes_from_df_with_n_occurrences(n_occurrences)
        self.glyph_class_dictionary = self.__create_dictionary()

    # Translates txt from original form to YOLO form
    def translate_txt(self, original_file_path, new_file_path):
        with open(original_file_path, 'r') as original_file, open(new_file_path, 'w') as new_file:
            for line in original_file:
                line.strip()
                if len(line.split(",")) == 5:
                    image_name, x_top, y_top, x_bottom, y_bottom = line.split(",")
                else:
                    image_name, x_top, y_top, x_bottom, y_bottom, _ = line.split(",")
                gardiner_id = image_name.split(".")[0].split("_")[1]
                if gardiner_id in self.gardiner_ids:
                    class_by_id = self.glyph_class_dictionary[gardiner_id]

                    x_center = ((int(x_top) + int(x_bottom)) / 2) / 1150
                    y_center = ((int(y_top) + int(y_bottom)) / 2) / 1600
                    width = (int(x_bottom) - int(x_top)) / 1150
                    height = (int(y_bottom) - int(y_top)) / 1600


                    new_file.write(f"{class_by_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        original_file.close()
        new_file.close()
        
    def generate_yaml(self, path):
        with open("data.yaml", "w") as data_yaml:
            lines = [f"path: {path}\n"
                  f"train: images/train\n",
                  f"val: images/val\n",
                  "\n",
                  f"names:\n"
                  ]

            for key, value in self.glyph_class_dictionary.items():
                lines.append(f"    {value}: {key}\n")

            data_yaml.writelines(lines)

    def generate_txt(self, annotations, filename):
        with open(self.paths.ASCII_AUGMENTATION + "/labels/" + filename, "w") as text_file:
            for annotation in annotations:
                text_file.write(f"{annotation}\n")

        text_file.close()

    def __create_dictionary(self):
        classes_number = np.arange(0, len(self.gardiner_ids))
        dictionary = {}
        for name, number in zip(self.gardiner_ids, classes_number):
            dictionary[name] = number

        return dictionary


    def __select_classes_from_df_with_n_occurrences(self, n_occurrences):
        if self.source == "glyphdataset":
            result = self.analyzer.find_hieroglyphics_with_at_least_n_occurrences(self.complete_dataframe, n_occurrences)
        else:
            ids = GardinerToUnicodeMap().gardiner2unicode.keys()
            dataframe = None
            result = [ids, dataframe]
        return result