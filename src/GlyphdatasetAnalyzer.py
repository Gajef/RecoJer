from PathsProvider import PathsProvider
from os import listdir
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import numpy as np

class GlyphdatasetAnalyzer:
    def __init__(self):
        self.paths = PathsProvider()

    # Generate dataframes from a path
    def generate_dataframe(self, locations_path, verbose = False):
        entries = []
        txt_files = list(filter(lambda path: path.endswith(".txt"), listdir(locations_path)))
        if verbose:
            print(txt_files)
        for txt in txt_files:
            with open(locations_path + "/" + txt, 'r') as file:
                locations = file.read()
                locations = locations.split("\n")
                # Elimino la parte final (despues del punto)
                locations = map(lambda location: location.split(".")[0], locations)
                # Separo entre imagen, id y gardiner
                locations = list(map(lambda location: [location[:2]] + location[2:].split("_"), locations))
                entries += locations

        return pd.DataFrame(data=entries, columns=["page", "id", "gardiner"]).dropna()

    # Plots dataframes by its classes
    def plot_dataframe(self, dataframe):
        dataframe.value_counts(["gardiner"]).plot(kind='bar', figsize=(28, 5))
        plt.show()

    # Finds and returns a sub-dataframe from dataframe. If verbose, prints how many were found and shows the first one
    def find_hieroglyphics_by_gardiner(self, dataframe, gardiner_id, verbose = False):
        query = dataframe.loc[dataframe['gardiner'] == gardiner_id]
        if verbose:
            if not query.empty:
                print(f"Se han encontrado {len(query)} jeroglíficos con el id de Gardiner {gardiner_id}")
                first_row = query.iloc[0] # Primer jeroglífico del dataframe
                hg_path = self.paths.get_path_by_picture_number_gardiner(first_row["page"], first_row["id"], first_row["gardiner"])
                plt.imshow(cv2.imread(hg_path))
                plt.show()
            else:
                print(f"No se han encontrado jeroglíficos con el id de Gardiner {gardiner_id}")

        return query

    #
    def find_hieroglyphics_by_gardiner_classes(self, dataframe, gardiner_ids, verbose = False):
        query = dataframe.loc[dataframe['gardiner'].isin(gardiner_ids)]
        return query

    def find_hieroglyphics_with_at_least_n_occurrences(self, dataframe, n, verbose = False):
        query = dataframe['gardiner'].value_counts()
        gardiner_ids = query[query >= 50].index
        query = dataframe[dataframe['gardiner'].isin(gardiner_ids)]

        return np.array(gardiner_ids), query