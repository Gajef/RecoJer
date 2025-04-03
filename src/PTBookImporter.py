# Requires: pdf2image, poppler-utils
# pip install pdf2image
# sudo apt-get install poppler-utils

from pdf2image import convert_from_path
from os import listdir
from os.path import isfile, join
from PathsProvider import PathsProvider
# path='/home/vruiz/RecoJer/data/LibroMuertos/'
paths = PathsProvider()
path = paths.PYRAMIDTEXTS_BOOK
dst_path = "/imagenes_jpg_gris" # imagenes_jpg
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print(onlyfiles)

# Convierto pdf en jpg
for d in range(len(onlyfiles)):

    number_doc = int(onlyfiles[d].partition("PT ")[2][0])
    if number_doc > 1:  # Importo imágenes de todos los documentos menos del primero
        file_path = path + onlyfiles[d]
        # Store Pdf with convert_from_path function
        images = convert_from_path(file_path)

        for i in range(len(images)):
            if i > 1:
                # Save pages as images in the pdf
                images[i].save(path + dst_path + '/doc_' + str(number_doc) + '_page_' + str(i) + '.jpg', 'JPEG')