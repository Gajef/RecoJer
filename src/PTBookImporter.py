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

class PTBookImporter:
    def __init__(self, book_path=path, gray=False):
        self.book_path = book_path
        self.gray = gray
        if gray:
            self.dst_path = "/imagenes_jpg_gris"
        else:
            self.dst_path = "/imagenes_jpg"

    def generate_images(self):
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
                        if self.gray:
                            image = images[i].convert('L')
                        else:
                            image = images[i]
                        # Save pages as images in the pdf
                        image.save(self.book_path + self.dst_path + '/doc_' + str(number_doc) + '_page_' + str(i) + '.jpg', 'JPEG')