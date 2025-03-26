from PathsProvider import PathsProvider
from gardiner2unicode import GardinerToUnicodeMap, UnicodeGlyphGenerator


class GlyphAsciiGenerator:
    def __init__(self):
        self.paths = PathsProvider()
