import os

from cv_functions import read_image_data


class Person:
    def __init__(self):
        self.id = 0
        self.label = int(0)
        self.data = int(0)

    @staticmethod
    def to_person(folder, filename):
        p = Person()
        f, file_extension = os.path.splitext(filename)
        str_data = f.split("_")
        p.id = int(str_data[0])
        p.label = int(str_data[1])
        img = read_image_data(folder, filename)
        if img is not None:
            p.data = img
            return p
        return None



