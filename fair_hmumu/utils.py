import os
import pickle


def makedir(d):
    """
    Creates a new directory if it doesn't exist.
    """
    if not os.path.exists(d):
        os.makedirs(d)

    return d

class Saveable:

    def __init__(self, name):
        self.name = name

    @classmethod
    def classname(cls):
        return cls.__name__

    @classmethod
    def from_file(cls, path):
        print('--- Reading {} from {}' .format(cls.classname(), path))
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        print('--- Saving {} as {}' .format(self.classname(), path))
        with open(path, 'wb') as f:
            pickle.dump(self, f)

