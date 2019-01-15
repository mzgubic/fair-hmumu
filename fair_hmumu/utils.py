import os

def makedir(d):
    """
    Creates a new directory if it doesn't exist.
    """
    if not os.path.exists(d):
        os.makedirs(d)

    return d

