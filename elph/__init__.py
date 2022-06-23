__version__ = '1.0'
name = 'elph'

import os
import errno
from contextlib import contextmanager

def mkdir(folder):
    """Creates folder in current wd unless OSError occurs.

    Args:
        folder (str): Name of folder to create.
    """
    try:
        os.mkdir(folder)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

@contextmanager
def chdir(folder):
    """Changes the working directory to folder if not already current wd. 

    Args:
        folder (str): Name of folder to make wd.
    """
    dir = os.getcwd()
    os.chdir(str(folder))
    yield
    os.chdir(dir)
