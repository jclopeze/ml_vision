import os
import sys
import logging
import tempfile
import platform

from .env import ENV, ENVS

__all__ = ['get_logger']

LOGGING_PATH = os.environ.get('LOGGING_PATH')
if LOGGING_PATH is not None:
    if os.path.isdir(LOGGING_PATH):
        FILENAME = "ml.log"
        PATH = os.path.join(LOGGING_PATH, FILENAME)
    else:
        os.makedirs(os.path.dirname(LOGGING_PATH), exist_ok=True)
        PATH = LOGGING_PATH
else:
    # Verify if system is macOS
    foldername = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()
    foldername = os.path.join(foldername, 'logs')
    os.makedirs(foldername, exist_ok=True)
    FILENAME = "ml.log"
    PATH = os.path.join(foldername, FILENAME)

FORMAT = "%(asctime)s [%(name)-12s] [%(levelname)-5.5s]  %(message)s"
if os.environ.get('LOGGING_LEVEL', None) is not None:
    DEFAULT_LEVEL = logging._nameToLevel[os.environ['LOGGING_LEVEL']]
else:
    if ENV == ENVS.Prod:
        DEFAULT_LEVEL = logging.ERROR
    elif ENV == ENVS.Dev_Prod:
        DEFAULT_LEVEL = logging.INFO
    else:
        DEFAULT_LEVEL = logging.DEBUG
logFormatter = logging.Formatter(FORMAT)
logging.basicConfig(stream=sys.stderr, format=FORMAT)


def get_logger(name, path=PATH, level=DEFAULT_LEVEL):
    """Return a logger with the specified name.

    Parameters
    ----------
    name : str
        Name of the logger
    path : str, optional
        Path of the file for logging (default is PATH)
    level : int or str, optional
        Logging level of this logger (default is DEFAULT_LEVEL)

    Returns
    -------
    Logger
        Instance of a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    filename = path
    fileHandler = logging.FileHandler(filename, mode='a')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    return logger
