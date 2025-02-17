import os
from math import floor
import platform
import sys
import tempfile
import urllib.request
import zipfile

from functools import reduce
import multiprocessing

from .logger import get_logger, debugger

logger = get_logger(__name__)
debug_parallel = debugger.get_debug_parallel_env()
global_temp_dir = None


def is_array_like(obj):
    """Determine if an object is of type array

    Parameters
    ----------
    obj : Object
        Object instance

    Returns
    -------
    Bool
        Whether the passed object is like an array or not
    """
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__') and not type(obj) == str


def str2bool(v):
    """Converts a string representation to its corresponding Boolean value. It can be used by the
    ``parser.add_argument`` function to include arguments that can receive a Boolean value.
    Accepted values for `v` are the following case insensitive strings:
        - True: yes, true, t, y, 1
        - False: no, false, f, n, 0
    E.g. ``parser.add_argument("--bool_arg", type=str2bool, nargs='?', const=True, default=True)``
    And the usage in the command line could be as follows: ``python my_script.py --bool_arg False``

    Parameters
    ----------
    v : str
        String value that will be converted to its Boolean representation

    Returns
    -------
    bool
        Boolean representation of `v`

    Raises
    ------
    Exception
        In case `v` is not a valid string representation of a Boolean value
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def get_chunk(elements, num_chunks, chunk_num, sort_elements=True):
    """Divide a set of `elements` into chunks so that they can be processed in different tasks.
    If `num_chunks` is None or `chunk_num` is None, the original elements will be returned.

    Parameters
    ----------
    elements : list
        List of elements to divide in tasks
    num_chunks : int or None
        Number of tasks between which you want to divide the elements
    chunk_num : int or None
        Current task number. Must be in range [1, `num_chunks`]
    sort_elements : bool, optional
        Whether to sort the elements before taking the chunk or not, by default True

    Returns
    -------
    list
        Resulting list of elements to be processed

    """
    if num_chunks is None or chunk_num is None:
        return elements

    _elements = sorted(elements) if sort_elements else elements
    if not 0 < chunk_num <= num_chunks:
        raise ValueError(f"chunk_num must be in [1, num_chunks]")
    chunk_size = floor(len(_elements) / num_chunks)
    first = chunk_size*(chunk_num-1)
    last = len(_elements) if chunk_num == num_chunks else chunk_size*chunk_num
    return _elements[first:last]

def parallel_exec(func, elements, **kwargs):
    """Function to perform the execution of `func` in parallel from the elements in `elements`,
    sending the arguments contained in `kwargs`.
    If any of those arguments is a callable, it will be called by sending it the `elem` element
    obtained from the iteration in `elements`.
    In case the environment variable `DEBUG_PARALLEL == True` the execution will be performed
    iteratively, allowing the debugging of the `func` function.

    Parameters
    ----------
    func : Callable
        Function to be executed in parallel (or iteratively)
    elements : Iterable
        Set of elements to iterate over to execute the `func` function
    **kwargs :
        Set of named arguments that will be sent to the `func` function, and that in case of
        depending on the elements obtained from the iteration of `elements` must be passed in the
        form of a callable, whose argument will be the element of the iteration
        (e.g. `lambda elem: elem` to send the element itself)
    """
    tuples = []
    warn_shown = False
    if 'debug_parallel' in kwargs:
        _debug = kwargs['debug_parallel']
        del kwargs['debug_parallel']
    else:
        _debug = debug_parallel
    for elem in elements:
        args = tuple([fld(elem) if callable(fld) else fld for fld in kwargs.values()])
        if _debug:
            if not warn_shown:
                logger.warning("You are currently debugging in parallel mode")
                warn_shown = True
            func(*args)
        else:
            tuples.append(args)
    if not _debug:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(func, tuples)


def get_temp_folder(preferred_name='ml'):
    global global_temp_dir

    if global_temp_dir is None:
        tmp_dir = '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()
        global_temp_dir = os.path.join(tmp_dir, preferred_name)
        os.makedirs(global_temp_dir, exist_ok=True)

    return global_temp_dir


def download_file(url, dest_filename=None, force_download=False, verbose=False):
    """Download an image from an URL and store it with the specified file name

    Parameters
    ----------
    url : str
        URL from where the image is downloaded
    dest_filename : str, optional
        File name or directory where the file will be stored, by default None
    force_download : bool, optional
        Whether or not to force the download, by default False
    verbose : bool, optional
        Whether or not to be verbose, by default False

    Returns
    -------
    str
        Downloaded file path
    """
    def _progress(count, block_size, total_size):
        perc = int(min(float(count * block_size) / total_size * 100.0, 100))
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (dest_filename, perc))
        sys.stdout.flush()
    if dest_filename is None:
        dest_filename = get_temp_folder()
    if not os.path.exists(dest_filename) or force_download or os.path.isdir(dest_filename):
        if os.path.isdir(dest_filename):
            dest_filename = os.path.join(dest_filename, url.split('?', 1)[0].split('/')[-1])
        dir_name = os.path.dirname(dest_filename)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest_filename, _progress)
        except Exception as e:
            logger.exception(f"Exception in retrieving file from {url}")
    elif verbose:
        logger.debug(f'Bypassing download of already-downloaded file {dest_filename}')

    return dest_filename


def unzip_file(input_file: str, output_folder: str = None) -> None:
    """Unzip a zipfile to the specified output folder, defaulting to the same location as
    the input file

    Parameters
    ----------
    input_file : str
        Input zip file
    output_folder : str, optional
        Folder where to extract all the files. If None, it will be the same location as
        `input_file`.
        By default None
    """
    if output_folder is None:
        output_folder = os.path.dirname(input_file)

    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(output_folder)

