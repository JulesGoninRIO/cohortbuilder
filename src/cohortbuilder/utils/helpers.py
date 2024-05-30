"""
This module includes general helper functions that can be used in all contexts.
"""

import io
import json
import sys
import pathlib
from typing import BinaryIO, Callable, Union, TypeVar, Iterable, Iterator, Tuple
from typing_extensions import ParamSpec
from functools import wraps
from itertools import islice
import os
import tempfile
import hashlib
import string
from pydicom import dcmread

from loguru import logger
import numpy as np
from IPython import get_ipython
from lxml import etree
from PIL import Image
from scipy.ndimage.interpolation import rotate
from svglib.svglib import load_svg_file
from jsonschema import validate, ValidationError
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

# Declare type and parameter specification variables
T = TypeVar('T')
P = ParamSpec('P')

# TODO: Handle the "x_order_2: colinear!" message
def bypass_outs(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator for bypassing the printed outputs of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Store the old outputs
        stdout = sys.stdout
        stderr = sys.stderr

        # Replace the outputs
        file = io.StringIO()
        sys.stdout = file
        sys.stderr = file

        # Execute the function
        results = func(*args, **kwargs)

        # Restore the old outputs
        sys.stdout = stdout
        sys.stderr = stderr

        return results

    return wrapper

def log_errors(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator for logging the errors occuring in a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Execute the function
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error and then raise it
            logger.exception(e)
            raise e

    return wrapper

def list2str(items: list, delimeter: str = ', ') -> str:
    """
    Returns an string constructed by string representation of the elements of an input list.

    Args:
        items: The list of the items
        delimeter: Delimeter to separate the elements. Defaults to ', '.

    Returns:
        The string representation of the list.
    """

    return delimeter.join([str(item) for item in items])

def str2list(line: str, n: int = 50) -> list[str]:
    """Breaks a long string into multiple strings of a maximum width.

    Args:
        line: The long string.
        n: The maximum width. Defaults to 50.

    Returns:
        The splitted strings.
    """

    return [line[i:i+n] for i in range(0, len(line)-n+1, n)] + [line[len(line)-(len(line) % n):]]

def print_with_linebreak(msg: str, n: int) -> None:
    """Prints a message with line breaks after a maximum number of characters ``n``."""

    print('\n\t\t'.join(str2list(line=msg, n=n)))

def batched(iterable: Iterable[T], chunk_size: int) -> Iterator[Tuple[T]]:
    """Gets an iterable and returns an iterator of chunks of its elements."""

    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk

def enumerate_path(p: pathlib.Path) -> pathlib.Path:
    """
    Returns the input path with an index appended to it depending on
    the number of the already existing paths.
    """

    occ = len(list(p.parent.glob(f'{p.stem}_*{p.suffix}')))
    return p.parent / f'{p.stem}_{occ+1:02d}{p.suffix}'

def read_json(file: Union[str, pathlib.Path], permission: str = 'r') -> Union[list, dict]:
    """Reads a jason file and returns the object."""

    assert permission in ['r', 'rb']
    with open(file, permission) as f:
        obj = json.load(f)

    return obj

def read_img(file: Union[pathlib.Path, str, BinaryIO]) -> Union[np.ndarray, etree._Element]:
    """Reads an image from a file or a binary IO object."""

    # Convert str to Path object
    if isinstance(file, str):
        file = pathlib.Path(file)

    # Read a file
    if isinstance(file, pathlib.Path):
        if file.suffix.lower() in ['.svg', '.svgz']:
            svg = load_svg_file(str(file))
            for p in svg[1]:
                p.attrib['fill-opacity'] = '1'
            return svg
        else:
            img = Image.open(file)
            img = np.asarray(img)
            return img

    # Read from a file-like object
    elif all([
            hasattr(file, 'read'),
            hasattr(file, 'seek'),
    ]):
        file.seek(0)
        img = Image.open(file)
        img = np.asarray(img)
        return img

def is_notebook() -> bool:
    """Determines whether a module is called in the context of ipython notebooks."""

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

def rotate_cor(cor: tuple, angle: float, shape: tuple) -> tuple:
    """Converts coordinates to coordinates in rotated frames.

    Args:
        cor: Coordinates in the initial frames.
        angle: Angle of rotations in degrees.
        shape: Shape of the initial frames.

    Returns:
        Coordinates in unrotated frames.
    """

    # Get the shape of the rotated frames
    shape_rot = rotate(np.zeros(shape=shape), angle=angle).shape

    # Read the coordinates and the angle
    h, w = cor
    angle_r = np.radians(angle)

    # Get the coordinates in the rotated frames
    g = shape[0] - h
    W_q = w * np.cos(angle_r) - g * np.sin(angle_r)
    G_q = w * np.sin(angle_r) + g * np.cos(angle_r)

    # Define shifted coordinates
    if angle >= 0:
        W = W_q + shape[0] * np.sin(angle_r)
        G = G_q
    else:
        W = W_q
        G = G_q - shape[1] * np.sin(angle_r)
    H = shape_rot[0] - G

    # Round the calculated coordinates
    H = int(np.round(H))
    W = int(np.round(W))

    return H, W

def recover_cor(cor: tuple, angle: float, shape: tuple) -> tuple:
    """
    Recovers coordinates from coordinates in rotated frames.

    Args:
        cor: Coordinates in rotated frames.
        angle: Angle of rotations an degrees.
        shape: Shape of the initial frames.

    Returns:
        Coordinates in the initial frames.
    """

    # Get the shape of the rotated frames
    shape_rot = rotate(np.zeros(shape=shape), angle=angle).shape

    # Read the coordinates and angle
    H, W = cor
    G = shape_rot[0] - H
    angle_r = np.radians(angle)

    # Define shifted coordinates
    if angle >= 0:
        W_q = W - shape[0] * np.sin(angle_r)
        G_q = G
    else:
        W_q = W
        G_q = G + shape[1] * np.sin(angle_r)

    # Recover unrotated coordinates
    w = W_q * np.cos(-angle_r) - G_q * np.sin(-angle_r)
    g = W_q * np.sin(-angle_r) + G_q * np.cos(-angle_r)
    h = shape[0] - g

    # Round the calculated coordinates
    h = int(np.round(h))
    w = int(np.round(w))

    return h, w

def thumbnail2angles(svg: etree._Element) -> list[float]:
    """Calculates the angle of the thumbnail lines. Works on ``OCT_LINE`` and ``OCT_STAR``.

    Args:
        svg: The thumbnail SVG read with read_img.

    Returns:
        The list of the line angles.
    """

    # Check length
    if len(svg) != 3:
        return None

    # Calculate angles
    angles = list()
    for p in svg[2]:
        if all([
            'x1' in p.attrib,
            'x2' in p.attrib,
            'y1' in p.attrib,
            'y2' in p.attrib,
        ]):
            x1 = float(p.attrib['x1'])
            x2 = float(p.attrib['x2'])
            y1 = float(p.attrib['y1'])
            y2 = float(p.attrib['y2'])
            if x1 == x2:
                angle = 90.
            elif y1 == y2:
                angle = 0.
            else:
                angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))
        else:
            angle = None
        angles.append(angle)

    return angles

def read_and_validate_json(file: Union[str, pathlib.Path], schema: Union[list, dict], permission: str = 'r') -> Union[list, dict]:
    gathered_json = read_json(file=file, permission=permission)
    try:
        validate(instance=gathered_json, schema=schema)
    except ValidationError as e:
        msg = f"Incorrect configuration given: {e}."
        logger.critical(msg)
        print(msg)
        sys.exit(2) # Exit code for incorrectly-passed configuration options

    return gathered_json

def upload_stuck_dir_decorator_factory(parser):
    '''
    Factory for the upload_stuck_dir_decorator so that parsers can be passed as an argument to the decorator at runtime.
    '''
    def upload_stuck_dir_decorator(func):
        '''
        Wrapper to prepare the folder in which failed uploads are outputted, on top of outputting info about what's inside upon termination.
        If the failed uploads folder is empty when the program ends, it is removed.
        '''

        @wraps(func)
        def inner(*args, **kwargs):
            FAILED_UPLOADS_DIR = pathlib.Path(parser.settings['general']['cohorts_dir'] + '/' + parser.args.logs_folder.parent.stem + '/' + parser.args.logs_folder.stem + '-rejects')
            FAILED_UPLOADS_DIR.mkdir()
            try:
                return_val = func(*args, **kwargs)
            except Exception as e:
                if not tuple(FAILED_UPLOADS_DIR.glob("*")): # We remove this folder if there was an error, and then continue with the error.
                    FAILED_UPLOADS_DIR.rmdir()
                raise e
            if not tuple(FAILED_UPLOADS_DIR.glob("*")):
                FAILED_UPLOADS_DIR.rmdir()
            else:
                print(f'{len(list(FAILED_UPLOADS_DIR.rglob("*")))} file(s) were stuck as pending, and have been gathered together at {FAILED_UPLOADS_DIR.as_posix()}. See the logs for more information.')
            return return_val
        return inner
    return upload_stuck_dir_decorator

def threading_locked_decorator_factory(lock):
    '''
    Factory for the theading_locked decorator so that an arbitrary lock can be passed to the decorator at runtime.
    '''
    @contextmanager
    def threading_locked(func):
        '''
        Execute the function while locked by a threading lock
        '''
        @wraps(func)
        def inner(*args, **kwargs):
            with lock:
                ret = func(*args, **kwargs)
            return ret
        return inner
    return threading_locked
                
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

@contextmanager
def suppress_stderr_only():
    """A context manager that redirects only stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:
            yield err

class strong_suppress_stderr_only(object):
    """Context to capture stderr at C-level.
    """

    def __init__(self):
        self.orig_stderr_fileno = sys.__stderr__.fileno()
        self.output = None

    def __enter__(self):
        # Redirect the stdout/stderr fd to temp file
        self.orig_stderr_dup = os.dup(self.orig_stderr_fileno)
        self.tfile = tempfile.TemporaryFile(mode='w+b')
        os.dup2(self.tfile.fileno(), self.orig_stderr_fileno)

        # Store the stdout object and replace it by the temp file.
        self.stderr_obj = sys.stderr
        sys.stderr = sys.__stderr__

        return self

    def __exit__(self, exc_class, value, traceback):

        # Make sure to flush stdout
        print(end='', flush=True)

        # Restore the stdout/stderr object.
        sys.stderr = self.stderr_obj

        # Close capture file handle
        os.close(self.orig_stderr_fileno)

        # Restore original stderr and stdout
        os.dup2(self.orig_stderr_dup, self.orig_stderr_fileno)

        # Close duplicate file handle.
        os.close(self.orig_stderr_dup)

        # Copy contents of temporary file to the given stream
        self.tfile.flush()
        self.tfile.seek(0, io.SEEK_SET)
        self.output = self.tfile.read().decode()
        self.tfile.close()

def sha256sum(filename):
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def create_dicom_uid_from_file(filename):
    dicom_field_length = 38

    image_data = dcmread(filename)[(0x7FE0, 0x0010)].value # Pixel Data, as bytes
    hash_object = hashlib.sha256(image_data)

    bunch_of_numbers = ''.join([x for x in hash_object.hexdigest() if x in string.digits])

    if len(bunch_of_numbers) < dicom_field_length: bunch_of_numbers *= 2

    return bunch_of_numbers[:dicom_field_length]
