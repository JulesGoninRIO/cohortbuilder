"""
This module includes the utility methods that are
used in the tests.
"""

import filecmp
import hashlib
from contextlib import contextmanager
import pathlib
from time import sleep
from typing import Callable
from threading import Thread
from typing import Union

from src.cohortbuilder.utils.helpers import list2str


def test_process() -> None:
    """A test that always passes."""

    assert True

def get_hash_tree(path: Union[pathlib.Path, str]) -> int:
    """Creates a hash for the structure of the files in a directory.

    Args:
        path (Union[pathlib.Path, str]): The path of the directory.

    Returns:
        int: The hash created from the folder tree of the directory.
    """

    path = pathlib.Path(path)
    files = [(f.relative_to(path)).as_posix() for f in path.glob('**/*') if f.is_file()]
    h = list2str(sorted(files))

    return hashlib.sha256(h.encode('ASCII')).hexdigest()

def get_hash_files(path: Union[pathlib.Path, str]) -> str:
    """Creates a hash from the content of all the files in a directory.

    Args:
        path (Union[pathlib.Path, str]): The path of the directory.

    Returns:
        int: The hash created from the files.
    """

    path = pathlib.Path(path)
    files = [f for f in path.glob('**/*') if f.is_file()]

    h = str()
    for file in sorted(files):
        with open(file, 'rb') as f:
            h += hashlib.sha256(f.read()).hexdigest()

    return hashlib.sha256(h.encode('ASCII')).hexdigest()

# UNUSED
def cmpdirs(org: pathlib.Path, new: pathlib.Path, shallow: bool = True) -> None:
    """Compares the files in two directories.

    Args:
        org (pathlib.Path): The source (original) directory.
        new (pathlib.Path): The new directory.
        shallow (bool, optional): If ``True``, metadata of the files won't be compared. Defaults to ``True``.
    """

    assert new.exists()
    for child in org.iterdir():
        child_ = new / child.relative_to(org)
        if child.is_file():
            if not filecmp.cmp(child, child_, shallow=shallow):
                pass
        else:
            cmpdirs(org=child, new=child_, shallow=shallow)

@contextmanager
def timeout(time: float = None, handler: Callable = None, handler_kwargs: dict = {}) -> None:
    """Context manager for setting a time limit on cohort builder.

    Args:
        time: Time limit in seconds. If ``None``, no limit is applied.
        handler: handler to execute on a parallel thread if time limit is reached.
        handler_kwargs: Keyword arguments to be passed to the handler.
    """

    def worker(time):
        sleep(time)
        if not done and handler:
            handler(**handler_kwargs)

    done = False
    if time is not None:
        t = Thread(target=worker, args=(time,))
        t.start()

    try:
        # Run the block
        yield
    finally:
        # Disable the error
        done = True
