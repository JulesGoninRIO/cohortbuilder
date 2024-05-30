"""
This module includes the class for capturing and handling the pause signal.
"""

import signal
from typing import Callable
from functools import wraps
import tqdm

class Pauser:
    """
    Decorator class for pausing (with ``CTRL+C``) and resuming a process.

    Examples:
        A pauser can be used in a loop of a class as a decorater:

        >>> from src.cohortbuilder.utils.pauser import Pauser
        >>>
        >>> class MyClassWithPauser:
        >>>
        >>>     pauser = Pauser()
        >>>
        >>>     def __init__(*args):
        >>>         ...
        >>>
        >>>     @pauser
        >>>     def launch(*args):
        >>>         for _ in range(100):
        >>>             ...  # Do the process
        >>>
        >>>             # Check if CTRL+C is pushed
        >>>             if self.pauser.paused:
        >>>                 ...
        >>>                 resume = self.pauser.pause()
        >>>                 if resume:
        >>>                     ...
        >>>                 else:
        >>>                     ...
        >>>                     break  # break the loop
    """

    def __init__(self):
        #: Flag for indicating that the process has to be paused
        self.paused: bool = False
        self.forced: bool = False

    def __call__(self, func: Callable[[], None]) -> Callable[[], None]:
        """Returns a decorator which redirects the SIGINT signal to the pause handler."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            default_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, handler=self._handler)
            print('(Press CTRL+C to pause or stop, then press again to force it.)')
            try:
                output = func(*args, **kwargs)
            finally:
                signal.signal(signal.SIGINT, default_handler)

            return output

        return wrapper

    def _handler(self, signum, frame) -> None:
        if not self.paused:
            tqdm.tqdm.write('>>>>> PAUSE SIGNAL DETECTED <<<<<')
            self.paused = True
        else:
            tqdm.tqdm.write('>>>>> FORCE SIGNAL DETECTED <<<<<')
            self.forced = True

    def pause(self) -> bool:
        """
        Asks the user if they want to resume the process and returns the answer.
        The process is blocked in the meantime.
        """

        # Get the command
        print('Would you like to resume (R) or stop (S) the process?', end=' ')
        answer = input('Answer: ')

        # Return True if YES
        if answer.strip().upper() in ['R', 'RESUME']:
            self.paused = False
            self.forced = False
            return True
        # Return False if NO
        elif answer.strip().upper() in ['S', 'STOP']:
            return False
        # Retry if the answer is unvalid
        else:
            print(f'"{answer}" is not a valid answer. Please answer again.')
            return self.pause()
