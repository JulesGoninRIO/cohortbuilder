import pathlib
from datetime import datetime
import pytz
import getpass
from loguru import logger
from shutil import get_terminal_size

from src.cohortbuilder.parser import Parser
from src.cohortbuilder.utils.helpers import str2list

class LogTracker:
    """
    Class for managing the logger.

    Args:
        settings: Settings dictionary in the format of loguru configurations.
        name: A given name to this session.
    """

    def __init__(self, settings: dict, name: str) -> None:
        self.counter: dict[str, int] = {level: 0 for level in ['ERROR', 'WARNING', 'DEBUG']}
        self.settings: dict = settings
        self.name: str = name
        self.folder: pathlib.Path = self.configure()

    def callback(self, record: dict) -> None:
        """Callback function that is invoked before each record."""

        level = record['level'].name
        if level in self.counter.keys():
            self.counter[level] += 1

    def configure(self) -> pathlib.Path:
        """Reads the logger configurations and applying the filters.

        Returns:
            pathlib.Path: The folder of the logs of this session.
        """

        # Define the filters
        only_exceptions = lambda record: bool(record['exception'])
        no_exceptions = lambda record: not bool(record['exception'])

        # Create the log folder
        now = datetime.now(pytz.timezone('Europe/Zurich')).strftime('%Y%m%d.%H%M%S.%f')
        if 'user' in Parser.args and Parser.args.user:
            username = Parser.args.user
        else:
            username = getpass.getuser().lower()
        folder = pathlib.Path(self.settings.pop('root')) / username / '.'.join([now, self.name])
        folder.mkdir(parents=True, exist_ok=True)

        for handler in self.settings['handlers']:
            # Set the path of the file
            handler['sink'] = folder / handler['sink']

            # Delete the old log files (if any) and create new ones
            open(handler['sink'], 'w').close()

            # Set the filters
            if pathlib.Path(handler['sink']).stem == 'exceptions':
                handler['filter'] = only_exceptions
            else:
                handler['filter'] = no_exceptions

        # Apply the configurations
        logger.configure(**self.settings, patcher=self.callback)

        return folder

    def report(self) -> None:
        """Prints a summary of the logs."""

        # Print the counts
        if self.counter:
            elements: list[str] = []
            for level, count in self.counter.items():
                if not count: continue
                elements.append(f'{count} {level.upper()}')
                if count > 1:
                    elements[-1] += 'S'
            if elements:
                if len(elements) > 1:
                    msg = f'{", ".join(elements[:-1])} and {elements[-1]} have been logged.'
                else:
                    msg = f'{", ".join(elements)} {"has" if int(elements[0].split(" ")[0]) == 1 else "have"} been logged.'
                msg = '----> ' + msg + ' <----'
                print(msg)
                logger.info(msg)

        # Print the logs folder
        print(f'Check {self.folder.as_uri()} for more details.')
        print('Consider keeping the logs (or their path) for possible future references.')

    def print_warnings(self) -> None:
        """Prints errors and warnings."""

        file = self.folder / 'warnings.log'
        if file.exists() and file.stat().st_size:
            print('-' * Parser.settings['progress_bar']['description'])
            print('ERRORS AND WARNINGS:')
            with open(file, 'r') as f:
                for line in f.readlines():
                    # Break to multiplie lines and print
                    line = '\n\t\t'.join(str2list(line=line, n=get_terminal_size().columns))
                    print(f'\t{line}')
