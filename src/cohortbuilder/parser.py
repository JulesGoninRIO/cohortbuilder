"""
This module includes the parser class for storing the settings, the arguments
passed on the command line, and the configurations.
"""

import argparse
import getpass
import pathlib

from src.cohortbuilder.utils.helpers import log_errors


class Parser:
    """
    Parser class for reading and storing arguments,
    settings, and configurations.

    Examples:
        It can be initialized in the main script:

        >>> from src.parser import Parser
        >>> from src.cohortbuilder.utils.helpers import read_json
        >>>
        >>> args = parser.parse_args()
        >>> settings = read_json('settings.json')
        >>> configs = read_json('configs/template.json')
        >>> Parser.store(args=args, settings=settings)
        >>> Parser.configs = configs

        And be used in another module:

        >>> from src.parser import Parser
        >>> settings = Parser.settings

    .. seealso::
        :ref:`Configurations <buildconfigs>`
            Fields of the configurations file for building.
    """

    #: Arguments (passed by command-line or GUI)
    args: argparse.Namespace = None
    #: General settings
    settings: dict = None
    #: Configurations
    configs: dict = None
    #: Paramenters
    params: dict = None

    @staticmethod
    def store(args: argparse.Namespace, settings: dict) -> None:
        """Reads the configurations and the settings and stores them if they are valid.

        Args:
            args: Argument parser with arguments passed by the command line.
            settings: Dictionary object containing the settings of the builder.
                It should be loaded from 'settings.json'.
        """

        # Check and store the settings
        settings = Parser.check_settings(settings)
        Parser.settings = settings

        # Check and store the arguments
        if args:
            args = Parser.check_args(args)
        Parser.args = args

        # Override the settings
        if args and 'threads' in args and args.threads:
            Parser.settings['general']['threads'] = args.threads

        # Instantiate and store the parameters
        Parser.params = {
            'availablethreads': Parser.settings['general']['threads'],
        }

    @log_errors
    # TODO: Add custom exceptions.
    def check_args(args: argparse.Namespace) -> argparse.Namespace:
        """
        Work in progress...
        Checks sanity of the arguments and fetches them from the settings if not passed.

        Args:
            args: Arguments to be checked.

        Returns:
            The checked (and possibly modified) arguments.
        """


        if args.command in ['upload-pids', 'uploads-dir']:

            if args.pids and 'soin' in args.instances:
                message = 'Uploading from Heyex image pools is not supported on the SOIN space.'
                raise Exception(message)

            # Fetch configs_dir if not passed
            if 'user' in args and args.user:
                username = args.user
            else:
                username = getpass.getuser().lower()
            if not args.configs_dir:
                args.configs_dir = pathlib.Path(Parser.settings['general']['configs_dir']) / username
                args.configs_dir.mkdir(parents=False, exist_ok=True)
            ...

            # Check the existance of the configs file
            if not args.configs.endswith('.json'):
                args.configs += '.json'
            configs_file = args.configs_dir / args.configs
            if not configs_file.exists():
                raise Exception(f'Configuration file ({configs_file}) does not exist.')


        if args.command == 'reprocess-workbook':
            # Fetch configs_dir if not passed
            if 'user' in args and args.user:
                username = args.user
            else:
                username = getpass.getuser().lower()
            if not args.configs_dir:
                args.configs_dir = pathlib.Path(Parser.settings['general']['configs_dir']) / username
                args.configs_dir.mkdir(parents=False, exist_ok=True)

            # Check the existance of the configs file
            if not args.configs.endswith('.json'):
                args.configs += '.json'
            configs_file = args.configs_dir / args.configs
            if not configs_file.exists():
                raise Exception(f'Configuration file ({configs_file}) does not exist.')


        elif args.command == 'build':
            # Fetch configs_dir and cohorts_dir if not passed
            if 'user' in args and args.user:
                username = args.user
            else:
                username = getpass.getuser().lower()
            if not args.configs_dir:
                args.configs_dir = pathlib.Path(Parser.settings['general']['configs_dir']) / username
                args.configs_dir.mkdir(parents=False, exist_ok=True)
            else:
                args.configs_dir = pathlib.Path(args.configs_dir)
            if not args.cohorts_dir:
                args.cohorts_dir = pathlib.Path(Parser.settings['general']['cohorts_dir']) / username
                args.cohorts_dir.mkdir(parents=False, exist_ok=True)
            else:
                args.cohorts_dir = pathlib.Path(args.cohorts_dir)

            # Check the existence of the Discovery instance in settings
            if args.instance not in Parser.settings['api'].keys():
                raise Exception(f'The settings for the Discovery instance "{args.instance}" does not exist.')

            # Check the existance of the configs file
            if not args.configs.endswith('.json'):
                args.configs += '.json'
            configs_file = args.configs_dir / args.configs
            if not configs_file.exists():
                raise Exception(f'Configuration file ({configs_file}) does not exist.')

            # Check the existance of the cohorts directory
            if not args.cohorts_dir.exists():
                raise Exception(f'Cohorts directory ({args.cohorts_dir}) does not exist.')

        return args

    @log_errors
    # TODO: Implement
    def check_settings(settings: dict) -> dict:
        """
        Work in progress...
        Checks sanity of the settings.

        Args:
            settings: Settings to be checked.

        Returns:
            The checked (and possibly modified) settings.
        """

        return settings
