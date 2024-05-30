#!/opt/miniconda3/envs/cb/bin/python

"""
The main script for launching the program.
"""

VERSION_NUMBER = '1.1.3'
VERSION_NAME = 'stable'

import argparse
import os
import getpass
import platform
import pathlib
import pytz
import warnings
from datetime import datetime
from loguru import logger
import signal
import sys
from cryptography.utils import CryptographyDeprecationWarning
from threading import Thread
from http.server import HTTPServer
from shutil import get_terminal_size
from tqdm.auto import tqdm

# Ignore a specific warning
warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
# Disable changing the console control handler
# NOTE: This is necessary for the CTRL+C event
# NOTE: Fortran runtime library (used by Scipy) overrides it
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# Change the working directory
# NOTE: Allows for running the script from any directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from src.cohortbuilder.builder import Builder, Uploader
from src.cohortbuilder.discovery.manager import DiscoveryManager
from src.cohortbuilder.managers import HeyexMetadataManager
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.files import DicomFile, process_fda_or_e2e_file, GenericFileUploadWrapper
from src.cohortbuilder.utils.helpers import read_json, read_and_validate_json, upload_stuck_dir_decorator_factory
from src.cohortbuilder.tools.list_and_delete_pending import get_stuck_datasets, prompt_for_acquisiton_selection, delete_acquisition_and_file, clear_empty_studies_in_workbook, clear_empty_patients_from_workbook
from src.cohortbuilder.definitions import CONFIGURATION_SCHEMA
from src.cohortbuilder.server import SERVER_ADDRESS, CohortBuilderServerRequestHandler, server_handle_jobs_forver, send_job_to_server, list_jobs_from_server, kill_ongoing_job, job_comm_queue, reverse_comm_queue
from src.cohortbuilder.tools.argumentparser import get_parser
from src.cohortbuilder.tools.logtracker import LogTracker


def is_launch_allowed():
    """Prints a warning and returns True if the user wants to launch the process during busy hours."""

    # Return True if not during the busy hours
    hour = datetime.now().hour
    if hour < Parser.settings['general']['busyhours_start'] or hour >= Parser.settings['general']['busyhours_end']:
        return True

    # Print a message and get an answer from the user
    print('Warning: Launching Cohort Builder during the busy hours might disrupt clinical procedures.'
        f'\nConsider launching it after {Parser.settings["general"]["busyhours_end"]}:00.'
        '\nType CONTINUE if you want to launch the process anyway: ', end='')
    answer = input()

    # Return True if CONTINUE
    if answer.strip().upper() == 'CONTINUE':
        return True
    # False otherwise
    else:
        return False

def launch_setup():
    """Launches the setup process based on the subcommand."""

    if Parser.args.extract_heyex_metadata:
        HeyexMetadataManager.init()
        HeyexMetadataManager.update()
    
    if Parser.args.delete_pending_datasets:
        msg = ''
        if Parser.args.instance is None:
            msg += 'Instance was not specified via CLI, please re-run with -i option completed.\n'
        if Parser.args.project is None:
            msg += 'Project was not specified via CLI, please re-run with -p option completed.\n'
        if Parser.args.workbook is None:
            msg += 'Workbook was not specified via CLI, please re-run with -w option completed.\n'
        if msg:
            logger.error(msg)
            print(msg)
            sys.exit(2) # Bad args code

        with DiscoveryManager(
                instances=[Parser.args.instance],
                projectname=Parser.args.project,
                workbookname=Parser.args.workbook,
                permission='read',
            ) as manager:
            acquisitions, msg = get_stuck_datasets(manager.workbooks[0], list_all_not_just_stuck=False)
            if not acquisitions:
                print('Nothing to do, everything looks good!')
                return
            else:
                print(msg)

            if Parser.args.all:
                to_del_idxs = list(range(len(acquisitions)))
            else:
                to_del_idxs = prompt_for_acquisiton_selection(acquisitions)

            disco = manager.discoveries[Parser.args.instance]
            wb = manager.workbooks[0]

            for i in tqdm(to_del_idxs, desc='Removing acquisitions'):
                delete_acquisition_and_file(disco, wb, acquisitions[i])
            
            clear_empty_studies_in_workbook(disco, wb) # Clean up
            clear_empty_patients_from_workbook(disco, wb)


@upload_stuck_dir_decorator_factory(parser=Parser)
def launch_upload_dir():
    """Launches the upload process from a directory."""

    # Get the directory
    dir = pathlib.Path(Parser.args.dir)
    if not dir.exists():
        msg = f'The file directory {dir.as_posix()} does not exist.'
        logger.error(msg)
        print(msg)
        return

    if Parser.args.do_not_convert_to_dicom:
        msg = 'do_not_convert_to_dicom flag is set, will not convert FDA/E2E files to DICOM. Upload to anonymized instances is disabled.'
        logger.warning(msg)
        print(msg)

    # Get the path of the files
    uploadfiles = []
    for file in dir.iterdir():
        if file.is_file():
            if file.suffix.lower() == '.dcm':
                uploadfiles.append(DicomFile(path=file, mode='local'))
            if file.suffix.lower() == '.fda':
                try:
                    uploadfiles += process_fda_or_e2e_file(path_to_file=file, mode='local') if not Parser.args.do_not_convert_to_dicom else [GenericFileUploadWrapper(path=file, mode='local')]
                except (ValueError, TypeError) as e:
                    logger.warning(f'Failed to convert FDA->DCM for file {file.name}, skipping. (No fundus or OCT data inside?)')
                    continue
            if file.suffix.lower() == '.e2e':
                try:
                    uploadfiles += process_fda_or_e2e_file(path_to_file=file, mode='local') if not Parser.args.do_not_convert_to_dicom else [GenericFileUploadWrapper(path=file, mode='local')]
                except (ValueError, TypeError):
                    logger.warning(f'Failed to convert E2E->DCM for file {file.name}, skipping.')
                    continue

    # TODO: Add filtering the files
    # TODO: Unify the codes with launch_upload_pids to avoid code repetition

    # Upload the files
    for workbook in Parser.args.workbooks:
        with DiscoveryManager(
            instances=Parser.args.instances,
            projectname=Parser.args.project,
            workbookname=workbook,
            permission='write',
        ) as manager:
            manager.upload(
                files=uploadfiles,
                anonymize=[
                    instance for instance in Parser.args.instances
                    if Parser.settings['api'][instance]['anonymize']
                ],
                better_anonymisation=Parser.args.strong_anonymisation
            )

@upload_stuck_dir_decorator_factory(parser=Parser)
def launch_upload_pids():
    """Launches the upload process."""

    # Read the configurations
    configs_file = Parser.args.configs_dir / Parser.args.configs
    configs = read_and_validate_json(configs_file, schema=CONFIGURATION_SCHEMA)
    # Log configs
    logger.info(f'Reading configurations from {configs_file.as_posix()}.')
    logger.info(f'Configurations: {configs}')

    if Parser.args.do_not_convert_to_dicom:
        msg = 'do_not_convert_to_dicom flag is set, will not convert FDA/E2E files to DICOM. Upload to anonymized instances is disabled.'
        logger.warning(msg)
        print(msg)

    # Run the builder
    with Uploader(
        configs=configs,
        instances=Parser.args.instances,
        project=Parser.args.project,
        workbooks=Parser.args.workbooks,
    ) as uploader:
        uploadfiles = uploader.get_upload_files(
            pids=Parser.args.pids,
            pidsfile=Parser.args.pidsfile,
            updatemetadata=Parser.args.updatemetadata,
            copyfiles=Parser.args.copy,
        )
        uploadfiles = uploader.filter_files(uploadfiles)
        uploader.upload(uploadfiles,better_anonymisation=Parser.args.strong_anonymisation)

def launch_reprocess_workbook():
    """Launches the reprocess-workbook process."""

    # Read the configurations
    configs_file = Parser.args.configs_dir / Parser.args.configs
    Parser.configs = read_and_validate_json(configs_file, schema=CONFIGURATION_SCHEMA)
    # Log configs
    logger.info(f'Reading configurations from {configs_file.as_posix()}.')
    logger.info(f'Configurations: {Parser.configs}')

    with DiscoveryManager(
        instances=[Parser.args.instance],
        projectname=Parser.args.project,
        workbookname=Parser.args.workbook,
        permission='read',
    ) as manager:
        # Detect failed / pending acquisitions
        for workbook in manager.workbooks:
            acquisitions = workbook.get_acquisitions(separate=True, verbose=True)
            msg = f'Datasets in {repr(workbook)} :: ' + ' : '.join([
                f'{len(acquisitions["all"])} TOTAL',
                f'{len(acquisitions["pending"])} PENDING',
                f'{len(acquisitions["failed"])} UNSUCCESSFUL',
            ])
            logger.info(msg)
            print(msg)

        # Reprocess the acquisitions using a relaunch manager
        if not Parser.args.onlyreport and acquisitions['failed']:
            msg = 'Relaunching the unsuccessful processes.'
            logger.info(msg)
            print(msg)
            reprocess_acquisition = []
            for key, istrue in Parser.configs['status'].items():
                if istrue:
                    reprocess_acquisition.extend(acquisitions[key])
            manager.reprocess(acquisitions=reprocess_acquisition)

def launch_build():
    """Launches the build process."""
    import json

    # Read the configurations
    configs_file = Parser.args.configs_dir / Parser.args.configs
    configs = read_and_validate_json(configs_file, schema=CONFIGURATION_SCHEMA)
    configs['general']['taxonomy'] = False  # TODO: Remove from the configs if not relevant anymore

    if 'detect_fovea_and_recalculate_stats' in configs['general'] and configs['general']['detect_fovea_and_recalculate_stats']:
        if not configs['general']['post_process_segmentations']:
            logger.warning('detect_fovea_and_recalculate_stats requires post_process_segmentations, including')
            configs['general']['post_process_segmentations'] = True

    if 'post_process_segmentations' in configs['general'] and configs['general']['post_process_segmentations']:
        if not configs['types']['oct']:
            logger.warning('OCT download is required for post_process_segmentations, including')
            configs['types']['oct'] = True
        if not configs['types']['segmentation']:
            logger.warning('segmentation download is required for post_process_segmentations, including')
            configs['types']['segmentation'] = True

    # Log configs
    logger.info(f'Reading configurations from {configs_file.as_posix()} (Copy made in output).')
    output_dir = pathlib.Path(Parser.args.cohorts_dir / configs['general']['name'])
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'config.json', 'a+') as f:
        f.write(f'Config for run @ {datetime.now()}:\n')
        json.dump(configs, f, indent=4) # Save backup of config, in case someone changes the one which was used here.
        f.write('\n')
    with open(output_dir / 'cmdline.txt', 'a+') as f:
        f.write(f'Command line for run @ {datetime.now()}: {Parser.args}\n') # Save backup of command-line, in case we lose the logs.
    logger.info(f'Configurations: {configs}')

    # Run the builder
    with Builder(
        configs=configs,
        instance=Parser.args.instance,
        project=Parser.args.project,
        workbooks=Parser.args.workbooks,
        noconfirm_resume=Parser.args.noconfirm_resume
    ) as builder:
        if builder.create_folders():
            builder.get_metadata()
            builder.build()

@logger.catch(onerror=lambda _: sys.exit(1))
def launch_server():
    '''
    Launches the server, and waits for jobs.
    Does not need configuration files for now. (Handled already in main)
    '''
    # TODO: Add configuration option which changes the server pipe path (FIFO_PATH in main).

    def handle_server_end(signum, frame):
        msg = 'Server shutting down from SIGTERM.'
        logger.warning(msg)
        print(msg)
        httpd.shutdown()
        sys.exit(0) # Give clean exit signal

    signal.signal(signal.SIGTERM, handle_server_end) # Register clean-up. Overrides the registration below.
    # SIGTERM is called by systemd when terminating a daemon process, so this handles that. 

    # Setup up CB Req handler class
    CohortBuilderServerRequestHandler.job_communication_queue = job_comm_queue
    CohortBuilderServerRequestHandler.reverse_communication_queue = reverse_comm_queue
    CohortBuilderServerRequestHandler.logger = logger

    httpd = HTTPServer(
        server_address=SERVER_ADDRESS,
        RequestHandlerClass=CohortBuilderServerRequestHandler
    )
    
    http_handler_thread = Thread(target=httpd.serve_forever,
                                args=tuple(),
                                name='Server HTTP handler thread'
                               ) # Launch this in the background
    http_handler_thread.start()

    server_handle_jobs_forver(job_comm_queue, reverse_comm_queue, logger)


def handle_signal(signum, frame):
    msg = f'Got operating system signal {signum} ({signal.Signals(signum).name}). Terminating.\n'
    if signal.Signals(signum).value == signal.SIGINT:
        msg += 'SIGINT is an interrupt signal. You have probably voluntarily stopped the program.'
    elif signal.Signals(signum).value == signal.SIGTERM:
        msg += 'SIGTERM is a termination signal, but not a kill signal. The program was not killed from an out-of-memory incident.'
    elif signal.Signals(signum).value == signal.SIGHUP:
        msg += 'SIGHUP is a hangup signal. This means that you closed the terminal window which was running the program.'
    logger.critical(msg)
    print(msg)

    sys.exit(128 + signum) # Linux convention for signal-caused exit-codes.    

def main():
    # Get system-specific settings
    print(f'CohortBuilder version {VERSION_NUMBER}, {VERSION_NAME} release.')

    # Read the settings
    settings: dict[str, dict] = read_json('settings.json')

    # Read the keys and update the settings
    if pathlib.Path(settings['general']['keys']).exists():
        keys: dict[str, dict] = read_json(settings['general']['keys'])
        settings['slims'].update(keys['slims'])
        for api_name in keys['api']:
            settings['api'][api_name].update(keys['api'][api_name])
    else:
        message = (
            f'The path to the keys file ({settings["general"]["keys"]}) does not exist.'
            ' You can change the path in the settings and try again.'
        )
        print(message)
        exit()

    # Store args and settings
    args = get_parser().parse_args()
    Parser.store(
        args=args,
        settings=settings,
    )

    if 'client' in args and args.client:
        send_job_to_server() # This will exit the program once done. Will not setup logger, which is good!
        sys.exit(0)
    if args.command == 'server' and args.list_ongoing_jobs:
        list_jobs_from_server()
        sys.exit(0)
    if args.command == 'server' and args.kill_job > 0:
        kill_ongoing_job(args.kill_job)
        sys.exit(0)

    # Configure the logger settings
    logtracker = LogTracker(
        settings=settings['logging'],
        name=(args.command[0] if args.command else 'm'),
    )
    Parser.args.logs_folder = logtracker.folder

    # Orniments
    print('▼' * get_terminal_size().columns)

    # Check if launching is allowed
    allowed = is_launch_allowed() if args.command in ['upload'] else True
    if allowed:
        # Launch the corresponding process
        if 'user' in Parser.args and Parser.args.user:
            username = Parser.args.user
        else:
            username = getpass.getuser()
        logger.info(f'{args.command.upper()} launched by {username} with arguments: {Parser.args}')
        if args.command == 'setup':
            launch_setup()
        elif args.command == 'upload-dir':
            launch_upload_dir()
        elif args.command == 'upload-pids':
            launch_upload_pids()
        elif args.command == 'reprocess-workbook':
            launch_reprocess_workbook()
        elif args.command == 'build':
            launch_build()
        elif args.command == 'server':
            launch_server()
        else:
            raise Exception(f'Command not recognized: {args.command}.')

        # Print the final messages
        print('-' * get_terminal_size().columns)
        print(f'The {args.command or "main"} process is finished on {datetime.now().strftime("%d/%m/%y %H:%M:%S")}.'.upper())

    # Log tracker report
    logtracker.report()

    # Clear temp cache. Allows for reproducible behaviour, and cleans up the filesystem.
    tmp_cache_path = pathlib.Path(Parser.settings['general']['cache']) / 'tmp'
    files_in_tmp_cache = tuple(tmp_cache_path.rglob('*.*'))
    for f in files_in_tmp_cache:
        f.unlink()

    # Ornaments
    print('▲' * get_terminal_size().columns)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGHUP, handle_signal)
# We unfortunately cannot handle SIGKILL without some extra work-arounds, as by definition it cannot be caught or handled.
# As a work-around, we write a message to logs and STDOUT when the program successfully exits. If there is no information about the program exiting in the logs, it was killed!

if __name__ == '__main__':
    main()
    msg = 'Process successfully finished all tasks. Exiting.'
    logger.info(msg)
    print(msg)
    sys.exit(0) # Successful finish exit code
