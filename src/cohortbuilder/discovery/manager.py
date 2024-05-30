"""
This module includes the class for managing the connection to a Discovery server.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal
from shutil import get_terminal_size
import time
import pickle

from loguru import logger
import tqdm

from src.cohortbuilder.discovery.definitions import DiscoveryTask
from src.cohortbuilder.discovery.discovery import Discovery
from src.cohortbuilder.discovery.entities import Project
from src.cohortbuilder.discovery.queries import Q_PROJECTS
from src.cohortbuilder.discovery.exceptions import TokenRefreshMaxAttemptsReached, TokenAllCredentialsInvalid
from src.cohortbuilder.managers import UploadManager, ReprocessManager, AllThreadsDeadException
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.files import DicomFile
from src.cohortbuilder.definitions import UploadPipelineFileStatus
from src.cohortbuilder.utils.helpers import batched
from src.cohortbuilder.utils.pauser import Pauser
from src.cohortbuilder.tools.list_and_delete_pending import clear_empty_patients_from_workbook, clear_empty_studies_in_workbook


if TYPE_CHECKING:
    from src.cohortbuilder.discovery.file import Acquisition

class DiscoveryManager:
    """
    This class provides functionalities to work with (multiple) Discovery workbooks via API.
    It can (and should) be used as a context manager.

    Args:
        instances: The names of the Discovery instances.
        projectname: The name of the project.
        workbookname: The name of the workbook that exists in all instances.
        permission: If passed, the permissions of the workbooks will be checked.

    Raises:
        Exception: If one of the workbooks is not found on the instances.

    Examples:
        >>> from src.parser import Parser
        >>> from src.discovery.manager import DiscoveryManager
        >>>
        >>> ... # Get the settings and the arguments
        >>> Parser.store(args=args, settings=settings)
        >>>
        >>> with DiscoveryManager(
        ...     instances=['fhv_jugo', 'fhv_research'],
        ...     instances=['fhv_jugo', 'fhv_research'],
        ...     projectname='cohortbuilder',
        ...     workbookname='tmp',
        ... ) as manager:
        ...     manager.upload(files=files)

    .. seealso::
        `Discovery <src.discovery.discovery.Discovery>`
    """


    #: An instance from the decorator class for pausing and resuming the processes
    pauser: Pauser = Pauser()

    def __init__(self, instances: list[str], projectname: str, workbookname: str,
            permission: Literal['read', 'write', 'share'] = None):
        #: The name of the Discovery instances
        self.instances: set[list[str]] = set(instances)
        #: The upload manager
        self.uploader: UploadManager = None
        self.reprocesser: ReprocessManager = None

        # Instantiate the Discovery instances
        self.discoveries: dict[str, Discovery] = {instance: Discovery(instance=instance) for instance in instances}
        # Instantiate the Discovery projects
        projects = [
            Project(
                discovery=discovery,
                uuid=self.get_project_uuid(discovery=discovery, name=projectname),
                folder=None,
                name=projectname,
                )
            for discovery in self.discoveries.values()
        ]
        # Instantiate the Discovery workbooks
        workbook_name_matches = [
            [
                workbook for workbook in project.get_children()
                if workbook.attributes['name'].strip().lower() == workbookname.strip().lower()
            ]
            for project in projects
        ]
        # Make sure that there is only one workbook with this name
        for idx, workbooks in enumerate(workbook_name_matches):
            if len(workbooks) < 1:
                msg = (
                    f'There is no workbook with the name {workbookname.strip()} '
                    f'in Discovery instance "{projects[idx].discovery.instance}".'
                )
                raise Exception(msg)
            elif len(workbooks) > 1:
                msg = (
                    f'There are more than one workbook with the name {workbookname.strip()} '
                    f'in Discovery instance "{projects[idx].discovery.instance}".'
                )
                raise Exception(msg)
        # Get the first item of the workbooks and store it
        #: Workbooks of interest (can be on different instances)
        self.workbooks = [workbooks[0] for workbooks in workbook_name_matches]

        # Check workbook permissions
        if permission:
            for workbook in self.workbooks:
                if not workbook.attributes['currentPermission'][permission]:
                    msg = (
                        f'The workbook "{workbook.attributes["name"]}" does not have {permission.upper()} permissions. '
                        'Make sure to give the right permissions when you share a workbook with the Cohort Builder user on Discovery.'
                    )
                    raise Exception(msg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.uploader:
            # Stop the threads
            if self.uploader.isalive:
                self.uploader.kill()

        if self.reprocesser:
            # Stop the threads
            if self.reprocesser.isalive:
                self.reprocesser.kill()

    def _catch_errors(func: Callable[[], None]) -> Callable[[], None]:
        """Decorator for handling errors in the checking process."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the function
            try:
                return func(self, *args, **kwargs)
            except AllThreadsDeadException as e:
                logger.info('All the upload threads stopped working, possibly due to errors.')
            except (TokenRefreshMaxAttemptsReached, TokenAllCredentialsInvalid) as e:
                logger.info(f'Refreshing the token failed: {type(e).__name__}')
            except Exception as e:
                raise e

        return wrapper

    @staticmethod
    def get_project_uuid(discovery: Discovery, name: str) -> str:
        """Gets the UUID of a project on a Discovery instance.

        Args:
            discovery: The Discovery instance that contains the project.
            name: The name of the project.

        Raises:
            ValueError: If the project is not found or more than one
              has been found.

        Returns:
            The UUID of the project.
        """

        project_uuids = [
            project['uuid']
            for project in discovery.send_query(query=Q_PROJECTS)['data']['profile']['projects']
            if project['title'].strip().lower() == name.strip().lower()
        ]
        if len(project_uuids) == 1:
            return project_uuids[0]
        else:
            message = (
                f'There are {len(project_uuids)} projects '
                f'named "{name.strip().lower()}" '
                f'in Discovery instance {discovery.instance}.'
                f'\nMake sure the Cohort Builder user is a member of this project.'
            )
            raise ValueError(message)

    @_catch_errors
    @pauser
    def upload(self, files: list[DicomFile], anonymize: list[str] = [], better_anonymisation: bool = False) -> None:
        """Uploads dicom files to Discovery instances.

        Args:
            files: Remote or local dicom files that need to be uploaded.
            anonymize: Names of Discovery instances on which only anonymized files are allowed.
            remove: If ``True``, the local files will be removed afterwards.
        """

        # Set the upload manager
        bsz = Parser.settings['general']['upload_batch_size']
        self.uploader = UploadManager(
            n_threads=min(len(files), bsz, Parser.settings['general']['threads']),
            workbooks_raw=[workbook for workbook in self.workbooks if workbook.parent.discovery.instance not in anonymize],
            workbooks_anm=[workbook for workbook in self.workbooks if workbook.parent.discovery.instance in anonymize],
            pauser=self.pauser,
            better_anonymisation=better_anonymisation
        )
        self.uploader.launch()

        # Put the files in the uploader queue
        batches = batched(files, bsz)
        logger.info(f'Uploading the files in progress...')
        pbar = tqdm.tqdm(
            total=(len(files) // bsz) + (1 if (len(files) % bsz) else 0) * len(self.uploader.instances),
            desc=f'Processing the batches'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=False,
        )
        for idx, batch in enumerate(batches):
            # Create pbar for this batch
            pbar_batch = tqdm.tqdm(
                total=len(batch),
                desc=f'└── Uploading the files in the batch'.ljust(Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=False,
            )
            # Send the files in the batch to the uploader queues
            for file in batch:
                self.uploader.add(file=file, pbar=pbar_batch)
                file.status = UploadPipelineFileStatus.ENQUEUED

            # Wait until the queue is empty
            BATCH_TIMEOUT = Parser.settings['general']['pending_acquisition_timeout'] * 1.5  # Interval can go to 1.5x the settings interval (see UploadManager)
            UPLOAD_ATTEMPTS_MAX = Parser.settings['general']['upload_max_attempts']
            UNBLOCK_PENDING_FILES = Parser.args.unblock_pending_files
            if UNBLOCK_PENDING_FILES:
                BATCH_TIMEOUT *= UPLOAD_ATTEMPTS_MAX # Increase the batch timeout to the maximum amount of time we could wait for (n attempts).
            start = time.time()
            timeout_multiplier = len(batch) / Parser.args.threads # Scale timeout by mean number of files processed per thread
            timeout_multiplier = timeout_multiplier if timeout_multiplier > 1 else 1 # Handle cases where more threads than files in batch

            while (time.time() - start) < BATCH_TIMEOUT * 1.2 * timeout_multiplier: # Give a small time buffer for clean-up, and stop if goes over.
                if not self.uploader.q.unfinished_tasks:
                    break
                time.sleep(5)
            else:
                tqdm.tqdm.write('Batch seems to be stuck. Force-skipping the batch.')
                self.pauser.paused = True
                self.pauser.forced = True
            self.uploader.q.join()
            pbar_batch.close()
            pbar.update()

            # Pause if signaled
            if self.pauser.paused:
                # Hide the progress bars
                pbar.clear(nolock=True)

                # Print messages and log
                msg = 'Uploads to Discovery are paused.'
                logger.info(msg)
                print(msg)

                # Ask the user if they want to resume
                if self.pauser.pause():
                    msg = 'Uploads to Discovery are resumed.'
                    print(msg)
                    logger.info(msg)
                    # Unhide the progress bars
                    pbar.refresh(nolock=True)
                else:
                    print('Stopping the process in progress..')
                    logger.info('The process is aborted by the user.')
                    self.uploader.kill()
                    break

        # Close the pbar
        pbar.close()

        # Report the number of the files uploaded successfully
        if Parser.args.nowait:
            expected = [UploadPipelineFileStatus.UPLOADED, UploadPipelineFileStatus.SUCCESSFUL]
            count = sum([file.status in expected for file in files])
            msg = f'{count}/{len(files)} files are uploaded.'
        else:
            expected = [UploadPipelineFileStatus.SUCCESSFUL]
            count = sum([file.status in expected for file in files])
            msg = f'{count}/{len(files)} files are uploaded and processed successfully.'
        logger.info(msg)
        print(msg)
        # Report the status of the files if some failed
        if count != len(files):
            counts = {status: 0 for status in UploadPipelineFileStatus}
            for file in files:
                counts[file.status] += 1
            for status in expected:
                counts[status] = 0  # Already reported above
            counts = {status: count for status, count in counts.items() if count > 0}
            msg = (
                'The status of the other files: '
                + ', '.join([f'{count} {status.name}' for status, count in counts.items()])
            )
            print(msg)
            logger.info(msg)
        
        # Perform clean-up of workbooks, in case anything got deleted during upload.
        if not Parser.args.nowait:
            for workbook in self.workbooks:
                clear_empty_studies_in_workbook(discovery_instance=workbook.discovery, wb=workbook)
                clear_empty_patients_from_workbook(discovery_instance=workbook.discovery, wb=workbook)

    @_catch_errors
    @pauser
    def reprocess(self, acquisitions: list[Acquisition]) -> None:
        """
        Loops over a list of acquisition and reprocesses their failed processes in batches.

        Args:
            acquisitions: A list of the acquisitions that are not successfully processed.
        """

        # Set the reprocess manager
        bsz = Parser.settings['general']['reprocess_batch_size']
        processes = []
        for key, val in Parser.configs['processes'].items():
            if not val: continue
            process = DiscoveryTask[key]
            processes.append(process)
            if process.postprocess:
                processes.append(process.postprocess)
        self.reprocesser = ReprocessManager(
            n_threads=min(len(acquisitions),bsz,Parser.settings['general']['threads']),
            processes=processes,
        )
        self.reprocesser.launch()

        # Put the files in the queue
        batches = batched(acquisitions, bsz)
        logger.info(f'Reprocessing the acquisitions in progress...')
        pbar = tqdm.tqdm(
            total=(len(acquisitions) // bsz) + (1 if (len(acquisitions) % bsz) else 0),
            desc=f'Processing the batches'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=False,
        )
        for idx, batch in enumerate(batches):
            # Create pbar for this batch
            pbar_batch = tqdm.tqdm(
                total=len(batch),
                desc=f'└── Reprocessing the acquisitions in the batch'.ljust(Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=False,
            )
            # Send the acquisition in the batch to the reprocesser queues
            for acquisition in batch:
                self.reprocesser.add(acquisition=acquisition, pbar=pbar_batch)

            # Wait until the queue is empty
            self.reprocesser.q.join()
            pbar_batch.close()
            pbar.update()

            # Pause if signaled
            if self.pauser.paused:
                # Hide the progress bars
                pbar.clear(nolock=True)

                # Print messages and log
                msg = 'Reprocessing is paused'
                logger.info(msg)
                print(msg)

                # Ask the user if they want to resume
                if self.pauser.pause():
                    msg = 'Reprocesses are resumed.'
                    print(msg)
                    logger.info(msg)
                    # Unhide the progress bars
                    pbar.refresh(nolock=True)
                else:
                    print('Stopping the process in progress..')
                    logger.info('The process is aborted by the user.')
                    self.reprocesser.kill()
                    break

        # Close the pbar
        pbar.close()

        # Report the number of the files that are reprocessed successfully
        count = sum([(acquisition.issuccessful if acquisition.status else False) for acquisition in acquisitions])
        msg = f'{count}/{len(acquisitions)} unsuccessful acquisitions are reprocessed successfully.'
        logger.info(msg)
        print(msg)
