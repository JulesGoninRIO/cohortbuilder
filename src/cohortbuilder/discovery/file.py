"""
This module includes the objects that are related to files in Discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar
from loguru import logger
from collections import defaultdict

from src.cohortbuilder.discovery.definitions import (DiscoveryDatasetStatus,
                                       DiscoveryFileStatus,
                                       DiscoveryTaskStatus,
                                       DiscoveryJobStatus,
                                       DiscoveryTask,
                                       DiscoveryProcessor)
from src.cohortbuilder.discovery.discovery import Discovery
from src.cohortbuilder.discovery.queries import (Q_ACQUISITION_JOBS,
                                   Q_FILE_ACQUISITIONS, Q_FILE_INFO, Q_ACQUISITION_STARTJOB, Q_PROCESSOR)
from src.cohortbuilder.discovery.exceptions import UnauthorizedCallException
from src.cohortbuilder.utils.helpers import list2str

if TYPE_CHECKING:
    from src.cohortbuilder.discovery.discovery import Discovery

# Declare type variables
T = TypeVar('T')

class DiscoveryFile:
    """The class corresponding to a file entity in Discovery.

    Args:
        discovery: Discovery instance containing the file.
        uuid: UUID of the file.
        name: Name of the file. Defaults to ``None``.
        extension: extension of the file. Defaults to ``None``.
    """

    def __init__(self, discovery: Discovery, uuid: str, name: str = None, extension: str = None):
        #: The Discovery instance that contains the file
        self.discovery: Discovery = discovery
        #: The UUID of the file in Discovery
        self.uuid: str = uuid
        #: The name of the file
        self.name: str = name
        #: The extension of the file
        self.extension: str = extension
        #: The status of the file on Discovery
        self.status: DiscoveryFileStatus = None
        #: The Discovery datasets (aquisitions) created from the file
        #: and potentially, their jobs and tasks
        self.acquisitions: list[Acquisition] = None

    def fetch(self, acquisitions: bool = False) -> None:
        """Updates the file attributes if it has a UUID.

        Args:
            acquisitions: If ``True``, the acquisitions will also be fetched.
              Since it contains all the jobs and tasks, this makes the process longer.
              Defaults to ``False``.
        """

        # Skip if UUID isnot present
        if not self.uuid: return

        # Get file info
        info = self.discovery.send_query(query=(Q_FILE_INFO % self.uuid))['data']['file']
        if not info:
            logger.debug(f'{repr(self)} | File UUID ({self.uuid}) does not exist on this Discovery instance ({self.discovery.instance}).')
            return

        # Update the status
        self.status = DiscoveryFileStatus[info['status']]

        # Get the datasets and the processes
        if acquisitions:
            self.acquisitions = []
            q = Q_FILE_ACQUISITIONS % (self.uuid, 0, 0)
            edges = self.discovery.send_query(query=q)['data']['file']['datasets']['edges']
            # Get the jobs and tasks and append the acquisition
            for edge in edges:
                status = DiscoveryDatasetStatus[edge['node']['status']] if edge['node']['status'] else DiscoveryDatasetStatus.NONE
                acquisition = Acquisition(file=self, uuid=edge['node']['uuid'], status=status)
                acquisition.fetch()
                # Append the acquisition
                self.acquisitions.append(acquisition)

    @property
    def isuploaded(self) -> bool:
        """Indicates whether the file has been uploaded successfully."""

        # Return None if status is not available
        if self.status is None:
            raise UnauthorizedCallException

        return self.status.isuploaded

    @property
    def isended(self) -> bool:
        """
        Indicates whether the processing of the file has ended, regardless of its success.
        Make sure to call `fetch(processes=True) <src.discovery.entities.DiscoveryFile.fetch>`
        to update the file metadata.
        """

        # Return None if status is not available
        if self.status is None:
            raise UnauthorizedCallException

        # Check acquisitions
        if self.status.isended is None:
            # Return None if the acquisitions are not fetched
            if self.acquisitions is None:
                raise UnauthorizedCallException
            for acquisition in self.acquisitions:
                if not acquisition.isended:
                    return False
            return True
        # Return based on the status
        else:
            return self.status.isended

    @property
    def issuccessful(self) -> bool:
        """Indicates whether the processing of the file has been succssful."""

        # Return None if the file has not been fetched
        if self.status is None:
            raise UnauthorizedCallException

        # Return True if the file has been rejected
        if self.status is DiscoveryFileStatus.REJECTED:
            return True

        # Return None if the acquisitions are not fetched
        if self.acquisitions is None:
            raise UnauthorizedCallException

        # Return False if there is any unsuccessful acquisition, True otherwise
        for acqusition in self.acquisitions:
            if not acqusition.issuccessful:
                return False
        
        return True

    def __str__(self):
        return self.uuid

    def __repr__(self):
        return list2str(items=[self.discovery.instance, f'File {self.uuid}'], delimeter=' > ')

    def reprocess(self, processes: list[DiscoveryTask] = None) -> bool:
        """
        Reprocesses the file acquisitions and returns ``True``.
        If acquisitions have not been fetched, returns ``False``.
        Make sure to update the acquisitions before calling this method in order to
        avoid reprocessing the processes that already succeeded.

        Args:
            processes: If given, only a subset of all processes will be relaunched.
        """

        if self.acquisitions is None:
            logger.debug(f'{repr(self)} | File cannot be reprocessed without fetching its acquisitions.')
            return False

        for acquisition in self.acquisitions:
            acquisition.reprocess(processes)

        return True

class Acquisition:
    """
    Class for acquisitions of a file.
    It is at the same level as Dataset and has the same UUID
    but sometimes the behaviours are different (e.g., status).

    Args:
        file: Discovery file that generated this acquisitions.
        uuid: UUID of the acquisition (dataset).
        status: Status of the acqusition, if available. Defaults to ``None``.
    """

    def __init__(self, file: DiscoveryFile, uuid: str, status: DiscoveryDatasetStatus = None):
        #: The file that generated this acquisition
        self.file: DiscoveryFile = file
        #: Discovery instance
        self.discovery = self.file.discovery
        #: UUID of the acquisition (dataset)
        self.uuid: str = uuid
        #: Status of the acquisition
        self.status = status
        #: A list of the jobs of the acquisition
        self.jobs: list[dict] = None
        #: A list of the tasks of the acquisition for each processor
        self.tasks: dict[DiscoveryProcessor, list[dict]] = None
        #: The uuid of all successful tasks for each of the main processes
        self.processes: dict[DiscoveryTask, list[str]] = defaultdict(list)

    def fetch(self, status: bool = False) -> None:
        """
        Updates the status and all jobs and tasks of the dataset.
        The jobs that only have a post-processor task of a child dataset are ignored.
        This part can be tricky because there is no available documentation about the
        dependencies of Discovery jobs, processes, and statuses.
        Try not to modify this part unless you are sure.

        Args:
            status: If ``True``, the status of the acquisition will also be fetched.
              This makes the process slower.
        """

        # Get the jobs and store them
        q = Q_ACQUISITION_JOBS % (self.uuid, self.file.uuid)
        self.jobs = []
        self.tasks = defaultdict(list)
        for job in self.discovery.send_query(query=q)['data']['jobs']:
            # Build and append the job
            status = DiscoveryJobStatus[job['status']]
            tasks: dict[DiscoveryProcessor, list[dict]] = defaultdict(list)
            for task in job['tasks']:
                task_summary = {
                    'uuid': task['uuid'],
                    'status': DiscoveryTaskStatus[task['status']],
                    'input': task['io'][0]['inputDataset']['uuid'] if 'io' in task and task['io'] and task['io'][0]['inputDataset'] else None,
                    'output': task['io'][0]['outputDataset']['uuid'] if 'io' in task and task['io'] and task['io'][0]['outputDataset'] else None,
                    'iostatus': DiscoveryTaskStatus[task['io'][0]['status']] if task['io'] else None,
                }
                tasks[DiscoveryProcessor(task['processor']['name'])].append(task_summary)
                self.tasks[DiscoveryProcessor(task['processor']['name'])].append(task_summary)

            existing_processors = set([x['processor']['name'] for x in job['tasks']])

            self.jobs.append({
                'status': status,
                'tasks': tasks,
            })

        for process in DiscoveryTask:
            if process.processor.value not in existing_processors: # It does not exist for this dataset
                continue
            if process.processor is DiscoveryProcessor.POSTPROCESSOR:
                continue
            for task in self.tasks[process.processor]:
                if task['status'] is DiscoveryTaskStatus.SUCCESS:
                    if (task['iostatus'].isnegative or task['status'].isnegative) and task['iostatus'] is not DiscoveryTaskStatus.REJECTED: # The task is "valid" for this dataset (has tried to run), and has failed.
                        # Calling this is enough to instantiate the empty list with defaultdicts
                        # This then makes the success check fail, which is what we want if jobs are rejected.
                        self.processes[process]
                        self.processes[process.postprocess]
                        logger.warning(f"{process} has problem status {task['status']} (and {task['iostatus']}) for {repr(self)}.")
                    elif task['output']:
                        self.processes[process].append(task['uuid'])
                        for postprocess_task in self.tasks[DiscoveryProcessor.POSTPROCESSOR]:
                            if (
                                postprocess_task['input'] == task['output']
                                and postprocess_task['status'] in [DiscoveryTaskStatus.SUCCESS, DiscoveryTaskStatus.PENDING]
                                and postprocess_task['output']
                            ):
                                self.processes[process.postprocess].append(postprocess_task['uuid'])

        # Update the status
        if status:
            self.status = None
            q = Q_FILE_ACQUISITIONS % (self.file.uuid, 0, 0)
            edges = self.discovery.send_query(query=q)['data']['file']['datasets']['edges']
            # Look for the acquisition of this dataset
            for edge in edges:
                if edge['node']['uuid'] == self.uuid:
                    status = edge['node']['status']
                    self.status = DiscoveryDatasetStatus[status] if status else DiscoveryDatasetStatus.NONE
                    break

    @property
    def isended(self) -> bool:
        """Indicates whether the processing of the acquisition has ended, regardless of its success."""

        # Return None if status is not available
        if self.status is None:
            raise UnauthorizedCallException

        # Check jobs if status is not representative
        if self.status.isended is None:
            if self.jobs is None:
                raise UnauthorizedCallException
            for job in self.jobs:
                if job['status'].value < DiscoveryJobStatus.ENDED.value:
                    return False
            else:
                return True
        # Check status if it is representative
        else:
            return self.status.isended

    @property
    def issuccessful(self) -> bool:
        """Indicates whether the processing of the acquisition has been succssful."""

        # Return None if status is not available
        if self.status is None:
            raise UnauthorizedCallException

        # Check jobs if status is not representative
        if self.status.issuccessful is None:
            # Cannot be successful if still not ended
            if not self.isended:
                return False
            # Raise exception if the jobs are not fetched
            if self.jobs is None:
                raise UnauthorizedCallException
            # Check the processes
            return True if all(self.processes.values()) else False
        # Check status if it is representative
        else:
            return self.status.issuccessful

    def reprocess(self, processes: list[DiscoveryTask] = None) -> None:
        """
        Reprocesses the failed processes based on the latest update.
        Make sure to fetch the processes before calling this method in order to
        avoid reprocessing the processes that already succeeded.

        Args:
            processes: If given, only a subset of all processes will be relaunched.
        """

        # Return if the acquisition is not fetched
        if self.jobs is None:
            logger.debug(f'{repr(self)} | Acquisition cannot be reprocessed without fetching its processes.')
            return

        # Reprocess the failed processes
        for process, tasks in self.processes.items():
            # Skip if the process is not in the list
            if processes and process not in processes:
                continue
            # Skip if there are already successful tasks for this process
            if tasks:
                continue
            # Skip if the parent task is going to be launched (only postprocessor tasks)
            if process.processor is DiscoveryProcessor.POSTPROCESSOR and not self.processes[process.parent]:
                continue
            # Error for the main preprocessor
            if process is DiscoveryTask.PREPROCESSOR:
                logger.warning(f'{repr(self)} | Process {process} could not be relaunched.')
                continue
            # Skip for dependant processes
            if process.parent and not self.processes[process.parent]:
                logger.warning(f'{repr(self)} | Process {process} could not be relaunched because it depends on another missing process: {process.parent}.')
                continue
            # Get the input dataset
            input = None
            if process.parent:
                if not any(self.processes[process.parent]):
                    continue
                for parent_task in self.tasks[process.parent.processor]:
                    if parent_task['uuid'] in self.processes[process.parent] and parent_task['output']:
                        input = parent_task['output']
                if not input:
                    logger.warning(f'{repr(self)} | Process {process} could not be relaunched because no parent process has a valid output: {process.parent}.')
                    continue
            else:
                input = self.uuid
            # Launch the job
            processor = self.discovery.send_query(query=(Q_PROCESSOR % process.processor.value))['data']['processors']
            if not processor:
                msg = f"{process.processor.value} does not exist for {repr(self)}."
                logger.trace(msg)
                continue
            processor = processor[0]['uuid']
            r = self.discovery.send_query(query=(Q_ACQUISITION_STARTJOB % (input, processor)))
            if 'errors' in r:
                logger.warning(f'{repr(self)} | Process {process} could not be relaunched: {" ".join([e["message"] for e in r["errors"]])}.')
            else:
                try:
                    uuid = r['data']['startSimpleJob']['uuid']
                    logger.trace(f'{repr(self)} | Process {process} launched in job: {uuid}.')
                except:
                    logger.debug(f'{repr(self)} | Error while launching process {process}. Response: {r}.')

    def __str__(self):
        return self.uuid

    def __repr__(self):
        return list2str(items=[self.discovery, f'File {self.file.uuid}', f'Acquisition {self.uuid}'], delimeter=' > ')
