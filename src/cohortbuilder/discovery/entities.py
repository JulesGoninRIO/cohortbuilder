"""
This module includes the objects and entities that exist in different levels of Discovery.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import shutil
from shutil import get_terminal_size
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from time import time
from typing import TYPE_CHECKING, Callable, TypeVar, Union

from loguru import logger
from svglib.svglib import load_svg_file
import tqdm
import numpy as np

from src.cohortbuilder.discovery.definitions import (DiscoveryDatasetStatus,
                                       DiscoveryTask,
                                       LayerType,
                                       DiscoveryDatasetPurpose,
                                       LayerVariant)
from src.cohortbuilder.discovery.discovery import Discovery
from src.cohortbuilder.discovery.queries import (Q_DATASET_MOVE, Q_DATASET_REFRESH,
                                   Q_STUDY_DATASETS,
                                   Q_PATIENT_MOVE, Q_WORKBOOK_PATIENTS, Q_PATIENT_STUDIES,
                                   Q_STUDY_MOVE, Q_PROJECT_WORKBOOKS, Q_PATIENT_UNLINK, Q_PATIENT_DATASETS_SHORT)
from src.cohortbuilder.discovery.exceptions import UnauthorizedCallException
from src.cohortbuilder.discovery.file import DiscoveryFile, Acquisition
from src.cohortbuilder.managers import MultiThreader
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.taxonomy import CFIClassifier, FundusClassifier
from src.cohortbuilder.utils.helpers import enumerate_path, list2str, thumbnail2angles

if TYPE_CHECKING:
    from src.cohortbuilder.discovery.discovery import Discovery
    from src.cohortbuilder.managers import DownloadManager

# Declare type variables
T = TypeVar('T')

class Project:
    """
    The class corresponding to a project in Discovery.

    .. note::
        It should NOT inherit from the `src.discovery.entities.Entity`.

    Args:
        discovery: The Discovery instance that containst the project.
        uuid: The UUID of the project on Discovery.
        folder: The folder for downloading the content of the project. Defaults to ``None``.
        name: The name of the project. Defaults to ``None``.
    """

    def __init__(self, discovery: Discovery, uuid: str, folder: Union[str, pathlib.Path] = None, name: str = None):
        #: The Discovery instance that contains the project
        self.discovery: Discovery = discovery
        #: The UUID of the project
        self.uuid: str = uuid
        #: The given name of the project
        self.name: str = name or 'Project'

        # Instantiate the entity attributes
        #: The local folder of the project
        self.folder: pathlib.Path = folder
        #: The accessible workboos that are in the project
        self.children: list[Workbook]
        #: The path of the "downloaded" indicator file
        self.downloaded: pathlib.Path = self.folder / '.downloaded' if self.folder else None
        #: The number of the remaining workbooks
        self.rem: int = 0

        # Create the project folder
        if self.folder:
            self.folder.mkdir(exist_ok=True)

    def get_children(self) -> list[Workbook]:
        """
        Fetches the list of available workbooks and filters them.

        Returns:
            List of the pre-filtered workbooks.
        """

        # Create the children attribute
        children = []
        # Define number of children per page
        n = 15
        # Get the number of pages
        response = self.discovery.send_query(Q_PROJECT_WORKBOOKS % (self.uuid, 0, 0))['data']['workbooks']
        count = response['totalCount']
        for edge in response['edges']:
            children.append(Workbook(parent=self, attributes=edge['node']))

        return children

    @property
    def all(self) -> bool:
        """Checks if all related files of this project have been put in the download queue."""

        return all([workbook.all for workbook in self.children])

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        instance = self.discovery.instance
        project = self
        return list2str(
            items=[instance, project],
            delimeter=' > ',
        )

class Entity(ABC):
    """
    Parent class for Discovery user interface entities:
    workbook, patient, study, dataset, layer.

    Args:
        parent: Parent entity.
        attributes: Attributes fetched from Discovery.
    """

    def __init__(self, parent: Entity, attributes: dict):
        #: The Discovery instance that contains the entity
        self.discovery: Discovery = parent.discovery if parent else None
        #: The parent entity (one level higher)
        self.parent: Entity = parent
        #: The attributes of the entity that are fetched from Discovery
        self.attributes: dict = attributes
        #: The UUID of the project
        self.uuid: str = self.attributes['uuid']
        #: The path of the folder for downloading the entity
        self.folder: pathlib.Path = parent.folder / str(self) if (parent and parent.folder) else None
        #: The list of the children of the entity
        self.children: list[Entity] = None
        #: The path of the "downloaded" indicator file
        self.downloaded: pathlib.Path = None
        #: The progress bar to update when downloaded
        self.pbar: tqdm.tqdm = None
        #: The number of the remaining children of the entity
        self.rem: int = 0
        #: Flag for indicating that all the files are in the download queue
        self.all: bool = False
        #: Flag for indicating wether to show a progress bar for children of the entity
        self.create_pbar: bool = True

    @abstractmethod
    def pre_filter(self, configs: dict) -> bool:
        """Abstract method for checking whether the entity passes the preliminary filters."""
        ...

    @abstractmethod
    def post_filter(self, configs: dict) -> bool:
        """Abstract method for checking whether the entity passes the posterior filters."""
        ...

    def download(self, downloader: DownloadManager, pbar: Union[None, tqdm.tqdm] = None) -> None:
        """
        Calls the download method of the children of the entity.

        Args:
            downloader: The download manager.
            pbar: The progress bar object. If ``None``, The progress bar won't be shown. Defaults to ``None`` .
        """

        # Store pbar
        self.pbar = pbar

    	# Skip if downloaded
        if self.downloaded.exists():
            logger.info(f'{repr(self)} | Download skipped. All the content is already downloaded.')
            self.all = True
            self.parent.rem += 1
            self.rem = 0
            self.update()
            return

        # Loop over children
        n_total = len(self.children)
        pbar = tqdm.tqdm(
                total=n_total,
                desc=f'{repr(self)}'.ljust(
                    Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=None,
            ) if self.create_pbar else None
        for child in self.children:
            child.download(downloader=downloader, pbar=pbar)

        # Flag all files in queue
        self.all = True
        self.parent.rem += 1
        self.update()

    def copy(self, *args, **kwargs) -> None:
        """Calls the copy method of the children of the entity."""

        for child in self.children:
            child.copy(*args, **kwargs)

    # TODO: Keep a cache file instead of .downloaded (?)
    def update(self) -> None:
        """
        Marks an entity as being downloaded successfully by:

        - Removing the downloaded indicators of the children and creating
          the indicator for the entity;
        - Updating the progress bar of the entity, if any;
        - Updating the number of the remaining children of the parent,
          if it's an entity.
        """

        # Return if downloading the entity is still not finished
        if not self.all or self.rem:
            return

        # NOTE: For patients that are filtered out, no folder is created
        if self.folder.exists():
            # Flag as downloaded by creating a file
            f = open(self.downloaded, 'w')
            f.close()

            # Remove downloaded flags of the children
            for child in self.children:
                # NOTE: Layer and childLayer entities don't have this file
                if child.downloaded:
                    child.downloaded.unlink(missing_ok=True)

        # Update the progress bar if it exists
        if self.pbar:
            self.pbar.update(1)
            if self.pbar.n == self.pbar.total:
                self.pbar.close()

        # Update the parent's remaining children if it exists
        if self.parent:
            self.parent.rem -= 1
            if isinstance(self.parent, Entity):
                self.parent.update()

    def log_filtered_out(self, reason: str) -> None:
        """Logs the reason why the entity has been filtered out."""

        logger.trace(f'{repr(self)} | {self.__class__.__name__} filtered out: {reason}.')

    @staticmethod
    def _catch_errors_in_get_children(func: Callable[[], T]) -> Callable[[], T]:
        """
        Decorator for catching the errors in
        `self.get_children <src.discovery.entities.Entity.get_children>`.

        Args:
            func: The ``self.get_children`` function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                msg = f'{repr(self)} | Error while trying to get children. Entity will be skipped.'
                logger.warning(msg)
                logger.exception(f'{msg}: {e}')
                return list()

        return wrapper

    def __repr__(self) -> str:
        if self.parent:
            return list2str(items=[repr(self.parent), self], delimeter=' > ')
        else:
            return str(self)

class Workbook(Entity):
    """
    The class corresponding to the workbook entity of Discovery.
    You can think of workbooks as folders containing data.

    Args:
        parent: Parent project that contains this workbook.
        attributes: Attributes fetched from Discovery.
    """

    def __init__(self, parent: Project, attributes: dict):
        super().__init__(parent=parent, attributes=attributes)
        #: The project that contains this workbook
        self.parent: Project
        #: List of the patients of the workbook
        self.children: list[Patient]
        #: List of datasets in the workbook
        self.datasets: list[Dataset] = []
        #: List of the files that have a dataset in the workbook
        self.files: list[DiscoveryFile] = []
        self.downloaded: pathlib.Path = self.folder / '.downloaded' if self.folder else None

    @Entity._catch_errors_in_get_children
    def get_children(self) -> list[Patient]:
        """
        Fetches the list of the children of the workbook from Discovery
        and returns it.

        Returns:
            List of the children.
        """

        # Create the children attribute
        children = list()
        # Define number of children per page
        BATCH_SIZE = 250
        # Get the number of pages
        count = self.discovery.send_query(query=(Q_WORKBOOK_PATIENTS % (self.uuid, 1, 0, self.uuid)))['data']['patientSearchFeed']['totalCount']
        NUM_BATCHES = int(np.ceil(count / BATCH_SIZE))
        for b in range(NUM_BATCHES):
            query = Q_WORKBOOK_PATIENTS % (self.uuid, BATCH_SIZE, b * BATCH_SIZE, self.uuid)
            response = self.discovery.send_query(query)
            for edge in response['data']['patientSearchFeed']['edges']:
                children.append(Patient(self, edge['node']))

        return children

    # TODO: Use multi-threading
    def get_files(self, pbar: tqdm.tqdm = None) -> list[DiscoveryFile]:
        """
        Returns a list of the Discovery files that have at least one dataset in this workbook.
        If there are multiple acquisitions of a single file in the workbook, the file will
        be repeated in the list.
        """


        # Get the patients
        patients = self.get_children()

        # Update the pbar
        if pbar is not None:
            pbar.total = len(patients)
            pbar.refresh()

        # Get the files of each patient
        files: list[DiscoveryFile] = []
        for patient in patients:
            acquisitions = patient.get_acquisitions(fetch=False)
            files.extend([acquisition.file for acquisition in acquisitions])
            if pbar: pbar.update()
        if pbar: pbar.close()

        return files

    def get_acquisitions(self, separate: bool = False, verbose: bool = False) -> dict[str, list[Acquisition]]:
        """
        Lists the acquisitions in the workbook and optionally filters them.

        Args:
            separate: If ``True``, the acquisitions will fetched and separated
              based on their status. Defaults to ``False``.
            verbose: If ``True``, the details will be logged.

        Returns:
            A dictionary with the fields: "all", "pending", "failed", and "successful".
        """

        # Instantiate the output
        acquisitions: dict[str, list[Acquisition]] = {
            'all': [],
            'pending': [] if separate else None,
            'failed': [] if separate else None,
            'successful': [] if separate else None,
        }

        # Get the patients
        if not self.children:
            if verbose:
                msg = f'{repr(self)} | Getting the patients..'
                logger.info(msg)
                print(msg)
            self.children = self.get_children()
        patients = self.children

        # Divide the threads
        n_threads = Parser.settings['general']['threads']
        if len(patients) < n_threads:
            n_threads_out = len(patients)
            n_threads_in = n_threads // n_threads_out
        else:
            n_threads_out = n_threads
            n_threads_in = None

        # Instantiate the progress bar
        pbar = tqdm.tqdm(
            total=len(patients),
            desc=f'Getting the patient acquisitions'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=None,
        ) if verbose else None

        # Define the process on each patient
        def get_patient_acquisitions(patient: Patient) -> None:
            # Get the patient acquisitions and append them to the list
            patient_acquisitions = patient.get_acquisitions(
                fetch=(True if separate else False),
                n_threads=n_threads_in,
                verbose=verbose,
            )
            acquisitions['all'].extend(patient_acquisitions)
            # Skip the separation
            if not separate:
                return
            # Separate them based on their status
            for acquisition in patient_acquisitions:
                if not acquisition.isended:
                    if verbose: logger.trace(f'{repr(acquisition)} | Pending acquisition detected.')
                    acquisitions['pending'].append(acquisition)
                    continue
                if not acquisition.issuccessful:
                    if verbose: logger.trace(f'{repr(acquisition)} | Unsuccessful acquisition detected.')
                    acquisitions['failed'].append(acquisition)
                    continue
                acquisitions['successful'].append(acquisition)
            # Update the progress bar
            if pbar: pbar.update()

        # Process the patients in multiple threads
        with MultiThreader(
            n_threads=n_threads_out,
            process=get_patient_acquisitions,
            items=patients,
            verbose=verbose,
            # NOTE: The outer threads are paused when
            # their inner threads are sending requests
            limited=False,
        ) as multithreader:
            multithreader.execute()

        # Close the progress bar
        if pbar: pbar.close()

        return acquisitions

    # UNUSED
    def pre_filter(self, configs: dict) -> bool:
        """Checks whether the workbook passes the preliminary filters."""

        return True

    # UNUSED
    def post_filter(self, configs: dict) -> bool:
        """Checks whether the workbook passes the posterior filters."""

        return True

    def download(self, downloader: DownloadManager, pbar: Union[None, tqdm.tqdm] = None) -> None:
        """
        Filters and downloads the content of a workbook.

        Args:
            downloader: Download manager.
            pbar: Progress bar to be updated once the workbook is downloaded. Defaults to ``None`` .

        Raises:
            AllThreadsDeadException: When all the threads of the downloader are dead.
        """

        # Store the pbar
        self.pbar = pbar

        # Fetch the children
        # NOTE: Multi-threading is safe here because the download queue is empty
        self.children = self.get_children()

        # Skip if downloaded
        if self.downloaded.exists():
            logger.info(f'{repr(self)} | Download skipped. All the content is already downloaded.')
            self.all = True
            self.parent.rem += 1
            self.rem = 0
            self.update()
            return

        # Pre-filter the patients
        patients = [patient for patient in self.children if patient.pre_filter(configs=downloader.configs)]
        logger.info(f'{repr(self)} | Pre-filtered patients {len(patients)}/{len(self.children)}: {list2str(patients)}')

        # Loop over the pre-filtered patients
        patients_filtered = list()
        n_total = len(patients)
        pbar = tqdm.tqdm(
                total=n_total,
                desc=f'{repr(self)}'.ljust(
                    Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=None,
            ) if self.create_pbar else None
        for i, patient in enumerate(patients):
            # Filter the patient and its children
            patient.filter(configs=downloader.configs)
            patient_is_filtered = patient.post_filter(configs=downloader.configs)

            if patient_is_filtered:
                # Add it to the filtered patients
                patients_filtered.append(patient)
                # Download the patient
                patient.download(downloader=downloader, pbar=pbar)

            # Flag all files in queue
            # NOTE: Doing this here to allow for q.join()
            if (i+1) == n_total:
                self.all = True
                self.parent.rem += 1

            if not patient_is_filtered:
                # Update the patient as if it was downloaded
                patient.pbar = pbar
                patient.all = True
                patient.parent.rem += 1
                patient.rem = 0
                patient.update()

            # Finish the downloads for this patient
            if downloader.isalive:
                logger.trace(f'{repr(patient)} | Joining the download queue.')
                downloader.q.join()
                logger.trace(f'{repr(patient)} | Download queue joined.')
            else:
                downloader.dead()
            self.update()

        # Store the post-filtered patients
        logger.info(f'{repr(self)} | Post-filtered patients {len(patients)}/{len(self.children)}: {list2str(patients)}')
        self.children = patients_filtered

    # UNUSED
    def save(self) -> None:
        """Saves the workbook object as a pickle file."""

        with open(self.file, 'wb') as f:
            pickle.dump(obj=self, file=f)
        logger.info(f'{repr(self)} | Workbook saved to {self.file}.')

    # UNUSED
    def load(self) -> None:
        """Loads the workbook object from the pickle file."""

        # Check existence of the workbook file
        if not self.file.exists():
            raise FileNotFoundError

        # Load the workbook from its file
        with open(self.file, 'rb') as f:
            wb = pickle.load(file=f)
        self.__dict__.update(wb.__dict__)
        logger.info(f'{repr(self)} | Workbook loaded from {self.file}.')

        # Reset all and rem
        self.all = False
        self.rem = 0
        for patient in self.children:
            patient.all = False
            patient.rem = 0
            for study in patient.children:
                study.all = False
                study.rem = 0
                for dataset in study.children:
                    dataset.all = False
                    dataset.rem = 0

    def clear(self) -> None:
        """Removes all the patients of the workbook on Discovery."""

        _ = self.discovery.send_query(
            query=Q_PATIENT_UNLINK,
            variables={
                'input': {
                    'workbookUuid': self.uuid,
                    'patientUuids': [patient.uuid for patient in self.get_children()],
                }
            },
        )

    @property
    def datasetcount(self) -> int:
            """Returns the number of datasets."""

            return len(self.get_acquisitions(verbose=False)['all'])

    def __str__(self) -> str:
        return self.attributes['name']

class Patient(Entity):
    """
    The class corresponding to the patient entity of Discovery.

    Args:
        parent: Parent workbook.
        attributes: Attributes fetched from Discovery.
    """

    def __init__(self, parent: Workbook, attributes: dict):
        super().__init__(parent=parent, attributes=attributes)
        #: Parent workbook of the patient
        self.parent: Workbook
        #: List of the studies of the patient
        self.children: list[Study]
        self.downloaded: pathlib.Path = self.folder / '.downloaded' if self.folder else None

    @Entity._catch_errors_in_get_children
    def get_children(self) -> list[Study]:
        """
        Fetches the list of the children of the patient from Discovery
        and returns it.

        Returns:
            List of the children.
        """

        # Create the children attribute
        children = []
        response = self.discovery.send_query(Q_PATIENT_STUDIES % (self.uuid, self.parent.uuid, 0, 0, self.parent.uuid))
        for edge in response['data']['patient']['studies']['edges']:
            children.append(Study(parent=self, attributes=edge['node']))

        return children

    def get_acquisitions(self, fetch: bool = False, n_threads: int = None, verbose: bool = True) -> list[Acquisition]:
        """
        Returns a list of the acquisition datasets of the patient
        directly without looping over the studies.

        Args:
            fetch: If ``True``, the acquisitions will be fetched. Defaults to ``False``.
            n_threads: If passed, the datasets will be processed in multiple threads. Defaults to ``None``.
            verbose: If ``True``, details will be logged. Defaults to ``True``.
        """

        # Instantiate the acquisitions list
        acquisitions = []

        # Define the process for each dataset
        def append_acquisition(edge):
            dataset = edge['node']
            purpose = DiscoveryDatasetPurpose[dataset['purpose']]
            status = DiscoveryDatasetStatus[dataset['status']]
            if purpose is not DiscoveryDatasetPurpose.IMPORT:
                return
            if dataset['parentFile'] is None:
                logger.debug(f'Imported dataset ({dataset["uuid"]}) does not have a parentfile!')
            dfile = DiscoveryFile(
                discovery=self.discovery,
                uuid=dataset['parentFile']['uuid'],
                name=dataset['parentFile']['filename'],
                extension=dataset['parentFile']['extension'],
            )
            acquisition = Acquisition(file=dfile, uuid=dataset['uuid'], status=status)
            if fetch: acquisition.fetch()
            acquisitions.append(acquisition)

        query = Q_PATIENT_DATASETS_SHORT % (self.uuid, self.parent.uuid, 0, 0, self.parent.uuid)
        response = self.discovery.send_query(query)

        # Process the datasets in multiple threads
        edges = response['data']['patient']['datasets']['edges']
        if n_threads:
            with MultiThreader(
                n_threads=n_threads,
                process=append_acquisition,
                items=edges,
                verbose=verbose,
            ) as multithreader:
                multithreader.execute()
        else:
            for edge in edges:
                append_acquisition(edge)

        return acquisitions

    def pre_filter(self, configs: dict) -> bool:
        """Checks whether the patient passes the preliminary filters."""

        # Check patient name
        if configs['filters']['patients']['ids']:
            if not int(self.attributes['patientId'].strip()) in configs['filters']['patients']['ids']:
                self.log_filtered_out('Patient ID not wanted')
                return False

        # Check birthday range
        if not self.attributes['birthdate']:
            # False if patient has no registered birthdate and a constraint is set
            if configs['filters']['patients']['birthdate_inf'] or configs['filters']['patients']['birthdate_sup']:
                self.log_filtered_out('Birthdate not available')
                return False
        else:
            # False if birthday range is violated
            birthdate = datetime.strptime(self.attributes['birthdate'], '%Y-%m-%d')
            if any([
                configs['filters']['patients']['birthdate_inf'] and birthdate < datetime.strptime(configs['filters']['patients']['birthdate_inf'], '%Y-%m-%d'),
                configs['filters']['patients']['birthdate_sup'] and birthdate > datetime.strptime(configs['filters']['patients']['birthdate_sup'], '%Y-%m-%d'),
            ]):
                self.log_filtered_out(f'Birthday ({birthdate}) not in the range')
                return False

        # Check sex
        if not self.attributes['sex']:
            self.log_filtered_out('Sex not available')
            return False
        elif not self.attributes['sex'] in configs['filters']['patients']['sex']:
            self.log_filtered_out(f'Sex ({self.attributes["sex"]}) not wanted')
            return False

        return True

    def post_filter(self, configs: dict) -> bool:
        """Checks whether the patient passes the posterior filters."""

        # False if no study in this patient
        if not self.children:
            self.log_filtered_out('No remaining children')
            return False

        return True

    def filter(self, configs: dict) -> None:
        """
        Applies the filters on the entities in the patient.
        Extends layers and datasets. Creates a folder for the patient and stores it.
        """

        # Fetch the children
        self.children = self.get_children()

        # Pre-filter studies
        studies = [study for study in self.children if study.pre_filter(configs=configs)]
        logger.info(f'{repr(self)} | Pre-filtered studies {len(studies)}/{len(self.children)}: {list2str(studies)}')

        for study in studies:
            # Fetch the children
            study.children = study.get_children()

            # Pre-filter datasets
            datasets = [dataset for dataset in study.children if dataset.pre_filter(configs=configs)]
            logger.info(f'{repr(study)} | Pre-filtered datasets {len(datasets)}/{len(study.children)}: {list2str(datasets)}')

            for dataset in datasets:
                # Pre-filter layers and child layers
                layers = [layer for layer in dataset.children if layer.pre_filter(configs=configs)]
                logger.trace(f'{repr(dataset)} | Pre-filtered layers {len(layers)}/{len(dataset.children)}: {list2str(layers)}')

                # Extend the layers and the dataset
                # NOTE: Taxonomy will also be done for the layers that have been filtered out
                # NOTE: This is necessary if we want the layer variants of the dataset to be comprehensive
                for layer in dataset.children:
                    layer.extend()
                dataset.extend()

                # Post-filter the layers and the child layers
                layers = [layer for layer in layers if layer.post_filter(configs=configs)]
                logger.trace(f'{repr(dataset)} | Post-filtered layers {len(layers)}/{len(dataset.children)}: {list2str(layers)}')
                dataset.children = layers

            # Post-filter the datasets
            datasets = [dataset for dataset in datasets if dataset.post_filter(configs=configs)]
            logger.info(f'{repr(study)} | Post-filtered datasets {len(datasets)}/{len(study.children)}: {list2str(datasets)}')
            study.children = datasets

        # Post-filter the studies
        studies = [study for study in studies if study.post_filter(configs=configs)]
        logger.info(f'{repr(self)} | Post-filtered studies {len(studies)}/{len(self.children)}: {list2str(studies)}')
        self.children = studies

    def copy(self, target: str) -> None:
        """
        Copies all the studies of this patient from its parent workbook to another workbook
        on the same Discovery instance.

        Args:
            target: The UUID of the target workbook.
        """

        # Create the query and send it
        query = Q_PATIENT_MOVE % (self.uuid, self.parent.uuid, target)
        response = self.discovery.send_query(query)

        # Catch possible errors
        if 'errors' in response:
            logger.warning(f'{repr(self)} | Error in uploading: {response["errors"]}')

    @property
    def datasetcount(self) -> int:
            """Returns the number of datasets."""

            return len(self.get_acquisitions(
                n_threads=Parser.settings['general']['threads'],
                verbose=False,
            ))

    def __str__(self) -> str:
        pid = self.attributes['patientId']
        if pid.strip().isdigit():
            return f'{int(pid):08d} {self.uuid.split("-")[0]}'
        else:
            return f'{pid[:8]} {self.uuid.split("-")[0]}'

class Study(Entity):
    """
    The class corresponding to the study entity of Discovery.

    Args:
        parent: Parent patient.
        attributes: Attributes fetched from Discovery.
    """

    def __init__(self, parent: Patient, attributes: dict):
        super().__init__(parent=parent, attributes=attributes)
        #: The parent patient of the study
        self.parent: Patient
        #: List of the datasets of the study
        self.children: list[Dataset]
        self.downloaded: pathlib.Path = self.folder / '.downloaded' if self.folder else None
        self.create_pbar: bool = False

    @Entity._catch_errors_in_get_children
    def get_children(self) -> list[Dataset]:
        """
        Fetches the list of the children of the study from Discovery
        and returns it.

        Returns:
            List of the children.
        """

        # Create the children attribute
        children = []
        # Define number of children per page
        n = 10
        # Get the workbook uuid
        wb_uuid = self.parent.parent.uuid
        def add_datasets_by_type(dataset_type):
            query = Q_STUDY_DATASETS % (self.uuid, wb_uuid, dataset_type, 0, 0, wb_uuid, wb_uuid)
            response = self.discovery.send_query(query)
            for edge in response['data']['study']['datasets']['edges']:
                children.append(Dataset(parent=self, attributes=edge['node']))
        # Get acquisition and ecrf children
        add_datasets_by_type('ACQUISITION')
        add_datasets_by_type('ECRF')

        return children

    def pre_filter(self, configs: dict) -> bool:
        """Checks whether the study passes the preliminary filters."""

        # False if date is out of the allowed range
        studyDate = datetime.strptime(self.attributes['studyDatetime'], '%Y-%m-%dT%H:%M:%S.%fZ')
        if any([
            configs['filters']['studies']['date_inf'] and studyDate < datetime.strptime(configs['filters']['studies']['date_inf'], '%Y-%m-%d'),
            configs['filters']['studies']['date_sup'] and studyDate > datetime.strptime(configs['filters']['studies']['date_sup'], '%Y-%m-%d'),
        ]):
            self.log_filtered_out(f'Date ({studyDate}) out of range')
            return False

        # False if patient age at the time of study is out of the allowed range
        if configs['filters']['studies']['patient_age_inf'] or configs['filters']['studies']['patient_age_sup']:
            if not self.parent.attributes['birthdate']:
                self.log_filtered_out(f'Patient birthdate not available')
                return False
            # CHECK: The timedelta class does not have .year attribute
            patient_age = studyDate - datetime.strptime(self.parent.attributes['birthdate'], '%Y-%m-%d')
            if any([
                configs['filters']['studies']['patient_age_inf'] and patient_age.year < configs['filters']['studies']['patient_age_inf'],
                configs['filters']['studies']['patient_age_sup'] and patient_age.year > configs['filters']['studies']['patient_age_sup'],
            ]):
                self.log_filtered_out(f'Patient age at the time of the study ({patient_age.year}) out of range')
                return False

        # False if uuid is not in the list
        if (configs['filters']['studies']['uuids']) and (not self.uuid in configs['filters']['studies']['uuids']):
            self.log_filtered_out('UUID not wanted')
            return False

        return True

    def post_filter(self, configs: dict) -> bool:
        """Checks whether the study passes the posterior filters."""

        # False if no dataset in the study
        if not self.children:
            self.log_filtered_out('No remaining children')

            return False

        # False if none of the wanted combinations is available in the study
        if configs['filters']['studies']['variants']:
            layer_variants_list = [dataset.info['layerVariants'] for dataset in self.children]
            images_allowed = [
                {variant.strip() for variant in comb.split('+')}
                    for comb in configs['filters']['studies']['variants']
            ]
            for image in images_allowed:
                if not any(image.issubset(dataset_layer_variants) for dataset_layer_variants in layer_variants_list):
                    self.log_filtered_out('None of the variant combinations present')

                    return False

        return True

    def copy(self, target: str) -> None:
        """
        Copies all the datasets of this study from its parent workbook to another workbook
        on the same Discovery instance.

        Args:
            target: The UUID of the target workbook.
        """

        # Create the query and send it
        query = Q_STUDY_MOVE % (self.uuid, self.parent.parent.uuid, target)
        response = self.discovery.send_query(query)

        # Catch possible errors
        if 'errors' in response:
            logger.warning(f'{repr(self)} | Error in uploading: {response["errors"]}')

    def __str__(self) -> str:
        date = self.attributes['studyDatetime'][0:10]
        return f'{date} {self.uuid.split("-")[0]}'

class Dataset(Entity):
    """
    The class corresponding to the dataset entity of Discovery (items in a study).
    Note that each dataset might include multiple image modalities (OCT, CFI, etc.),
    or have multiple images of an image modality.

    Args:
        parent: Parent study.
        attributes: Attributes fetched from Discovery.
    """

    def __init__(self, parent: Study, attributes: dict):
        super().__init__(parent=parent, attributes=attributes)
        #: The parent study of the dataset
        self.parent: Study
        #: List of layers and child layers of the dataset
        self.children: list[Union[Layer, ChildLayer]] = self.get_children()
        self.downloaded: pathlib.Path = self.folder / '.downloaded' if self.folder else None
        #: Flag for indicating that the urls of the dataset are being refreshed
        self.refreshinprogress: bool = False
        #: Time of the last refresh (in seconds)
        self.lastrefreshed: float = time()
        #: A map between the UUID and the URLs of the content of the dataset
        self.urls: dict[str, str] = self.build_urls(attributes=self.attributes)
        #: Acquisition (same level as Dataset) object for processing-related functionalities
        self.acquisition: Acquisition = Acquisition(
            file=DiscoveryFile(discovery=self.discovery, uuid=self.attributes['parentFile']['uuid']),
            uuid=self.uuid,
            status=DiscoveryDatasetStatus[self.attributes['status']]
        ) if self.attributes['parentFile'] else None
        self.create_pbar = False

        # Store parents
        study = self.parent
        patient = study.parent
        workbook = patient.parent
        #: Summary of related information of the dataset
        self.info: dict = {
            'uuid': self.uuid,
            'parentFile': {
                'uuid': self.attributes['parentFile']['uuid'],
                'filename': self.attributes['parentFile']['filename'],
            } if self.attributes['parentFile'] else None,
            'owner': {
                'firstName': self.attributes['owner']['firstName'],
                'lastName': self.attributes['owner']['lastName'],
            } if self.attributes['owner'] else None,
            'createdAt': self.attributes['createdAt'],
            'updatedAt': self.attributes['updatedAt'],
            'laterality': self.attributes['laterality'],
            'manufacturer': self.attributes['manufacturer'] if 'manufacturer' in self.attributes else None,
            'device': self.attributes['device'] if 'device' in self.attributes else None,
            'acquisitionDatetime': self.attributes['acquisitionDatetime'] if 'acquisitionDatetime' in self.attributes else None,
            'seriesDatetime': self.attributes['seriesDatetime'] if 'seriesDatetime' in self.attributes else None,
            'layerTypes': list(),
            'layerVariants': list(),
            'study': {
                'uuid': study.uuid,
                'studyId': study.attributes['studyId'],
                'studyDatetime': study.attributes['studyDatetime'],
            },
            'patient': {
                'uuid': patient.uuid,
                'patientId': patient.attributes['patientId'],
                'name': patient.attributes['name'],
                'surname': patient.attributes['surname'],
                'birthdate': patient.attributes['birthdate'],
                'sex': patient.attributes['sex'],
            },
            'workbook': {
                'uuid': workbook.uuid,
                'name': workbook.attributes['name'],
            },
            'url': Parser.settings['api'][self.discovery.instance]['url_dataset'] % (
                workbook.uuid,
                patient.uuid,
                study.uuid,
                self.uuid,
            ),
        }

    @Entity._catch_errors_in_get_children
    def get_children(self) -> list[Union[Layer, ChildLayer]]:
        """
        Creates and returns the list of the children of the dataset.
        The children attribute of a dataset contains both layers and child layers.

        Returns:
            List of the children.
        """

        # Create layers
        layers = [Layer(parent=self, attributes=layer) for layer in self.attributes['layers']]
        # Create child layers
        child_layers = list()
        if 'children' in self.attributes:
            for child in self.attributes['children']:
                child_layers.extend([ChildLayer(parent=self, attributes=layer) for layer in child['layers']])
        # Store both
        children = layers + child_layers

        return children

    def pre_filter(self, configs: dict) -> bool:
        """Checks whether the dataset passes the preliminary filters."""

        # Check the laterality (except for eCRFs)
        if (configs['filters']['datasets']['laterality']) and (not self.attributes['device'] == 'Discovery'):
            # Laterality must be available
            if not self.attributes['laterality']:
                self.log_filtered_out('Laterality not available')
                return False
            # Laterality must match
            elif not self.attributes['laterality'] in configs['filters']['datasets']['laterality']:
                self.log_filtered_out('Laterality not wanted')
                return False

        # Check the device and the manufacturer
        if any([
            # Device, if specified, must match
            configs['filters']['datasets']['device'] and ('device' not in self.attributes or
                self.attributes['device']) != configs['filters']['datasets']['device'],
            # Manufacturer, if specified, must match
            configs['filters']['datasets']['manufacturer'] and ('manufacturer' not in self.attributes or
                self.attributes['manufacturer']) != configs['filters']['datasets']['manufacturer'],
        ]):
            self.log_filtered_out(f'Device ({self.attributes["device"]}) \
                or manufacturer ({self.attributes["manufacturer"]}) not wanted')
            return False

        # False if uuid is not in the list
        if (configs['filters']['datasets']['uuids']) and (not self.uuid in configs['filters']['datasets']['uuids']):
            self.log_filtered_out('UUID not wanted')
            return False

        # False if invisible
        if not configs['filters']['datasets']['invisible']:
            for layer in [layer for layer in self.children if isinstance(layer, Layer)]:
                if (
                    layer.attributes['scanType'] in {
                        LayerType.OCT.value,
                        LayerType.FUNDUS.value,
                        LayerType.RAW.value,
                    }
                    or layer.attributes['name'] in {
                        'volume',
                        'pdf',
                        'ecrf',
                    }
                ):
                    break
            else:
                self.log_filtered_out('Invisible in Discovery')
                return False

        return True

    def post_filter(self, configs: dict) -> bool:
        """Checks whether the dataset passes the posterior filters."""

        # Check if any parent file is available and wanted
        parentfile_available_and_wanted = False
        if (
            'parentFile' in self.attributes
            and self.attributes['parentFile']
            and self.attributes['parentFile']['signedUrl']
        ):
            ext = self.attributes['parentFile']['extension']
            if any([
                (not ext or ext.lower() in {'dcm', 'zip'}) and configs['types']['dicom'],
                ext and ext.lower() == 'e2e' and configs['types']['e2e'],
                ext and ext.lower() == 'fda' and configs['types']['fda'],
            ]):
                parentfile_available_and_wanted = True
        # False if no layer in the dataset and no dataset-level file wanted
        if not self.children and not any([
                configs['types']['h5'] and 'signedUrl' in self.attributes and self.attributes['signedUrl'],
                configs['types']['thumbnail'] and 'thumbnail'in self.attributes and self.attributes['thumbnail'],
                configs['types']['ecrf'] and LayerVariant.ECRF.value in self.info['layerVariants'],
                parentfile_available_and_wanted,
        ]):
            self.log_filtered_out('No remaining children or wanted file')
            return False

        # False if no allowed combination matches layerVariants
        if configs['filters']['datasets']['variants']:
            images_allowed = [
                {variant.strip() for variant in comb.split('+')}
                    for comb in configs['filters']['datasets']['variants']
            ]
            for image in images_allowed:
                if image.issubset(self.info['layerVariants']):
                    break
            else:
                self.log_filtered_out('None of the variant combinations present')
                return False

        return True

    def copy(self, target: str) -> None:
        """
        Copies the dataset from its parent workbook to another workbook on
        the same Discovery instance.

        Args:
            target: The UUID of the target workbook.
        """

        # Create the query and send it
        query = Q_DATASET_MOVE % (target, self.uuid, self.parent.parent.parent.uuid)
        response = self.discovery.send_query(query)

        # Catch possible errors
        if 'errors' in response:
            logger.warning(f'{repr(self)} | Error in uploading: {response["errors"]}')

    def download(self, downloader: DownloadManager, pbar: Union[None, tqdm.tqdm] = None) -> None:
        """
        Downloads all the related data in the dataset.

        Args:
            downloader: The download manager.
            pbar: The progress bar object. If ``None``, The progress bar won't be shown. Defaults to ``None`` .
        """

        # Store pbar
        self.pbar = pbar

    	# Skip if downloaded
        if self.downloaded.exists():
            logger.info(f'{repr(self)} | Download skipped. All the contents are already downloaded.')
            self.parent.rem += 1
            self.all = True
            self.rem = 0
            self.update()
            return

        # Skip if the dataset is still being processed
        if self.acquisition:
            self.acquisition.fetch()
            if not self.acquisition.isended:
                if Parser.args.all:
                    logger.warning(
                        f'{repr(self)} | The dataset is still being processed. '
                        f'Check {repr(self.acquisition)} for more details.'
                    )
                else:
                    logger.warning(
                        f'{repr(self)} | Download skipped. The dataset is still being processed. '
                        f'Check {repr(self.acquisition)} for more details.'
                    )
                    return
            # Skip if the dataset is not completely processed
            if not self.acquisition.issuccessful:
                if Parser.args.all:
                    logger.warning(
                        f'{repr(self)} | Dataset is not successfully processed. '
                        f'Check {repr(self.acquisition)} for more details.'
                    )
                else:
                    logger.warning(
                        f'{repr(self)} | Download skipped. Dataset is not successfully processed. '
                        f'Check {repr(self.acquisition)} for more details.'
                    )
                    return

        # Remove the previous folder and create a new one
        if self.folder.exists():
            logger.info(f'{repr(self)} | Folder exists but downloads were not finished. Overwriting the existing files.')
            while self.folder.exists():
                try:
                    shutil.rmtree(self.folder)
                except OSError: # Complains about dir not being empty, but we can power through it and get the same result if we iterate multiple times. (The whole point of rmtree is to remove the whole tree and it's not doing it >:/ )
                    pass
        self.folder.mkdir(parents=True)

        # Download H5
        if downloader.configs['types']['h5'] and 'signedUrl' in self.attributes and self.attributes['signedUrl']:
            url_uuid = self.uuid + '-' + '0000'
            out = self.folder / 'dataset.h5'
            downloader.add(dataset=self, url_uuid=url_uuid, out=out)

        # Download thumbnail
        if downloader.configs['types']['thumbnail'] and 'thumbnail'in self.attributes and self.attributes['thumbnail']:
            url_uuid = self.uuid + '-' + '0001'
            out = self.folder / 'thumbnail.svg'
            downloader.add(dataset=self, url_uuid=url_uuid, out=out)

        # Download parent file
        if (
            'parentFile' in self.attributes
            and self.attributes['parentFile']
            and self.attributes['parentFile']['signedUrl']
        ):
            ext = self.attributes['parentFile']['extension']
            url_uuid = self.uuid + '-' + '0002'
            # Dicom (NOTE: No extension means Dicom file)
            if (
                downloader.configs['types']['dicom']
                and (not ext or ext.lower() in {'dcm', 'zip'})
            ):
                out = self.folder / 'parent.dcm'
                downloader.add(dataset=self, url_uuid=url_uuid, out=out)
            # e2e
            elif downloader.configs['types']['e2e'] and (ext and ext.lower() == 'e2e'):
                out = self.folder / 'parent.e2e'
                downloader.add(dataset=self, url_uuid=url_uuid, out=out)
            elif downloader.configs['types']['fda'] and (ext and ext.lower() == 'fda'):
                out = self.folder / 'parent.fda'
                downloader.add(dataset=self, url_uuid=url_uuid, out=out)

        # Store eCRFs
        if downloader.configs['types']['ecrf'] and any([
            LayerVariant.FORM.name in self.info['layerVariants'],
            LayerVariant.ECRF.name in self.info['layerVariants'],
        ]):
            folder = self.folder / 'ecrf'
            folder.mkdir(exist_ok=True)
            out = enumerate_path(folder / 'ecrf.json')
            with open(out, 'w') as f:
                json.dump(
                    obj=self.attributes['layers'],
                    fp=f,
                    indent=4,
                )

        # Store dataset information
        with open(self.folder / 'info.json', 'w') as f:
            json.dump(
                obj=self.info,
                fp=f,
                indent=4,
            )

        # Download layers
        for layer in self.children:
            layer.download(downloader=downloader)

        # Flag all files in queue
        self.all = True
        self.parent.rem += 1
        self.update()

    def upload(self, target: str) -> None:
        """
        Copies the dataset from its parent workbook to another workbook
        on the same Discovery instance.

        Args:
            target: The UUID of the target workbook.
        """

        self.copy(target=target)

    def extend(self) -> None:
        """Fetches the types and variants of the layers in the dataset."""

        # Add layer types and layer variants
        types = set()
        variants = set()
        for layer in self.children:
            if isinstance(layer, Layer):
                types.add(LayerType(layer.attributes['scanType']).name)
                for variant in layer.attributes['scanVariant']:
                    variants.add(LayerVariant(variant).name)
        self.info['layerTypes'] = sorted(list(types))
        self.info['layerVariants'] = sorted(list(variants))

        # Add angles for OCT_LINE and OCT_CUBE
        angles = None
        if 'thumbnail'in self.attributes and self.attributes['thumbnail'] and any([
            'OCT_LINE' in self.info['layerVariants'],
            'OCT_STAR' in self.info['layerVariants'],
        ]):
            # Download the thumbnail SVG
            url_uuid = self.uuid + '-' + '0001'
            url = self.get_url(uuid=url_uuid)
            try:
                svg = self.discovery.download(url)
            except Exception as e:
                msg = f'{repr(self)} | Failed to load thumbnail for getting line angle.'
                logger.warning(msg)
                logger.exception(f'{msg}: {e}')
                svg = None

            # Get the angles
            if svg:
                thumbnail = load_svg_file(svg)
                angles = thumbnail2angles(thumbnail)
        # Store the angles in the dataset info
        self.info['angles'] = angles

    def get_url(self, uuid: str, force: bool = False) -> Union[str, None]:
        """
        Refreshes the dataset urls and fetches an URL by its UUID.

        Args:
            uuid: The UUID of the URL.
            force: If ``True``, the URLs get refreshed even if they've been refreshed recently.

        Returns:
            The refreshed URL.
        """

        # Refresh if necessary
        if self.refreshinprogress:
            while True:
                recently = (time() - self.lastrefreshed) < Parser.settings['api'][self.discovery.instance]['timeout']
                if recently:
                    break
        else:
            self.refresh_urls(force=force)

        # Get the URL
        if uuid in self.urls:
            url = self.urls[uuid]
        else:
            url = None

        return url

    def refresh_urls(self, force: bool = False) -> None:
        """
        Refreshes the URLs of the dataset from its UUID and the UUID of the
        workbook, if they are expired or close to getting expired.

        Args:
            force: If ``True``, the URLs get refreshed even if they've been
              refreshed recently.
        """

        # Check the last time the URLs got refreshed
        recently = (time() - self.lastrefreshed) < Parser.settings['api'][self.discovery.instance]['timeout']
        if recently and not force:
            return

        # Mark refresh in progress
        self.refreshinprogress = True

        # Create the query and send it
        query = Q_DATASET_REFRESH % (
            self.uuid,
            self.parent.parent.parent.uuid,
            self.parent.parent.parent.uuid,
        )
        result = self.discovery.send_query(query)

        # Update the URLs
        self.urls = self.build_urls(attributes=result['data']['dataset'])
        logger.trace(f'{repr(self)} | URLs got refreshed.')

        # Mark refresh done
        self.lastrefreshed = time()
        self.refreshinprogress = False

    def build_urls(self, attributes: dict) -> dict[str, str]:
        """
        Generates a map of URLs in the dataset with their corresponding UUIDs.

        For dataset level URLs, a 4-digits index is appended to the tail of the dataset UUID.
        The indices are as following:

        - ``signedUrl`` (h5): ``0000``
        - ``thumbnail``: ``0001``
        - ``parentFile``: ``0002``

        For each layer and child layer URL, a 4-digits index (in the order of the content)
        is appended to the tail of the layer UUID.

        Args:
            attributes: Dataset attributes.

        Returns:
            Dataset URLs.
        """

        # Instantiate
        urls = dict()

        # Add dataset related urls
        if 'signedUrl' in attributes and attributes['signedUrl']:
            urls[self.uuid + '-' + '0000'] = attributes['signedUrl']
        if 'thumbnail'in attributes and attributes['thumbnail']:
            urls[self.uuid + '-' + '0001'] = attributes['thumbnail']
        if 'parentFile' in attributes and attributes['parentFile']\
            and attributes['parentFile']['signedUrl']:
            urls[self.uuid + '-' + '0002'] = attributes['parentFile']['signedUrl']

        # Add layer urls
        for layer in attributes['layers']:
            for i, url in enumerate(layer['content']):
                urls[layer['uuid'] + '-' + str(i).zfill(4)] = url

        # Add child layer urls
        if 'children' in attributes:
            for layer in [layer for child in attributes['children'] for layer in child['layers']]:
                for i, url in enumerate(layer['content']):
                    urls[layer['uuid'] + '-' + str(i).zfill(4)] = url

        return urls

    def export_json(self) -> None:
        """Stores initial dataset information and attributes in a JSON file for logging purposes."""

        with open('logs/datasets.json', 'a') as f:
            json.dump(
                obj={'attributes': self.attributes, 'info': self.info},
                fp=f,
                indent=4,
            )
            f.write(',\n')

    def reprocess(self, processes: list[DiscoveryTask] = None) -> None:
        """
        Reprocesses the failed processes of the acquisition.

        Args:
            processes: If given, only a subset of all processes will be relaunched.
        """

        if not self.acquisition:
            raise UnauthorizedCallException

        self.acquisition.reprocess()

    def __str__(self) -> str:
        return self.uuid.split('-')[0]

class Layer(Entity):
    """
    A layer in `Dataset.attributes["layers"] <src.discovery.entities.Dataset.attributes>`.

    Args:
        parent: Parent dataset.
        attributes: Attributes fetched from Discovery.

    .. seealso::
        `ChildLayer <src.discovery.entities.ChildLayer>`
            A layer of a child dataset.
    """

    def __init__(self, parent: Dataset, attributes: dict):
        super().__init__(parent=parent, attributes=attributes)
        #: The parent dataset of the layer
        self.parent: Dataset
        self.folder: pathlib.Path = parent.folder if (parent and parent.folder) else None
        self.pbar: tqdm.tqdm = None
        self.rem: int = None
        self.all: bool = None
        self.create_pbar: bool = False

    @Entity._catch_errors_in_get_children
    def get_children(self) -> None:
        """Returns None because layers don't have any children."""

        return None

    def pre_filter(self, configs: dict) -> bool:
        """Checks whether the layer passes the preliminary filters."""

        # Check layer type and layer name
        if not (
            self.attributes['scanType'] in {
                LayerType.OCT.value,
                LayerType.FUNDUS.value,
                LayerType.RAW.value,
            }
            or self.attributes['name'] in {
                'volume',
                'pdf',
                'ecrf',
                'form',
                'biomarkers',  # CHECK: It should only exist in child datasets
                'segmentation',  # CHECK: It should only exist in child datasets
                'etdrs_thickness',  # CHECK: It should only exist in child datasets
                'etdrs_volume',  # CHECK: It should only exist in child datasets
                'projection',  # CHECK: It should only exist in child datasets
                'thickness',  # CHECK: It should only exist in child datasets
            }
        ):
            self.log_filtered_out(f'Layer not expected ({self.attributes["name"]}, \
                {self.attributes["scanType"]}, {str(self.attributes["scanVariant"])})')
            return False

        # Check configurations
        # TODO: Minimize this conditions with possible Layers
        if any([
            self.attributes['scanType'] == LayerType.OCT.value and not configs['types']['oct'],
            self.attributes['scanType'] == LayerType.FUNDUS.value and not configs['types']['fundus'],
            self.attributes['scanType'] == LayerType.RAW.value and not configs['types']['rawimage'],
            self.attributes['name'] == 'pdf' and not configs['types']['pdf'],
            self.attributes['name'] == 'ecrf' and not configs['types']['ecrf'],
            self.attributes['name'] == 'form' and not configs['types']['ecrf'],
            self.attributes['name'] == 'biomarkers' and not configs['types']['biomarkers'],
            self.attributes['name'] == 'segmentation' and not configs['types']['segmentation'],
            self.attributes['name'] == 'etdrs_thickness' and not configs['types']['thicknesses'],
            self.attributes['name'] == 'etdrs_volume' and not configs['types']['volumes'],
            self.attributes['name'] == 'projection' and not configs['types']['projection_images'],
            self.attributes['name'] == 'thickness' and not configs['types']['thickness_images'],
        ]):
            self.log_filtered_out(f'Type ({self}) is not wanted')
            return

        # False if content is empty
        if not self.attributes['content']:
            self.log_filtered_out('No content')
            return False

        return True

    def post_filter(self, configs: dict) -> bool:
        """Checks whether the layer passes the posterior filters."""

        return True

    def download(self, downloader: DownloadManager) -> None:
        """
        Downloads the layer based on its type.

        Args:
            downloader: The download manager.
        """

        # Download OCT, fundus, angio
        if self.attributes['scanType'] in [LayerType.OCT.value, LayerType.FUNDUS.value]:
            # Create folder
            if self.attributes['scanType'] == LayerType.OCT.value:
                folder = self.folder / 'oct' / self.attributes['name']
            elif self.attributes['scanType'] == LayerType.FUNDUS.value:
                # CHECK: Why 10?
                if len(self.attributes['content']) > 10:
                    folder = self.folder / 'angio'
                else:
                    folder = self.folder / 'fundus'
            folder.mkdir(parents=True)

            # Store information
            with open(folder / 'info.json', 'w') as f:
                json.dump(
                    obj={
                        'count': len(self.attributes['content']),
                        'variants': sorted([LayerVariant(variant).name for variant in self.attributes['scanVariant']]),
                        'shape': self.attributes['shape'],
                        'scale': self.attributes['scale'],
                        'spacing': self.attributes['spacing'],
                        'range': [count_dim * spacing_dim for count_dim, spacing_dim in zip(self.attributes['shape'], self.attributes['spacing'])],
                    },
                    fp=f,
                    indent=4,
                )

            # Download images
            for i in range(len(self.attributes['content'])):
                out = folder / (str(i).zfill(4) + '.jpg')
                url_uuid = self.uuid + '-' + str(i).zfill(4)
                downloader.add(dataset=self.parent, url_uuid=url_uuid, out=out)

        # Download children and other layers
        elif self.attributes['scanType'] == LayerType.OTHER.value:
            # Define file name and folder
            if self.attributes['name'] == 'etdrs_volume':
                name = 'volume.json'
                folder = None
            elif self.attributes['name'] == 'etdrs_thickness':
                name = 'thickness.json'
                folder = None
            elif self.attributes['name'] == 'biomarkers':
                name = 'biomarkers.json'
                folder = None
            elif self.attributes['name'] == 'segmentation':
                name = '.svg'
                folder = self.folder / 'segmentation'
            elif self.attributes['name'] == 'projection':
                name = '.png'
                folder = self.folder / 'projection'
            elif self.attributes['name'] == 'thickness':
                name = '.png'
                folder = self.folder / 'thickness'
            else:
                # NOTE: In case of eCRF or form
                return

            # Add to downloads
            if folder:
                folder.mkdir()
                for i in range(len(self.attributes['content'])):
                    out = folder / (str(i).zfill(4) + name)
                    url_uuid = self.uuid + '-' + str(i).zfill(4)
                    downloader.add(dataset=self.parent, url_uuid=url_uuid, out=out)
            else:
                out = self.folder / name
                url_uuid = self.uuid + '-' + '0000'
                downloader.add(dataset=self.parent, url_uuid=url_uuid, out=out)

        # Download Raw layers (PDF and Other Acquisitions)
        elif self.attributes['scanType'] == LayerType.RAW.value:
            if self.attributes['name'] == 'pdf':
                name = 'report.pdf'
            elif self.attributes['name'] == 'volume':
                name = 'rawimage.jpeg'
            else:
                name = None
                logger.debug(f'{repr(self)} | Unexpected layer encountered: {self.attributes["name"]}.')

            if name:
                out = self.folder / name
                url_uuid = self.uuid + '-' + '0000'
                downloader.add(dataset=self.parent, url_uuid=url_uuid, out=out)

    def copy(self) -> None:
        """
        There is no possibility to move single layers in Discovery.
        So this method should never be called.

        Raises:
            NotImplementedError: If the method is called.
        """

        raise NotImplementedError

    def extend(self) -> None:
        """Augments the scan type and scan variants of the layer."""

        # Extend scanType if None
        # NOTE: etdrs_volume, etdrs_thickness, biomarkers, segmentation, projection, thickness
        if self.attributes['scanType'] in {None, 'None'}:
            self.attributes['scanType'] = LayerType.OTHER.value

        # Extend scanVariant for forms and eCRFs
        if self.attributes['scanType'] == LayerType.OTHER.value:
            if self.attributes['name'] == 'form':
                self.attributes['scanVariant'] = [LayerVariant.FORM.value]
            elif self.attributes['name'] == 'ecrf':
                self.attributes['scanVariant'] = [LayerVariant.ECRF.value]

        # Detect reconstruction layer
        # CHECK: What is reconstruction?
        if self.attributes['scanType'] == LayerType.FUNDUS and self.attributes['scanVariant'] == []:
            self.attributes['scanVariant'].append(LayerVariant.F_RECONSTRUCTION.value)

        # Extend scanVariants by doing taxonomy
        # TODO: Do it in the upload pipeline
        # TODO: Read the associated Dicom tag / layer attribute
        # if configs['general']['taxonomy']:
        #     self.get_taxonomy()

    # UNUSED
    # NOTE: Might be needed later
    def get_taxonomy(self) -> None:
        """Classifies the image and adds the classification to the scan variants."""

        # Skip if OCT, multicolor fundus, or reconstruction
        if any([
            self.attributes['scanType'] == LayerType.OCT.value,
            LayerVariant.F_MFI.value in self.attributes['scanVariant'],
            LayerVariant.F_RECONSTRUCTION.value in self.attributes['scanVariant'],
        ]):
            return

        if LayerVariant.F_CFI.value in self.attributes['scanVariant']:
            classifier = CFIClassifier(layer=self)
        else:
            classifier = FundusClassifier(layer=self)
        self.attributes['scanVariant'].extend(classifier.classify())

    def __str__(self):
        return self.attributes['name']

class ChildLayer(Entity):
    """
    A layer of a child dataset (generated by processors).

    The following layers are usually child layers:

    .. hlist::
        * ``"biomarkers"``
        * ``"etdrs_thickness"``
        * ``"etdrs_volume"``
        * ``"segmentation"``
        * ``"projection"``
        * ``"thickness"``

    Args:
        parent: Parent dataset.
        attributes: Attributes fetched from Discovery.

    .. seealso::
        `Layer <src.discovery.entities.Layer>`
            A layer of a dataset.
    """

    def __init__(self, parent: Dataset, attributes: dict):
        super().__init__(parent=parent, attributes=attributes)
        #: The parent dataset of the child layer
        self.parent: Dataset
        self.folder: pathlib.Path = parent.folder / 'children' if (parent and parent.folder) else None
        self.pbar: tqdm.tqdm = None
        self.rem: int = None
        self.all: bool = None
        self.create_pbar: bool = False

    @Entity._catch_errors_in_get_children
    def get_children(self) -> None:
        """Returns ``None`` because child layers don't have any children."""

        return None

    def pre_filter(self, configs: dict) -> bool:
        """Checks whether the layer passes the preliminary filters."""

        # Check layer name
        if not self.attributes['name'] in {
                'biomarkers',
                'segmentation',
                'etdrs_thickness',
                'etdrs_volume',
                'projection',
                'thickness',
            }:
            self.log_filtered_out(f'Child layer not expected ({self.attributes["name"]})')
            return False

        # False if content is empty
        if not self.attributes['content']:
            self.log_filtered_out('No content')
            return False

        # Check configurations
        if any([
            self.attributes['name'] == 'biomarkers' and not configs['types']['biomarkers'],
            self.attributes['name'] == 'segmentation' and not configs['types']['segmentation'],
            self.attributes['name'] == 'etdrs_thickness' and not configs['types']['thicknesses'],
            self.attributes['name'] == 'etdrs_volume' and not configs['types']['volumes'],
            self.attributes['name'] == 'projection' and not configs['types']['projection_images'],
            self.attributes['name'] == 'thickness' and not configs['types']['thickness_images'],
        ]):
            self.log_filtered_out(f'Type ({self}) is not wanted')
            return

        return True

    def post_filter(self, configs: dict) -> bool:
        """Checks whether the layer passes the posterior filters."""

        return True

    def download(self, downloader: DownloadManager) -> None:
        """
        Downloads the child layer based on its type.

        Args:
            downloader: The download manager.
        """


        # Create the children folder if it does not exist
        # MODIFY: Not the best place to do it
        if not self.folder.exists():
            self.folder.mkdir()

        # Define file name and folder
        if self.attributes['name'] == 'biomarkers':
            name = 'biomarkers.json'
            folder = None
        elif self.attributes['name'] == 'segmentation':
            name = '.svg'
            folder = self.folder / 'segmentation'
        elif self.attributes['name'] == 'etdrs_thickness':
            name = 'thickness.json'
            folder = None
        elif self.attributes['name'] == 'etdrs_volume':
            name = 'volume.json'
            folder = None
        elif self.attributes['name'] == 'projection':
            name = '.png'
            folder = self.folder / 'projection'
        elif self.attributes['name'] == 'thickness':
            name = '.png'
            folder = self.folder / 'thickness'
        else:
            return

        # Add to downloads
        if folder:
            folder = enumerate_path(folder)
            folder.mkdir()
            for i in range(len(self.attributes['content'])):
                out = folder / (str(i).zfill(4) + name)
                url_uuid = self.uuid + '-' + str(i).zfill(4)
                downloader.add(dataset=self.parent, url_uuid=url_uuid, out=out)
        else:
            out = self.folder / name
            out = enumerate_path(out)
            open(out, 'w').close()  # Create the file before downloading it
            url_uuid = self.uuid + '-' + '0000'
            downloader.add(dataset=self.parent, url_uuid=url_uuid, out=out)

        # Store child layer biomarkers
        if folder and 'biomarkers' in self.attributes and self.attributes['biomarkers']:
            with open(folder / 'biomarkers.json', 'w') as f:
                json.dump(
                    obj=self.attributes['biomarkers'],
                    fp=f,
                    indent=4,
                )

    def copy(self) -> None:
        """
        There is no possibility to move single layers in Discovery.
        So this method should never be called.

        Raises:
            NotImplementedError: If the method is called.
        """

        raise NotImplementedError

    def extend(self) -> None:
        # Not needed for now
        ...

    def __str__(self):
        return self.attributes['name'] + '(child)'
