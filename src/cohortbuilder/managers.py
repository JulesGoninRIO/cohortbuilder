"""
This module includes the classes that manage connections to external sources.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import pathlib
import requests
import subprocess
import shutil
import os
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from functools import wraps
from queue import Empty, Queue
from threading import Thread
from time import sleep, time
from typing import TYPE_CHECKING, Callable, Union, Tuple, Iterable, TypeVar, Generator
from shutil import get_terminal_size
from random import random

import paramiko
import pandas as pd
import tqdm
from loguru import logger
from paramiko.ssh_exception import (AuthenticationException,
                                    BadHostKeyException, SSHException)
from paramiko.channel import ChannelFile, ChannelStderrFile
import pyodbc

from src.cohortbuilder.discovery.exceptions import (NoUrlPassedException,
                                      RequestExpiredException,
                                      RequestMaxAttemptsReached,
                                      TokenRefreshMaxAttemptsReached,
                                      TokenAllCredentialsInvalid,
                                      UrlNotFoundException)
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.definitions import UploadPipelineFileStatus
from src.cohortbuilder.utils.helpers import read_json
from src.cohortbuilder.files import DicomFile, _DICOM_METADATA_TAGNAMES, DicomTag, GenericFileUploadWrapper
from src.cohortbuilder.tools.list_and_delete_pending import delete_acquisition_and_file, delete_discovery_file
from src.cohortbuilder.discovery.definitions import DiscoveryTaskStatus, DiscoveryFileStatus
from src.cohortbuilder.discovery.exceptions import UnknownStatusCodeException

if TYPE_CHECKING:
    from src.cohortbuilder.discovery.definitions import DiscoveryTask
    from src.cohortbuilder.discovery.entities import Dataset, Workbook
    from src.cohortbuilder.discovery.file import Acquisition
    from src.cohortbuilder.utils.pauser import Pauser


class DataBase:
    """
    Manages connection to a database.

    Args:
        server: The address of the server.
        name: The name of the database.
        driver: The name of the ODBC driver.
          If ``None``, it will be fetched from the settings. Defaults to ``None``.

    Examples:
        >>> from src.managers import DataBase
        >>> medisight = DataBase(server='dbmedisightprd\medisight', name='medisight')
        >>> query = ...
        >>> r = medisight.send(query=query)

    .. seealso::
        `src.managers.Client`
            Manages SSH and SFTP connections.
    """

    def __init__(self, server: str, name: str, driver: str = None):
        # Store the database server and name
        #: The address of the server
        self.server: str = server
        #: The name of the database
        self.name: str = name
        #: The name of the driver
        self.driver: str = driver if driver else Parser.settings['medisight']['driver']

        # Connect to the database
        # NOTE: Only works if a valid Kerberos ticket is available
        # NOTE: You can check the ticket with `klist` and create a new ticket with `kinit`
        # TODO: Make this automatic if it is necessary
        self.connection = pyodbc.connect(f'DRIVER={{{driver}}};SERVER={server};DATABASE={name};Trusted_Connection=YES;TrustServerCertificate=Yes')

    def send(self, query: str, chunksize: int = None) -> Union[pd.DataFrame, Generator(pd.DataFrame)]:
        """
        Sends a SQL query and returns the response.

        Args:
            query: The query that needs to be sent.
            chunksize: If specified, returns an iterator where chunksize is
              the number of rows to include in each chunk.
              It should be used for querying large tables.
              Defaults to ``None``.


        Returns:
            The response of the SQL query.
            If ``chunksize`` is passed it will be an iterator.
        """

        response = pd.read_sql_query(sql=query, con=self.connection, chunksize=chunksize)

        return response

class Client:
    """
    Manages SSH and SFTP connections.

    Args:
        settings: The dictionary of settings including these fields:
          ``hostname``, ``port``, ``username``, ``password``, ``keytype``, ``key``.
        sftp: If ``True``, an SFTP session will be opened on the SSH server.
          Defaults to ``False``.

    Examples:
        >>> from src.managers import Client
        >>> settings = ...
        >>> heyex = Client(settings=settings, sftp=True)
        >>> stdout, stderr = heyex.run(command='pwd')

    .. seealso::
        `src.managers.DataBase`
            Manages connection to a database.
    """

    def __init__(self, settings: dict, sftp: bool = False):
        # Store the settings
        self.open_sftp = sftp
        self.hostname = settings['hostname']
        self.port = settings['port']
        self.username = settings['username']
        self.password = settings['password']
        self.keytype = settings['keytype']
        self.key = settings['key']

        # Initialize the clients
        self.ssh: paramiko.SSHClient = None
        self.sftp: paramiko.SFTPClient = None

        # Connect the clients
        self.refresh()

    def refresh(self) -> None:
        """Creates SSH and SFTP clients to the server."""

        self.ssh = self.connect()
        if self.open_sftp:
            self.sftp = self.ssh.open_sftp()

    def connect(self) -> paramiko.SSHClient:
        """Connects to the SSH server."""

        client = paramiko.SSHClient()
        known_hosts = pathlib.Path('cache/known_hosts.tmp')
        with open(known_hosts, 'w') as f:
            f.write(f'{self.hostname} {self.keytype} {self.key}')
        client.load_host_keys(filename=known_hosts)
        known_hosts.unlink()
        try:
            client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
                look_for_keys=False,
            )
        # TODO: Handle exceptions
        except BadHostKeyException as e:
            logger.info(f'SSH to {self.hostname} failed: The server’s host key could not be verified.')
            raise e
        except AuthenticationException as e:
            logger.info(f'SSH to {self.hostname} failed: Authentication failed.')
            raise e
        except SSHException as e:
            logger.info(f'SSH to {self.hostname} failed: Authentication failed.')
            raise e
        except Exception as e:
            logger.info(f'SSH to {self.hostname} failed: Authentication failed.')
            raise e

        return client

    def isalive(self) -> bool:
        """Checks if the SSH connection is alive."""

        if self.ssh.get_transport() is None:
            return False
        else:
            return self.ssh.get_transport().is_active()

    def run(self, command: str) -> Tuple[ChannelFile, ChannelStderrFile]:
        """
        Runs a command on the SSH server.

        Args:
            command: The terminal command.

        Returns:
            Standard output and standard error channels.
        """

        if not self.isalive():
            self.refresh()
        _, ssh_stdout, ssh_stderr = self.ssh.exec_command(command)

        return ssh_stdout, ssh_stderr

    def close(self) -> None:
        """Closes the SSH and the SFTP clients."""

        if self.ssh is not None:
            self.ssh.close()
        if self.sftp is not None:
            self.sftp.close()

class MaximumThreadsReachedException(Exception):
    """Exception for reaching the maximum number of allowed threads."""

class QueueManager(ABC):
    """
    The abstract class for queue managers.
    A queue manager has multiple threads which pick items from a queue and
    process them in parallel.

    Args:
        n_threads: Number of threads for handling the queue items.
        name: The name of the threads.
        verbose: If ``True``, details will be logged. Defaults to ``True``.
        limited: If ``True``, the global limit on the number of threads will be applied.

    .. seealso::
        `src.managers.UploadManager`
            Queue manager for uploading files to Discovery
        `src.managers.DownloadManager`
            Queue manager for downloading files from Discovery
    """

    def __init__(self, n_threads: int, name: str = None, verbose: bool = True, limited: bool = True):
        # Initialize attributes
        #: Name of the queue manager (and its threads)
        self.name: str = name or self.__class__.__name__
        #: Indicator for logging details
        self.verbose: bool = verbose
        #: The download queue
        self.q: Queue = Queue()
        #: The list of the download threads
        self.threads: list[Thread] = list()
        #: Indicator for killing the threads
        self.kill_flag: bool = False
        self.limited = limited

        # Define the threads
        for idx in range(n_threads):
            t = Thread(target=self._worker, args=(idx,), name=f'{self.name}-{idx+1:03d}', daemon=True)
            self.threads.append(t)

    def _catch_errors(func: Callable[[int], None]) -> Callable[[int], None]:
        """Decorator for handling errors in the thread workers."""

        @wraps(func)
        def wrapper(self: DownloadManager, *args, **kwargs):
            # Execute the function
            try:
                return func(self, *args, **kwargs)
            # Catch Discovery related errors
            except (TokenRefreshMaxAttemptsReached, TokenAllCredentialsInvalid) as e:
                self.q.task_done()
                self.clear()
                logger.warning(f'Refreshing the token failed. Uploading some files might be skipped: {type(e).__name__}')
            # Catch OS-related errors
            except OSError as e:
                self.q.task_done()
                self.clear()
                msg = 'OSError'
                logger.error(msg)
                logger.exception(f'{msg}: {e}')
            # Catch other errors
            except Exception as e:
                self.q.task_done()
                self.clear()
                msg = f'Unexpected error occured ({type(e).__name__})'
                logger.error(msg)
                logger.exception(f'{msg}: {e}')

        return wrapper

    @_catch_errors
    def _worker(self, idx_thread: int) -> None:
        """
        The process that is run in each thread. It fetches an item from the queue,
        calls the task-specific worker method on it, and marks the item as done in
        the queue afterwards. It stops fetching items from the queue if the stop
        flag is toggled.

        Args:
            idx_thread: Index of the thread that is running this function.
        """

        while not self.kill_flag:
            # Try getting an item from the queue
            try:
                item = self.q.get(block=False)
            except Empty:
                continue

            # Call the task-specific worker
            self.process(item=item, idx_thread=idx_thread)

            # Mark the item as finished
            self.q.task_done()

        else:
            # Record the killing of the thread
            if self.verbose: logger.trace(f'Thread({self.threads[idx_thread].name}) is killed.')
            if self.limited: Parser.params['availablethreads'] += 1

    @abstractmethod
    def process(self, item: tuple, idx_thread: int) -> None:
        """
        Abstract method for the function that gets an item from
        the queue and processes it.

        Args:
            item: An item of the queue.
            idx_thread: Index of the thread that is running this function.
        """

        ...

    def launch(self) -> None:
        """Launches the threads."""

        # Raise an exception if the maximum number of threads is reached
        if self.limited and Parser.params['availablethreads'] < len(self.threads):
            msg = (f'Launching {len(self.threads)} threads is not allowed.'
                f'Only {Parser.params["availablethreads"]} threads are available.')
            raise MaximumThreadsReachedException(msg)

        # Launch the threads
        for t in self.threads:
            if self.limited: Parser.params['availablethreads'] -= 1
            t.start()

        # Log
        if self.verbose:
            logger.info(f'{len(self.threads)} threads are launched for {self.name}.')

    def clear(self) -> None:
        """Clears the queue."""

        size_init = self.q.qsize()
        for _ in range(size_init):
            try:
                _ = self.q.get(block=False)
                self.q.task_done()
            except Empty:
                break
        msg = (f'The queue of {self.name} is cleared of {size_init} items. '
            f'Unfinished tasks: {self.q.unfinished_tasks}.')
        logger.info(msg)

    def kill(self, join: bool = True) -> None:
        """
        Kills the threads by activating a flag.

        Args:
            join: If ``True``, the calling thread is paused until all the
              therads of this manager are joined.
        """

        self.kill_flag = True
        if self.verbose:
            logger.info(f'Killing the threads of {self.name}..')

        if join:
            self.join()

    def join(self) -> None:
        """
        Emulates Thread.join for all the threads. The calling thread is paused until all the
        therads of this manager are joined.
        """

        for t in self.threads:
            t.join()

    @property
    def isalive(self) -> bool:
        """Checks if any of the threads of the downloader is still alive."""

        for t in self.threads:
            if t.is_alive():
                return True
        else:
            return False

    def dead(self) -> None:
        """Raises AllThreadsDeadException."""

        raise AllThreadsDeadException

class MultiThreader(QueueManager):
    """
    Class for doing a process concurrently using multi-threading.
    Its primary use is for avoiding loops that are IO-bounded.

    Args:
        n_threads: Number of the threads.
        process: This function gets called on each item of the queue.
        items: Items to put in the queue. Defaults to ``None``.
        name: The name of the threads.
        verbose: If ``True``, details will be logged. Defaults to ``True``.
        limited: If ``True``, the global limit on the number of threads will be applied.
    """

    T = TypeVar('T')

    def __init__(self, n_threads: int, process: Callable[[T], None],
            items: Iterable[T] = None, name: str = None, verbose: bool = True, limited: bool = True):
        super().__init__(n_threads=n_threads, name=(name or process.__name__), verbose=verbose, limited=limited)
        self._callback_process: Callable[[self.T], None] = process
        if items: self.put(items=items)

    def process(self, item: T, idx_thread: int) -> None:
        """
        Calls the process on an item.

        Args:
            item: An item that needs to be processed.
            idx_thread: The index of the thread.
        """

        self._callback_process(item)

    def put(self, items: list) -> None:
        """
        Adds a set of items to the queue.

        Args:
            items: List of the items to be added to the queue in the same order.
        """

        for item in items:
            self.q.put(item)

    def execute(self) -> None:
        """Processes all the items in the queue and returns."""

        # Launch the threads
        if len(self.threads) > 1:
            if not self.isalive:
                self.launch()
        # Do it in the main thread if n_threads is 1
        else:
            while not self.q.empty():
                item = self.q.get()
                self.process(item=item, idx_thread=0)
                self.q.task_done()

        # Join the queue
        self.q.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Make sure to stop the threads
        if self.isalive:
            self.kill()
            self.join()

class DownloadManager(QueueManager):
    """
    Download manager class. Creates a download queue and
    downloads the items in the queue in multiple threads.

    Args:
        n_threads: Number of threads.
        configs: Configurations dictionary. Defaults to ``None``.

    .. seealso::
        :ref:`Configurations <buildconfigs>`
            Fields of the configurations file for building.
        `src.managers.UploadManager`
            Queue manager for uploading files to Discovery.
    """

    def __init__(self, n_threads: int, configs: dict = None):
        super().__init__(n_threads=n_threads)
        #: Download configurations
        self.configs: dict = configs

    def process(self, item: Tuple[Dataset, str, Union[pathlib.Path, str]], idx_thread: int) -> None:
        """
        Download an item of the queue and update the corresponding dataset object.

        Args:
            item: An item that needs to be downloaded.
            idx_thread: Index of the thread that is running this function.
        """

        # Unpack the item
        dataset, url_uuid, out = item

        # Download the content of the URL
        self.download(dataset, url_uuid, out)
        dataset.rem -= 1
        dataset.update()

    def download(self, dataset: Dataset, url_uuid: str, out: Union[pathlib.Path, str]) -> None:
        """
        Downloads a url from a dataset and stores it in the path given.
        It fetches the refreshed URL from the dataset given the UUID of the URL.

        Args:
            dataset: The dataset containing the URL.
            url_uuid: the UUID of the URL.
            out: The path of the output file.
        """

        for attempt in range(Parser.settings['general']['download_max_attempts']):
            # Get the refreshed URL
            url = dataset.get_url(url_uuid)

            # Download the URL and handle possible exceptions
            try:
                dataset.discovery.download(url, out)
                break

            # Handle the download-specific exceptions
            except NoUrlPassedException as e:
                logger.warning(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Download failed: {type(e).__name__}')
                logger.debug(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | UUID {url_uuid} \
                    of the URL is not available in the updated URLs ({dataset.urls}).')
                break
            except RequestExpiredException as e:
                logger.trace(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Attempt {attempt + 1} failed: {type(e).__name__}')
                logger.debug(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Unexpected {type(e).__name__} occured.')
                url = dataset.refresh_urls(force=True)
            except UrlNotFoundException as e:
                logger.warning(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Download skipped: {type(e).__name__}')
                break
            except RequestMaxAttemptsReached as e:
                logger.trace(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Attempt {attempt + 1} failed: {type(e).__name__}')
                sleep(20)

            # Handle the OS errors
            except OSError as e:
                if e.errno == 28:
                    logger.warning(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Download failed: No space left on device.')
                    break
                else:
                    logger.trace(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Attempt {attempt + 1} failed: {type(e).__name__}')
                    msg = f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Unexpected OSError occured.'
                    logger.debug(msg)
                    logger.exception(f'{msg}: {e}')

            # Handle other errors
            except Exception as e:
                logger.trace(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Attempt {attempt + 1} failed: {type(e).__name__}')
                msg = f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Unexpected exception occured.'
                logger.debug(msg)
                logger.exception(f'{msg}: {e}')

        # Log the download is failed if the maximum attempts is reached
        else:
            logger.warning(f'{repr(dataset)} > {out.relative_to(dataset.folder)} | Download failed: Maximum attempts reached.')

    def add(self, dataset: Dataset, url_uuid: str, out: Union[str, pathlib.Path]) -> None:
        """
        Adds a download URL to the queue.

        Args:
            dataset: Dataset associated to this download order.
            url_uuid: Corresponding UUID of the URL.
                For dataset URLs, it's the UUID of the dataset.
                For layer URLs, it's the UUID of the layer.
            out: The path to put the downloaded file.
        """

        dataset.rem += 1
        self.q.put(
            item = (
                dataset,
                url_uuid,
                pathlib.Path(out) ,
            ),
        )

class UploadManager(QueueManager):
    """
    Upload manager class. Creates an upload queue, fetches items from the Heyex servers
    and uploads them to Discovery in multiple threads.

    Args:
        n_threads: Number of threads.
        workbooks_raw: Workbooks to upload the raw (original) files to. Defaults to ``None``.
        workbooks_anm: Workbooks to upload the anonymized files to. Defaults to ``None``.

    .. seealso::
        `src.managers.DownloadManager`
            Queue manager for downloading files from Discovery
    """

    def __init__(self, n_threads: int, workbooks_raw: list[Workbook] = None, workbooks_anm: list[Workbook] = None, pauser: Pauser = None, better_anonymisation: bool = False):
        super().__init__(n_threads=n_threads)

        # Create the cache dir
        self.cache_dir = pathlib.Path(Parser.settings['general']['cache']) / 'tmp' / 'upload'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        #: Discovery workbooks to upload the unanonymized files to
        self.workbooks_raw: list[Workbook] = workbooks_raw if workbooks_raw else []
        #: Discovery workbooks to upload the anonymized files to
        self.workbooks_anm: list[Workbook] = workbooks_anm if workbooks_anm else []
        #: Discovery instances that correspond to the workbooks
        self.instances: set[str] = set([wb.discovery.instance for wb in (self.workbooks_raw + self.workbooks_anm)])
        #: Object for handling pause signals
        self.pauser = pauser
        # Mark whether uploader will do better anonymisation.
        # Done here because is a subclass of more generic classes which don't need to do this.
        self.better_anonymisation = better_anonymisation

    def process(self, item: Tuple[DicomFile, tqdm.tqdm, int], idx_thread: int) -> None:
        """
        Uploads an item of the queue to Discovery workbooks.

        - Transfers the dicom file to a local directory.
        - Rectifies the file tags.
        - Uploads the file to the first set of Discovery workbooks (raw).
        - Anonymizes the file.
        - Uploads the file to the second set of Discovery workbooks (anonymized).
        - Removes the local file.

        Args:
            item: An item that needs to be uploaded.
            idx_thread: Index of the thread that is running this function.
            upload_attempt: Internal argument so that re-upload attempts do not happen indefinitely.
        """

        # Unpack the item
        file, pbar, instance, upload_attempt = item

        # Update the file status
        file.status = UploadPipelineFileStatus.PICKED
        file.log_status_changed()

        # Check if the file exists locally
        if not file.path.exists():
            file.status = UploadPipelineFileStatus.ERROR
            file.log_status_changed(message='Local file does not exist.')
            return
        # Make a local copy
        file.copy(target=(self.cache_dir / file.name))
        file.status = UploadPipelineFileStatus.FETCHED
        # file.log_status_changed()

        # Enrich the copy
        file.rectify()
        file.status = UploadPipelineFileStatus.RECTIFIED
        # file.log_status_changed()

        # Upload the rectified file to Discovery
        my_workbooks_raw = [x for x in self.workbooks_raw if x.discovery.instance == instance]
        for workbook in my_workbooks_raw:
            file.upload(workbook=workbook)

        # Anonymize the copy
        file.anonymize(hide_patient_sex=False, new_anonymisation_method=self.better_anonymisation)
        file.status = UploadPipelineFileStatus.ANONYMIZED
        # file.log_status_changed()

        # Upload the anonymized copy to Discovery
        my_workbooks_anm = [x for x in self.workbooks_anm if x.discovery.instance == instance] if not isinstance(file, GenericFileUploadWrapper) else []
        for workbook in my_workbooks_anm:
            file.upload(workbook=workbook)

        # Delete the local copy if the original file is remote
        file.remove()
        file.status = UploadPipelineFileStatus.UPLOADED
        file.log_status_changed()

        if not file.discovery or file.discovery[instance].uuid is None: # Make sure that upload was successful, and linked a DiscoveryFile to this DicomFile. (Prevents crashes.)
            logger.warning(f'File {repr(file)} has no corresponding DiscoveryFile after upload: has not been successful.')
            # There's nothing we can do if the upload failed: we don't have a DiscoveryFile, nor any way to check status of what's going on, so we stop!
            file.status = UploadPipelineFileStatus.ERROR
            file.log_status_changed()
            if pbar: pbar.update()
            return

        # Wait until the file is successfully processed on all the instances
        start_time = time()
        PENDING_TIMEOUT = Parser.settings['general']['pending_acquisition_timeout']
        PENDING_TIMEOUT = PENDING_TIMEOUT * (1 + random() / 2) # Between 1 and 1.5 times
        UPLOAD_ATTEMPTS_MAX = Parser.settings['general']['upload_max_attempts']
        UNBLOCK_PENDING_FILES = Parser.args.unblock_pending_files
        FAILED_UPLOADS_DIR = Parser.settings['general']['cohorts_dir'] + '/' + Parser.args.logs_folder.parent.stem + '/' + Parser.args.logs_folder.stem + '-rejects'

        dfile = file.discovery[instance]
        first_500 = True
        grace_time_for_500 = 20
        while True:
            sleep(5) # Limit Discovery bombardment, this might be what's getting us banned
            problem = False
            # Update DiscoveryFile attributes
            try:
                dfile.fetch(acquisitions=True)
            except UnknownStatusCodeException as e:
                if 'ERROR 500' in str(e):
                    msg = f'Got an error 500 (internal error) from Discovery while fetching file info for {repr(dfile)}.'
                    if first_500:
                        msg += f' Waiting for {grace_time_for_500} seconds for problem to fix itself.'
                        logger.info(msg)
                        sleep(grace_time_for_500)
                        first_500 = False
                        continue # Retry
                    else:
                        msg += ' Deleting file.'
                        logger.warning(msg)
                        pbar.write(msg)
                        for wb in my_workbooks_raw + my_workbooks_anm:
                            delete_discovery_file(discovery_instance=dfile.discovery, wb=wb, discovery_file=dfile)
                        problem = True

            # Check whether processing the file has ended
            if dfile.isended:
                # If there's an error, the file will fail, even on successive reprocesses. Do like below, where we delete the file, and make a link to it in a folder for failed uploads.
                if dfile.status == DiscoveryFileStatus.ERROR:
                    msg = f'Discovery file {repr(dfile)} has ended with an error on {instance}, removing.'
                    logger.warning(msg)
                    pbar.write(msg)
                    for wb in my_workbooks_raw + my_workbooks_anm:
                        delete_discovery_file(discovery_instance=dfile.discovery, wb=wb, discovery_file=dfile)
                    problem = True
                else:
                    # Break if processing the file has been successful
                    if dfile.issuccessful and dfile.status:
                        file.status = UploadPipelineFileStatus.SUCCESSFUL
                        file.log_status_changed()
                        break
            else:
                # Check if parsing the file is stuck, or the processors are stuck. If they are, we delete the file from Discovery and retry by adding to back of queue
                if dfile.status == DiscoveryFileStatus.PARSING and (time() - start_time) > PENDING_TIMEOUT:
                    assert not dfile.acquisitions # Check empty (should be)
                    msg = f'Discovery file {repr(dfile)} is stuck in parsing state on {instance}, removing.'
                    if upload_attempt < UPLOAD_ATTEMPTS_MAX and UNBLOCK_PENDING_FILES:
                        msg += " Re-uploading."
                    else:
                        msg += " NOT attempting again."
                    logger.warning(msg)
                    pbar.write(msg)
                    for wb in my_workbooks_raw + my_workbooks_anm:
                        delete_discovery_file(discovery_instance=dfile.discovery, wb=wb, discovery_file=dfile)
                    problem = True

                for aq in dfile.acquisitions:
                    deleted = False
                    task_status = defaultdict(list)
                    for k, v in aq.tasks.items():
                        for one_instance_of_task in v:
                            task_status[k].append(one_instance_of_task['status'])
                    pending_status = [any([x == DiscoveryTaskStatus.PENDING or x == DiscoveryTaskStatus.DISPATCHED for x in y]) for y in task_status.values()] # Dispatched is just before pending, but still in queue basically. Should delete to get out of this state.
                    
                    for j, k in enumerate(task_status.keys()):
                        if pending_status[j] and (time() - start_time) > PENDING_TIMEOUT:
                            msg = f"{k.value} has timed out after {PENDING_TIMEOUT:.1f} seconds for {repr(aq)}."
                            if upload_attempt < UPLOAD_ATTEMPTS_MAX and UNBLOCK_PENDING_FILES:
                                msg += " Re-uploading."
                            else:
                                msg += " NOT attempting again."
                            logger.warning(msg)
                            pbar.write(msg)
                            if not deleted:
                                for wb in my_workbooks_raw + my_workbooks_anm:
                                    delete_acquisition_and_file(discovery_instance=dfile.discovery, wb=wb, aq=aq)
                                deleted = True
                            problem = True
            
            if problem and UNBLOCK_PENDING_FILES:
                logger.info(f'Re-trying stuck upload for file {repr(file)}.')
                sleep(3) # Do not bombard Discovery too hard
                self.__add(file=file, pbar=pbar, instance=instance, upload_attempt=upload_attempt+1) # Send to back of queue
                return # Do *not* update the progress bar!
            elif problem:
                logger.info(f'Making link to failed file {repr(file)} in {FAILED_UPLOADS_DIR}.')
                output_symlink = pathlib.Path(FAILED_UPLOADS_DIR + '/' + file.path.name)
                try:
                    shutil.copy(src=file.path.as_posix(), dst=output_symlink.as_posix())
                except PermissionError:
                    pass # We can get an error while trying to chmod on Windows filesystems, but we can safely ignore it.
                break

            # Wait until the file is processed, unless uploading in one batch
            if Parser.args.nowait:
                break
            elif self.pauser.forced:
                logger.warning(f'{repr(file)} | Skipped waiting for the Discovery file to get fully processed. It might need further inspection.')
                break
            else:
                sleep(3)

        # Update the pbar
        if pbar: pbar.update()

    def add(self, file: DicomFile, pbar: tqdm.tqdm = None) -> None:
        """
        Adds an item to the upload queue.

        Args:
            file: File that has to be uploaded.
            pbar: Progress bar to update once the file is processed.
        """

        for instance in self.instances:
            self.__add(file=file, pbar=pbar, instance=instance, upload_attempt=1)
    

    def __add(self, file: DicomFile, pbar: tqdm.tqdm, instance: str, upload_attempt: int) -> None:
        """
        Internal function to add an item to the upload queue.

        Args:
            file: File that has to be uploaded.
            pbar: Progress bar to update once the file is processed.
            instance: The specific Discovery instance to which we want to upload.
            upload_attempt: Internal argument so that re-upload attempts do not happen indefinitely.
        """

        self.q.put(item=(file, pbar, instance, upload_attempt))

class ReprocessManager(QueueManager):
    """
    Manager class for reprocessing. Reprocesses items in batches

    Args:
        n_threads: Number of threads.
    """

    def __init__(self, n_threads: int, processes: list[DiscoveryTask] = None):
        super().__init__(n_threads=n_threads)
        self.processes = processes

    def process(self, item: Tuple[Acquisition, tqdm.tqdm], idx_thread: int) -> None:
        """
        Fetches and reprocesses an item of the queue.
        """

        # Unpack the item
        acquisition, pbar = item

        # Reprocess the acquisition
        acquisition.reprocess(processes=self.processes)
        logger.trace(f'{repr(acquisition)} | Reprocesses requested.')

        # Wait until the acquisition is successfully processed
        reprocesses = 1
        while True:
            # Update the acquisition attributes
            acquisition.fetch(status=True)
            # Check whether processing the acquisition has ended
            if acquisition.isended:
                # Break if processing the acquisition has been successful
                if acquisition.issuccessful:
                    logger.info(f'{repr(acquisition)} | Successfully reprocessed.')
                    break
                # Report failed reprocess
                logger.trace(f'{repr(acquisition)} | Attempt {reprocesses} for reprocessing the failed processes failed.')
                # Break if maximum reprocesses has been reached
                if reprocesses >= Parser.settings['general']['reprocess_max_attempts']:
                    logger.warning(f'{repr(acquisition)} | Skipped. Maximum reprocessing attempts reached.')
                    break
                # Reprocess the failed processes otherwise
                reprocesses += 1
                acquisition.reprocess(processes=self.processes)
                logger.trace(f'{repr(acquisition)} | Reprocesses requested.')

            # Wait before the next fetch
            sleep(3)

        # Update the pbar
        if pbar: pbar.update()

    def add(self, acquisition: Acquisition, pbar: tqdm.tqdm = None) -> None:
        """
        Adds an item to the upload queue.

        Args:
            acquisition: Discovery acquisition that needs to be reprocessed.
            pbar: Progress bar to update once the acquisition is reprocessed.
        """

        self.q.put(item=(acquisition, pbar))

class AllThreadsDeadException(Exception):
    """Exception for handling the case when all the threads of the downloader are dead."""

class HeyexMetadataManager:
    """Class for managing the Heyex metadata files."""

    #: Whether the class has been initialized
    initialized: bool = False
    #: Path to the metadata file
    metadata: pathlib.Path = None
    #: Path to the register file
    register: pathlib.Path = None
    # Path to the register database created by {m,p}locate programs
    linux_register_database: pathlib.Path = None
    #: Root directory of the image pools
    DIR: pathlib.Path = None

    @classmethod
    def init(cls):
        # Skip if already initialized
        if cls.initialized:
            return
        # Set the class attributes
        CACHE_DIR = pathlib.Path(Parser.settings['general']['cache']) / 'heyex'
        CACHE_DIR.mkdir(exist_ok=True)
        cls.metadata = CACHE_DIR / 'metadata.csv'
        cls.register = CACHE_DIR / 'register.json'
        cls.linux_register_database = CACHE_DIR / 'locate_database.db'
        cls.DIR = pathlib.Path(Parser.settings['heyex']['root'])
        cls.initialized = True

    @classmethod
    def update(cls) -> None:
        """
        Parses the dicom files in image pools and updates the cached metadata.
        A register file is used to skip the files that are already parsed.
        If a register file is not found, the metadata file will be deleted and all the files will be parsed again.
        Parsing the files is done in 100 threads.
        """

        cls.init()
        logger.info('Updating the metadata of Heyex image pools.')

        # Read the register file
        if cls.register.exists():
            registered_files = set(read_json(cls.register))
        else:
            logger.info('Heyex metadata register not found. Metadata of all files will be extracted.')
            cls.metadata.unlink(missing_ok=True)
            registered_files = set()

        # Write headers to the metadata file
        if not cls.metadata.exists():
            with open(cls.metadata, 'w') as f:
                w = csv.writer(f)
                w.writerow(
                    ['FilePath', 'FileCreationTime', 'FileBytes']
                    + list(_DICOM_METADATA_TAGNAMES.values())
                )

        # Clean the register and metadata (remove the files of the relocated folders)
        # NOTE: The folders move and get renamed when a patient takes new scans
        register_tmp = cls.register.parent / (cls.register.stem + '.tmp.json')
        metadata_tmp = cls.metadata.parent / (cls.metadata.stem + '.tmp.csv')
        ## Remove the relocated files from registered files
        msg = 'Cleaning the old caches from the relocated folders..'
        logger.info(msg)
        print(msg)
        registered_files_df = pd.DataFrame({'file': list(registered_files)})
        registered_files_df['folder'] = registered_files_df['file'].apply(lambda p: (cls.DIR / p).parent.parent.parent)
        relocated_folders = []
        for folder in tqdm.tqdm(
            registered_files_df['folder'].unique(),
            desc=f'Folders of the old registered files'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=None,
        ):
            if not folder.exists():
                relocated_folders.append(folder.as_posix())
                registered_files_df.query('folder != @folder', inplace=True)
        registered_files = set(registered_files_df['file'].to_list())
        with open(register_tmp, 'w') as out:
            json.dump(list(registered_files), out)
        ## Re-write the metadata csv line by line
        with open(cls.metadata, 'r') as inp, open(metadata_tmp, 'w') as out:
            writer = csv.writer(out)
            reader = csv.reader(inp)
            writer.writerow(next(reader, None))  # Header row
            counter = {'rows': 0, 'relocated': 0}
            for row in tqdm.tqdm(
                reader,
                desc=f'Cleaning the old metadata files'.ljust(Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=None,
            ):
                counter['rows'] += 1
                path = cls.DIR / row[0]
                for folder in relocated_folders:
                    if path.is_relative_to(folder):
                        relocated = True
                        counter['relocated'] += 1
                        break
                else:
                    relocated = False
                if not relocated:
                    writer.writerow(row)
        msg = f'{counter["relocated"]} / {counter["rows"]} files (in {len(relocated_folders)} folders) have been relocated and are removed from the old metadata.'
        logger.info(msg)
        print(msg)
        ## Rename the files and delete the old ones
        cls.register.rename(cls.register.parent / (cls.register.stem + '.old' + '.json'))
        cls.metadata.rename(cls.metadata.parent / (cls.metadata.stem + '.old' + '.csv'))
        register_tmp.rename(cls.register)
        metadata_tmp.rename(cls.metadata)
        if cls.linux_register_database.exists():
            shutil.copy(cls.linux_register_database, str(cls.linux_register_database.parent / cls.linux_register_database.stem) + '.old' + '.db')

        # Open a progress bar and print message
        msg = 'Parsing Heyex image pools to find new files..'
        logger.info(msg)
        print(msg)
        start_time = time()
        found_new_paths = cls.__find_new_different_files()
        end_time = time()
        msg = f'Exploring Heyex pools to discover files took {int((end_time - start_time) // 60)} minutes.'
        del start_time
        del end_time
        logger.info(msg)
        print(msg)

        # Prepare pbar
        pbar = tqdm.tqdm(
            total=len(found_new_paths),
            desc=f'Extracting metadata of new files'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=None
        )

        msg = 'Started reading metadata of newly discovered files.'
        logger.info(msg)
        print(msg)

        write_queue = Queue()

        # Write metadata of one file.
        def getmetadata(file_location: str):
            file = DicomFile(path=file_location, mode='local')
            try:
                path = file.path.relative_to(cls.DIR).as_posix()
                stats = file.path.stat()
                meta = file.metadata
                write_queue.put([path, stats.st_ctime, stats.st_size] + list(meta))
                registered_files.add(path)
            except Exception as e:
                msg = f'Unexpected error for extracting the metadata of the file {file.path.as_posix()} ({type(e).__name__})'
                logger.debug(msg)
                logger.exception(f'{msg}: {e}')
            finally:
                if pbar: pbar.update()
        
        # Open the file (only once) for appending metadata
        with open(cls.metadata, 'a') as fp:
            def meta_writer(q: Queue):
                TIMEOUT = 30
                w = csv.writer(fp)
                while True:
                    try:
                        line_to_write = q.get(block=True, timeout=TIMEOUT)
                    except Empty:
                        logger.error(f'Timeout reached when waiting for queue contents in writer thread ({TIMEOUT} seconds). Terminating the thread immediately.')
                        return
                    if line_to_write == -1:
                        logger.info('Writer thread got signal to stop.')
                        return
                    w.writerow(line_to_write)

            # This MUST be single-threaded in order to not corrupt the data on write!!!!
            # (WRITING TO A FILE IS NOT THREAD-SAFE IN PYTHON)
            writer_thread = Thread(target=meta_writer,
                        args=tuple([write_queue]),
                        name='Heyex metadata writer',
                        daemon=True # Kill this thread if the main thread ends
            )

            # Launch in multiple threads. Can be done because pre-explored Heyex!
            with MultiThreader(n_threads=19, # 19 + 1 (writer) = 20
                               process=getmetadata,
                               items=found_new_paths,
                               name='Heyex metadata loaders',
                               verbose=True,
                               limited=True
                            ) as reader_threads:
                logger.info('Launching writer thread')
                writer_thread.start()

                logger.info('Executing reader threads (blocks main thread)')
                reader_threads.execute()
                logger.info('Reader threads done')

            write_queue.put(-1) # Stop writer
            writer_thread.join(timeout=3600) # Give one hour for the thread to terminate, and finish writing everything. Should be a very safe bet. (This is 1h *after* the reader thread is already done.)
            if writer_thread.is_alive():
                logger.error('Writer thread failed to terminate.')
            else:
                logger.info('Writer thread done')

        # Dump the list of the registered files
        with open(cls.register, 'w') as fp:
            json.dump(list(registered_files), fp)
        logger.info('Saved metadata file register')
        
        # Close progress bar
        if pbar: pbar.close()

        msg = 'Finished updating Heyex metadata cache!'
        logger.info(msg)
        print(msg)

    @classmethod
    def __find_new_different_files(cls) -> list[str]:
        old_reg_db_path = pathlib.Path(str(cls.linux_register_database.parent / cls.linux_register_database.stem) + '.old' + '.db')
        flat_dicom_index = '/tmp/flat_dicom_index.txt'
        old_flat_dicom_index = '/tmp/flat_dicom_index.old.txt'

        subprocess.run(args=['updatedb', # Update the old register cache
                     '-U', cls.DIR.as_posix(),
                     '-l', '0', # Build "unprotected" cache (other users can read it), but this mode allows non-root user to run the program
                     '-o', cls.linux_register_database.as_posix(),
                     '--prune-bind-mounts', 'no',
                     '--prunefs', '""',
                     '--prunepaths', '""' # Do not prune filesystems nor paths
                    ], check=True)
        
        intermediary_file = '/tmp/cb_processing.txt'
        with open(intermediary_file, 'w+') as f:
            subprocess.run(args=[
                        'locate',
                        '-d', cls.linux_register_database.as_posix(),
                        '*.dcm',
                ],
                check=True,
                stdout=f
            )
            f.seek(0) # Reset file pointer to start
            with open(flat_dicom_index, 'w') as dest:
                subprocess.run(args=['sort'],
                    stdin=f,
                    stdout=dest
                )
        os.unlink(intermediary_file) # Clean up

        if old_reg_db_path.exists():
            with open(intermediary_file, 'w+') as f:
                subprocess.run(args=[
                                'locate',
                                '-d', old_reg_db_path.as_posix(),
                                '*.dcm',
                    ],
                    check=True,
                    stdout=f
                )
                f.seek(0) # Reset file pointer to start
                with open(old_flat_dicom_index, 'w') as dest:
                    subprocess.run(args=['sort'],
                        stdin=f,
                        stdout=dest                    
                    )
            os.unlink(intermediary_file) # Clean up

            moved = subprocess.run(args=[
                            'diff',
                            old_flat_dicom_index,
                            flat_dicom_index,
                ],
                check=False, # Exit code of 1 means differences, this is what we expect! (But check=True stops the program in that case)
                capture_output=True,
                text=True
            ).stdout # Directly capture STDOUT from diff command.
            os.unlink(old_flat_dicom_index)
            os.unlink(flat_dicom_index)

            moved = moved.split('\n')
            added = [x[2:] for x in moved if '> ' in x]
            return added
            removed = [x[2:] for x in moved if '< ' in x]
        else:
            with open(flat_dicom_index, 'r') as mov:
                new = mov.read().split('\n')
            os.unlink(flat_dicom_index)
            # Make sure to exclude these, or run will crash.
            # Already done with above code ('> ' list comp) because only takes positive diff.
            new = [x for x in new if x != '.' and x != '..' and x != '']

            # Fall back to making difference with metadata cache file, if possible
            if cls.metadata.exists():
                old = pd.read_csv(cls.metadata, usecols=['FilePath'])['FilePath'].to_list()
                old = [str(cls.DIR) + '/' + x for x in old] # Prepend dir so that paths become absolute, like those from locate!
                added = list(set(new) - set(old)) # Difference between the two sets! (Equivalent to added above)
            else:
                added = new # Worst case. Rebuild everything because no index cache nor metadata cache exists.

            return added

    @classmethod
    def explore(cls, patients: list[PatientManager],
                filter_consent: bool = True, filter_inconsistent: bool = True, update: bool = False) -> list[dict]:
        """
        Explores Heyex metadata and finds the files that are related to a list of patient identifiers.

        - Filters out the PIDs that have not given consent.
        - Filters out the PIDs that have inconsistent information in Heyex.

        Args:
            patients: List of the patients for which files need to be found.
            filter_consent: If ``True``, the patients that have not given consent will be filtered out.
            filter_inconsistent: If ``True``, the patients that have inconsistent information in Heyex
              will be filtered out.
            update: If ``True``, the metadata of new files will be extracted before finding the related
              files. Defaults to ``False``.

        Returns:
            The related files (filtered).
        """

        # Initialize and update the metadata
        cls.init()
        if update: cls.update()

        # Load inconsistent pids
        inconsistent_pids = read_json(
            pathlib.Path(Parser.settings['general']['cache']) / 'heyex' / 'inconsistent_pids.json'
        )

        URM_PATIENTS = pd.read_excel(
            Parser.settings['general']['urm_patient_db_location'],
            usecols=['Numéro de patient', 'Début période', 'Fin période'],
            converters={
                'Début période': lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') if x else pd.NaT,
                'Fin période': lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') if x else pd.NaT,
            },
        ) # Get the df of all URM patient PIDs and their admission start + end date

        # Filter the patients
        pids = []
        slimsids = {}
        pbar = tqdm.tqdm(
            total=len(patients),
            desc=f'Filtering the patients'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=None,
        )
        for patient in patients:
            # Filter out if the PID is among the inconsistent list
            if filter_inconsistent and patient.is_inconsistent(inconsistent_pids):
                patient.log_filtered_out(f'Incosistent PID: {patient.cgpid}')
                continue

            # Filter out if the patient has not given consent, and isn't a URM patient prior to 2016
            URM_ENTRIES_PATIENT = URM_PATIENTS[URM_PATIENTS['Numéro de patient'] == patient.cgpid]
            URM_NO_NEED_CONSENT = (URM_ENTRIES_PATIENT.shape[0] != 0) and (URM_ENTRIES_PATIENT['Fin période'] < datetime.datetime(year=2016, month=1, day=1)).all()
            # The above line works because of lazy 'if' statement evaluation: if there are no entries, the resulting malformed output on the right never gets evaluated!

            if (filter_consent and not URM_NO_NEED_CONSENT) and not patient.has_consent(): # If we are filtering consent, patient needs to be checked for consent, and has not consented.
                patient.log_filtered_out(f'No consent')
                continue
            pids.append(patient.cgpid)
            # Store a mapping between the MediSIGHT and Slims patient identifiers
            slimsids[patient.cgpid] = patient.slimsid
            # Update the progress bar
            pbar.update()
        pbar.close()
        msg = f'{len(pids)}/{len(patients)} patients passed the filters.'
        logger.info(msg)
        print(msg)

        # Read the metadata
        msg = 'Reading the metadata of files on the image pools..'
        print(msg, end='')
        logger.info(msg)
        metadata = pd.read_csv(
            cls.metadata,
            usecols=(['FilePath'] + [_DICOM_METADATA_TAGNAMES[tag] for tag in [
                    DicomTag.PatientID,
                    DicomTag.PatientBirthDate,
                    DicomTag.StudyDate,
                ]]
            ),
            converters={
                'StudyDate': lambda x: datetime.strptime(str(x), '%Y%m%d') if x else pd.NaT,
            },
            dtype={
                'FilePath': str,
                _DICOM_METADATA_TAGNAMES[DicomTag.PatientID]: str,
                _DICOM_METADATA_TAGNAMES[DicomTag.PatientBirthDate]: int,
            },
        )
        msg = 'Done!'
        print(msg)
        logger.info(msg)

        # Get the file paths on Heyex for the wanted pids
        pbar = tqdm.tqdm(
            total=len(pids),
            desc=f'Finding related files on Heyex'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=None,
        )

        # Filter on date
        date = None   #TMP  # TODO: Read from a configs file
        if date:
            metadata = metadata.query('StudyDate <= @date')

        # Add the files of each patient
        files = []
        for pid in pids:
            paths = metadata[metadata[_DICOM_METADATA_TAGNAMES[DicomTag.PatientID]] == pid].FilePath.tolist()
            files.extend([{'pid': pid, 'path': path} for path in paths])
            pbar.update()
        pbar.close()
        msg = f'{len(files)} files are detected on Heyex.'
        logger.info(msg)
        print(msg)
        pids_with_at_least_one_path = len(set([f['pid'] for f in files]))
        proportion_with_at_least_one_path = pids_with_at_least_one_path / len(pids)
        msg = f'{proportion_with_at_least_one_path * 100:.2f}% of patients have at least one file detected on Heyex.'
        if proportion_with_at_least_one_path > 0.95:
            logger.info(msg)
        else:
            logger.warning(msg)
        print(msg)

        return files

    @classmethod
    # UNUSED
    # TODO: Finish
    def find_inconsistent_pids(cls) -> None:
        """
        Work in progress...
        Finds the inconsistencies in the Heyex metadata and writes them in a txt file.
        This file should then be checked to form a list of inconsistend PIDs which should
        be excluded in the uploads.

        The sources of the inconsistencies are:

        - typos/mistakes/etc.
        - Different variations of the same name.
        - Same `dcm_pid` for different patients.
        """

        # TODO: Read from the folders (short version does not work)
        ...

        # TODO: Extract patients and dcm_pids
        dcm_pids = ...
        patients = ...

        # NOTE: It takes around 9 hours!
        with open('inconsistent_pids.txt', 'w') as f:
            for pid in tqdm.tqdm(set(dcm_pids)):
                patient = [patient for patient in patients if patient['dcm_pid']==pid]
                names = [p['name'].upper() for p in patient]
                surnames = [p['surname'].upper() for p in patient]
                sexes = [p['sex'].upper() for p in patient]
                dobs = [p['dob'].split(' ')[0] for p in patient]
                pids_ = [p['pid'] for p in patient]

                if any([
                    len(set(names)) > 1,
                    len(set(surnames)) > 1,
                    len(set(sexes)) > 1,
                    len(set(dobs)) > 1,
                ]):
                    print(
                        '\t'.join([
                            f'{pid}',
                            str(Counter(names).most_common()),
                            str(Counter(surnames).most_common()),
                            str(Counter(sexes).most_common()),
                            str(Counter(dobs).most_common()),
                            str(Counter(pids_).most_common()),
                        ]),
                        file=f,
                    )

class PatientManager:
    """
    Class for managing patients for the upload pipelines.

    Args:
        pid: The patient identifier (patient hospital number).
    """

    #: The patient identifier to be used for testing purposes.
    TESTPID = '1552350'  # Ciara Bergin

    def __init__(self, pid: Union[str, int]):
        #: Patient identifier on MediSIGHT (hospital number)
        self.cgpid = str(pid)
        #: Patient identifier on Slims
        self.slimsid: str
        #: Consent status
        self.consent: str
        # Fetch Consent and patient identifier on SLIMS
        self.slimsid, self.consent = self.fetch_slims()

    def has_consent(self) -> bool:
        """
        Checks SLIMS to see if the patient has given consent.

        Returns:
            ``False`` if the patient has not given consent or there is no form on Slims.
            ``True`` otherwise, if the patient has not responded or has given consent.
        """

        # Skip for the test pid
        if self.cgpid == self.TESTPID:
            return True

        if not self.consent:
            logger.warning(f'{repr(self)} | Could not fetch the consent status.')
            return False
        elif self.consent in ['no', 'na', 'invalid']:
            return False
        elif self.consent in ['yes', 'pending']:
            return True
        else:
            logger.debug(f'{repr(self)} | Unexpected consent status: {self.consent}.')
            return False

    def is_inconsistent(self, inconsistent_pids: list) -> bool:
        """
        Checks if the patient's identifier is among the inconsistent identifiers.

        Args:
            inconsistent_pids: A list of th inconsistent PIDs.

        Returns:
            ``True`` if the patient CGPID is on the list of inconsistent PIDs,
            ``False`` otherwise.
        """

        # Skip for the test pid
        if self.cgpid == self.TESTPID:
            return False

        # Check for the other pids
        if self.cgpid in inconsistent_pids:
            return True
        else:
            return False

    def fetch_slims(self) -> Tuple[str, str]:
        """
        Fetches the SLIMS patient identification and patient consent.

        Returns:
            slimsid: Patient identifier on slims.
            status: Consent status on slims.
        """

        # Send a request
        url = Parser.settings['slims']['url'] % (self.cgpid)
        r = requests.get(
            url=url,
            auth=(Parser.settings['slims']['username'], Parser.settings['slims']['password']),
        )
        # Check the response
        if r.status_code != 200:
            logger.debug(f'Bad request for CGPID {self.cgpid} (ERROR {r.status_code}): (REASON: {r.reason}), (BODY: {r.request.body})')
            return (None, None)

        # Check the content
        entities = r.json()['entities']
        if entities:
            entity = entities[0]
        else:
            # NOTE: No record on SLIMS means that no form has been sent to the patient
            return 'na', 'na'

        # Fetch the information from the content
        status = None
        slimsid = None
        for c in entity['columns']:
            if c['name'] == 'cntn_id':
                slimsid = c['value'] if c['value'] else 'CG_00000000'
            elif c['name'] == 'cntn_cf_CGStatus':
                # NOTE: Empty (None) CGStatus on SLIMS means that the patient has not responded yet
                status = c['value'] if c['value'] else 'pending'
            if slimsid and status:
                break
        else:
            logger.debug(f'Patient with CGPID {self.cgpid} does not have valid info on Slims: {slimsid}, {status}')

        # Extract the outputs if they have been found
        if slimsid:
            slimsid = slimsid.split('_')[1]
        if status:
            status = status.lower()

        return slimsid, status

    def log_filtered_out(self, reason: str) -> None:
        """
        Logs the reason why the entity has been filtered out.

        Args:
            reason: Reason for being filtered out.
        """

        logger.info(f'{repr(self)} | Patient filtered out: {reason}.')

    def __repr__(self):
        return f'{self.__class__.__name__} "{self.slimsid}"'
