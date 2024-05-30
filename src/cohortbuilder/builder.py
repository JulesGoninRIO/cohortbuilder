"""
This module includes the class for managing the building and uploading processes.
"""

import json
from collections import Counter, deque
from functools import wraps
from typing import Callable, Union
import sys

from loguru import logger
import tqdm
import pathlib
import pytz
import shutil
from shutil import get_terminal_size
from datetime import datetime
from loguru import logger
from PIL import Image
from threading import Lock
import numpy as np
from joblib import delayed, Parallel
from multiprocessing import cpu_count
from itertools import chain

from src.cohortbuilder.discovery.definitions import LayerVariant
from src.cohortbuilder.discovery.discovery import Discovery
from src.cohortbuilder.discovery.entities import Project
from src.cohortbuilder.discovery.exceptions import TokenRefreshMaxAttemptsReached, TokenAllCredentialsInvalid
from src.cohortbuilder.discovery.queries import QueryBuilder, Q_PROJECTS
from src.cohortbuilder.discovery.manager import DiscoveryManager
from src.cohortbuilder.managers import AllThreadsDeadException, DownloadManager, HeyexMetadataManager, PatientManager, MultiThreader
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.files import DicomFile
from src.cohortbuilder.utils.helpers import read_json, list2str, strong_suppress_stderr_only
from src.cohortbuilder.utils.imageprocessing import get_clean_oct_segmentation, find_max_slice_presegmented, rebuild_thickness_map, generate_thickness_map_visualisation, compute_quadrant_stats, get_oct_scale_information
from src.cohortbuilder.definitions import index_to_cnn_modality


class Builder:
    """
    This class deals with building datasets from available workbooks in a project in Discovery.

    Args:
        configs: configurations (usually read from a file).
        instance: The name of the Discovery instance.
        project: The name of the Discovery project.
        workbooks: The names of the Discovery workbooks.

    Examples:
        >>> from src.parser import Parser
        >>> from src.builder import Builder
        >>> from src.cohortbuilder.utils.helpers import read_json
        >>>
        >>> args = ...
        >>> settings = read_json('settings.json')
        >>> Parser.store(args=args, settings=settings)
        >>> configs = ...
        >>>
        >>> with Builder(
        ...     configs=configs,
        ...     instance=Parser.args.instance,
        ...     project=Parser.args.project,
        ...     workbooks=Parser.args.workbooks,
        ... ) as builder:
        ...     if builder.create_folders():
        ...         builder.get_metadata()
        ...         builder.build()

    .. seealso::
        `src.parser.Parser`
            Class for parsing settings and arguments.
    """

    def __init__(self, configs: dict, instance: str, project: str, workbooks: list[str], noconfirm_resume: bool):
        # Instantiate the Discovery instance
        self.discovery = Discovery(instance=instance)
        # Store the configs
        self.configs: dict = self._check_configs(configs, project=project)
        # Instantiate the project
        self.project = Project(
            discovery=self.discovery,
            uuid=self.configs['general']['project_uuid'],
            folder=(Parser.args.cohorts_dir / self.configs['general']['name']),
            name=project,
        )
        #: The download manager
        self.downloader: DownloadManager = None

        # Build the dataset query
        self.query_builder: QueryBuilder = QueryBuilder()
        self.query_builder.build(self.configs)

        # Filter the workbooks
        wanted_workbooks = {workbook.strip().lower() for workbook in workbooks}
        all_workbooks = self.project.get_children()
        logger.info(f'All workbooks: {list2str(all_workbooks)}')
        workbooks = [workbook for workbook in all_workbooks if workbook.attributes['name'].strip().lower() in wanted_workbooks]
        logger.info(f'Pre-filtered workbooks: {list2str(workbooks)}')
        # Warn if some of the workbooks are not found
        if len(workbooks) < len(wanted_workbooks):
            msg = (
                'Some of the workbooks could not be found.'
                ' Make sure that all the workbooks are shared with the Cohort Builder user on Discovery.'
            )
            print(msg)
            logger.warning(msg)
        self.project.children = workbooks

        self.noconfirm_resume = noconfirm_resume

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Make sure to stop the threads
        if self.downloader and self.downloader.isalive:
            self.downloader.kill()

    def _catch_errors(func: Callable[[], None]) -> Callable[[], None]:
        """Decorator for handling errors in the builder."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the function
            try:
                return func(self, *args, **kwargs)
            except AllThreadsDeadException as e:
                logger.info('All the download threads stopped working, possibly due to errors.')
            except InvalidConfigsException as e:
                logger.warning(e.message)
                print(f'PROBLEM in the configs file: {e.message}')
                print(f'You can change this configuration by changing '
                    f'"{e.field}" in {(Parser.args.configs_dir / Parser.args.configs).absolute().as_posix()}.')
                exit()
            except (TokenRefreshMaxAttemptsReached, TokenAllCredentialsInvalid) as e:
                logger.info(f'Refreshing the token failed: {type(e).__name__}')
            except Exception as e:
                raise e

        return wrapper

    @ _catch_errors
    def _check_configs(self, configs: dict, project: str) -> dict:
        """
        Gets the configuration dictionary and checks the sanity of the configurations.
        Some fields will be modified in the returned configurations.

        Args:
            configs: The original configuration dictionary.
            project: The name of the Discovery project.

        Raises:
            InvalidConfigsException: If there is a problem with one of the configurations.

        Returns:
            dict: The modified configuration dictionary.
        """

        # Check the configs purpose
        if configs['purpose'] != 'build':
            raise InvalidConfigsException(
                    message=f'The provided configs file ({Parser.args.configs_dir / Parser.args.configs}) does not have proper format.',
                    field='purpose',
            )

        # TODO: Complete type check for configurations (null/list, null/bool, etc.)
        # TODO: Automatically extend configs ('male' -> 'M')
        # TODO: Taxonomy should be on if _taxonomy_needed() (it's needed in the variant filters)

        # Check the project name
        project_uuids = [
            p['uuid']
            for p in self.discovery.send_query(query=Q_PROJECTS)['data']['profile']['projects']
            if p['title'].lower() == project.lower() # Accept this as the "proper" notebook even if case unmatched.
        ]
        if len(project_uuids) == 1:
            configs['general']['project_uuid'] = project_uuids[0]
        else:
            raise InvalidConfigsException(
                message=(
                        f'There are {len(project_uuids)} projects '
                        f'named "{project}" '
                        f'in Discovery instance {self.discovery.instance}.'
                        f'\nMake sure the Cohort Builder user is a member of this project.'
                ),
                field='general.project',
            )

        configs['filters'] = configs['filters'] if 'filters' in configs.keys() else configs # For backwards-compatiblity with old templates

        return configs

    # UNUSED
    def _taxonomy_needed(self) -> bool:
        """
        Determines wether taxonomy is needed or not based on the configurations.

        Returns:
            bool: True if taxonomy is needed.
        """

        configs = self.configs

        # Fetch the variants
        dataset_variants = configs['filters']['datasets']['variants'] if configs['filters']['datasets']['variants'] else []
        study_variants = configs['filters']['studies']['variants'] if configs['filters']['studies']['variants'] else []

        # Split the variants
        dataset_variants = map(lambda x: {LayerVariant[y] for y in x.split('+')}, dataset_variants)
        study_variants = map(lambda x: {LayerVariant[y] for y in x.split('+')}, study_variants)

        # Fetch the fundus computed variants
        computed_variants = LayerVariant.fundus_variants_added()

        # Check the dataset variants
        for variant in dataset_variants:
            if variant.issubset(computed_variants):
                return True

        # Check the study variants
        for variant in study_variants:
            if variant.issubset(computed_variants):
                return True

        return False

    def askcontinue(self) -> bool:
        """
        Asks the user if they want to continue the process and returns the answer.
        The process is blocked in the meanwhile.
        """

        # If we have decided to skip confirmation prompts, just return yes
        if self.noconfirm_resume:
            return True

        # Get the command
        print('Would you like to continue (Y̲/N)?', end=' ')
        answer = input('Answer: ')

        # Return True if YES
        if answer.strip().upper() in ['Y', 'YES', '']:
            return True
        # Return False if NO
        elif answer.strip().upper() in ['N', 'NO']:
            return False
        # Retry if the answer is unvalid
        else:
            print(f'"{answer}" is not a valid answer. Please answer again.')
            return self.askcontinue()

    def create_folders(self) -> bool:
        """
        Checks the existence of the workbook folders and creates them.
        If a folder exists, the user will be asked if they want to continue
        the process. If the process is set to be continued, ``True`` will be returned.
        """

        flag = True
        for workbook in self.project.children:
            if workbook.folder.exists():
                # Ask the user if they want to continue
                msg = (
                    f'Workbook folder ({workbook.folder}) exists.'
                    ' Files that have not been downloaded before will be added to this folder.'
                    ' If an element in this cohort workbook is only partially downloaded in a previous run,'
                    ' all the downloads of that element will be skipped.'
                )
                logger.warning(msg)
                print(msg)
                if not self.askcontinue():
                    flag = False
                    break
            else:
                # Create the folder
                workbook.folder.mkdir()

        return flag

    @_catch_errors
    def get_metadata(self) -> None:
        """Fetches and stores the hierarchical metadata of the workbook."""

        # Return if metadata is not wanted
        if not self.configs['general']['metadata']:
            return

        for workbook in self.project.children:
            # Fetch the metadata
            patients = dict()
            wb_children = workbook.get_children()
            pbar = tqdm.tqdm(total=len(wb_children),
                desc=f'Fetching metadata ({repr(workbook)})'.ljust(Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=None
            )
            shared_patient_lock = Lock()
            def acquire_patient_info(patient):
                studies = dict()
                for study in patient.get_children():
                    datasets = dict()
                    for dataset in study.get_children():
                        datasets[dataset.uuid] = {
                            'attributes': dataset.attributes,
                            'info': dataset.info,
                        }
                    studies[study.uuid] = {
                        'studyId': study.attributes['studyId'],
                        'studyDatetime': study.attributes['studyDatetime'],
                        'modalities': study.attributes['modalities'],
                        'datasets': datasets,
                    }
                with shared_patient_lock: # This is a shared object between all threads, so we protect from race conditions
                    patients[patient.uuid] = {
                        'patientId': patient.attributes['patientId'],
                        'name': patient.attributes['name'],
                        'surname': patient.attributes['surname'],
                        'birthdate': patient.attributes['birthdate'],
                        'sex': patient.attributes['sex'],
                        'studies': studies,
                    }
                pbar.update(1)
            
            with MultiThreader(
                n_threads=Parser.settings['general']['threads'], # Max this out!
                process=acquire_patient_info,
                items=wb_children,
                name='Patient metadata gathering',
                verbose=True,
                limited=True
            ) as threader:
                threader.execute() # This blocks the main thread until done :)
            
            pbar.close()
            logger.info('Done fetching metadata')

            # Store the metadata
            with open(workbook.folder / 'patients.json', 'w') as f:
                json.dump(
                    obj=patients,
                    fp=f,
                    indent=None,
                )

    @_catch_errors
    # UNUSED
    def check_duplicates(self) -> None:
        """Reports duplicate patients in each workbook."""
        msg = 'Checking duplicates..'
        print(msg)
        logger.info(msg)
        for wb in self.project.children:
            patient_ids = [str(patient) for patient in wb.get_children()]
            freqs = Counter(patient_ids)
            for id, freq in freqs.items():
                if freq > 1:
                    logger.warning(f'{repr(wb)} | \
                        Patient ID {id} is repeated {freq-1} time(s).')

    @_catch_errors
    def __read_and_format_image(self,
                                image_path: str,
                                resizing_transformation: Callable,
                                tensor_transformation: Callable,
                                pytorch_module # The module itself (torch), on which you do calls (eg. torch.tensor)
                                ):
        '''
        Private class method to abstract away loading and formatting of an image.
        Requires passing in the torch module as an argument so that re-importing is not necessary.
        '''
        loaded_img = Image.open(image_path)
        loaded_img = tensor_transformation(loaded_img)
        if loaded_img.max() <= 1:
            loaded_img = loaded_img * 255 # Send to full [0, 255] RGB range
        tensor_img = loaded_img.to(dtype=pytorch_module.uint8)
        if len(tensor_img.size()) == 3 and tensor_img.size(dim=2) == 3: # This is an RGB image!
            tensor_img = tensor_img.permute(2, 0, 1)
        if len(tensor_img.size()) == 3 and tensor_img.size(dim=0) > 3: # This is a cube!
            tensor_img = tensor_img[tensor_img.size(dim=0) // 2, :, :]
        if len(tensor_img.size()) == 2:
            tensor_img = tensor_img.unsqueeze(dim=0)
        if tensor_img.size(dim=0) == 1:
            tensor_img = tensor_img.repeat(3, 1, 1)
        tensor_img = resizing_transformation(tensor_img)
        tensor_img = tensor_img.unsqueeze(dim=0) # Indicate batch size of 1
        tensor_img = tensor_img.to(dtype=pytorch_module.float) # Make dtypes compatible

        return tensor_img

    @_catch_errors
    def build(self) -> None:
        """
        Loops over the specified workbooks in Discovery,
        applies the filters set in the configs file, and downloads the filtered data.
        If a target workbook is set, the datasets that pass the criterion will be
        added to it.
        """

        # Instantiate the download manager
        self.downloader = DownloadManager(
            n_threads=Parser.settings['general']['threads'],
            configs=self.configs,
        )
        self.downloader.launch()

        n_total = len(self.project.children)
        pbar = tqdm.tqdm(
            total=n_total,
            desc=f'Processing the workbooks'.ljust(Parser.settings['progress_bar']['description']),
            ncols=get_terminal_size().columns,
            leave=None,
        )

        if 'reidentify_modality' in self.configs['general']:
            use_taxonomy_cnn = self.configs['general']['reidentify_modality'] and Parser.settings['general']['taxonomy_cnn_location'] is not None and Parser.settings['general']['taxonomy_cnn_input_size'] is not None # Bool is False if it's None or False (covers what we need to check for).
            logger.info('Reidentifying image modalities after download.')
        else:
            logger.info('NOT reidentifying image modalities after download.')
            use_taxonomy_cnn = False

        if use_taxonomy_cnn:
            import torch
            import warnings
            with warnings.catch_warnings():
                # Certain versions of torchvision.io have problems with .so files
                # But we don't use this submodule, so we don't care.
                warnings.simplefilter('ignore', category=UserWarning)
                from torchvision.transforms import Resize, ToTensor
            
            resizer_transform = Resize(size=Parser.settings['general']['taxonomy_cnn_input_size'], antialias=True)
            convert_img_to_tensor = ToTensor()
            neural_net = torch.jit.load(Parser.settings['general']['taxonomy_cnn_location'])
            neural_net.eval()

        post_process_segmentation = 'post_process_segmentations' in self.configs['general'] and self.configs['general']['post_process_segmentations']
        detect_fovea_and_recalculate_stats = 'detect_fovea_and_recalculate_stats' in self.configs['general'] and self.configs['general']['detect_fovea_and_recalculate_stats']

        def detect_fovea_and_recalculate_stats_for_acquisition(segmentation: pathlib.Path) -> None:
            '''
            Single-threaded function for re-making the Discovery stats, which takes in post-processed segmentations and recalculates statistics.
            Outputs, in a separate dir, a json file with the recomputed stats, and a visualisation of the EDTRS grid overlaid on top of the reconstructed thickness maps.
            '''

            # Find the index of the segmentation from within the path
            seg_idx = int(str(segmentation).split('_')[-2])

            output_dir = segmentation.parent / f'recomputed_statistics_{seg_idx:02}'
            output_dir.mkdir(exist_ok=True)

            all_seg_cube = [x for x in segmentation.rglob('*.npy') if 'all' in x.name][0] # Only a single 'all'-cube per segmentation
            all_seg = np.load(all_seg_cube)
            centre_slice, _, centre_of_fovea_in_slice = find_max_slice_presegmented(all_seg, parallel=False)

            with strong_suppress_stderr_only():
                cleaned_segmentations = get_clean_oct_segmentation(
                    location=segmentation.parent.parent,
                    segmentation_index=seg_idx,
                    filled_segmentation=True,
                    show_progress=False
                )

            thickness_maps, scaling_factor = rebuild_thickness_map(cleaned_segmentations, scaling_factor=None, show_progress=False)
            oct_info = get_oct_scale_information(location=segmentation.parent.parent)
            
            generate_thickness_map_visualisation(thickness_maps, oct_info, scaling_factor=scaling_factor, centre_point=(centre_slice, centre_of_fovea_in_slice), output_file_path=output_dir / 'thickness_maps.png')

            edtrs_stats = compute_quadrant_stats(layer_thicknesses=thickness_maps, oct_info=oct_info, scaling_factor=scaling_factor, centre_point=(centre_slice, centre_of_fovea_in_slice))
            with open(output_dir / 'edtrs_stats.json', 'w') as f:
                json.dump(edtrs_stats, f, indent=4)


        def predict_class_with_neural_net(model, wb) -> None:
            '''
            Iterate over a downloaded workbook, find fundus-labelled images to re-classify,
            and write their classification next to them, in .txt file.
            '''
            with torch.no_grad(): # Definitely don't want to train a model
                local_fundus_copies = tuple(wb.downloaded.parent.rglob('**/fundus/*.jpg'))
                logger.info(f'Found {len(local_fundus_copies)} images in download to classify.')
                for img in local_fundus_copies:
                    tensor_img = self.__read_and_format_image(img, resizer_transform, convert_img_to_tensor, torch)

                    net_pred = model(tensor_img).argmax(dim=1)
                    named_net_pred = index_to_cnn_modality(net_pred + 1)
                    with open(str(img.parent / img.stem) + '-classification.txt', 'w') as f:
                        f.write(named_net_pred.name)

        def post_process_segmentation_for_acquisition(aq: pathlib.Path) -> None:
            '''
            Single-threaded processing function for finding the segmentations in an image and cleaning / improving their quality.
            Outputs, in a separate dir, PNG conversions of the SVGs along with NPY cubes of the segmentations for easy downstream analysis.
            '''
            for segmentation_idx in [int(str(x).split('_')[-1]) for x in aq.glob('children/segmentation_*/') if 'cleaned' not in x.as_posix()]:
                output_dir = aq / 'children' / f'segmentation_{segmentation_idx:02}_cleaned'
                for f in output_dir.rglob('*'): # Clean up the directory before populating it
                    f.unlink(missing_ok=True)
                with strong_suppress_stderr_only():
                    seg_dict = get_clean_oct_segmentation(
                        location=aq,
                        segmentation_index=segmentation_idx,
                        filled_segmentation=False,
                        show_progress=False
                    )
                if not seg_dict: continue
                output_dir.mkdir(exist_ok=True)
                for key, value in seg_dict.items():
                    np.save((output_dir / key).as_posix() + '.npy', value) # Save to numpy-native .npy format
                for idx, slice in enumerate(seg_dict['all']): # 2D numpy array
                    pil_obj = Image.fromarray(slice)
                    pil_obj.save(output_dir / f'{idx:04}.png')

        for workbook in self.project.children:
            # Download the workbook
            logger.info(f'{repr(workbook)} | Downloading patients...')
            workbook.download(downloader=self.downloader, pbar=pbar)

            if use_taxonomy_cnn:
                predict_class_with_neural_net(model=neural_net, wb=workbook)

            # TODO: Update .downloaded etc.
            # MODIFY: The uploading phase is after everything is downloaded. Is this the best thing to do?
            if self.configs['general']['copy_filtered_workbook']:
                logger.info(f'{repr(workbook)} | Uploading to the target workbook...')
                target = self.configs['general']['copy_filtered_workbook']
                workbook.copy(target=target)

            if post_process_segmentation:
                acquisitions = [x.parent for x in workbook.downloaded.parent.rglob('oct')] # This folder exists for every study because was force-included in run.py
                logger.info(f"Post-processing segmentations for {len(acquisitions)} acquisitions.")
                deque(
                    tqdm.tqdm(
                        Parallel(n_jobs = cpu_count(), return_as='generator', verbose=0)(delayed(post_process_segmentation_for_acquisition)(aq) for aq in acquisitions),
                        # [post_process_segmentation_for_acquisition(aq) for aq in acquisitions],
                        total=len(acquisitions),
                        desc='Post-processing segmentations',
                        file=sys.stdout # Keeps printing active while supressing bad prints
                    ),
                maxlen=0) # Fastest way to consume an iterator in python
                logger.info(f"Finished post-processing segmentations.")
            
            if detect_fovea_and_recalculate_stats:
                acquisitions = [x.parent for x in workbook.downloaded.parent.rglob('oct')]
                cleaned_segmentations = [tuple((x / 'children').glob('segmentation_*_cleaned')) for x in acquisitions]
                cleaned_segmentations = tuple(chain(*cleaned_segmentations)) # Rectify
                logger.info(f"Detecting fovea and recalculating stats for {len(cleaned_segmentations)} segmentations.")
                deque(
                    tqdm.tqdm(
                        Parallel(n_jobs = cpu_count(), return_as='generator', verbose=0)(delayed(detect_fovea_and_recalculate_stats_for_acquisition)(aq) for aq in cleaned_segmentations),
                        # [detect_fovea_and_recalculate_stats_for_acquisition(aq) for aq in cleaned_segmentations],
                        total=len(cleaned_segmentations),
                        desc='Detecting fovea and recalculating stats',
                        file=sys.stdout # Keeps printing active while supressing bad prints
                    ),
                maxlen=0)
                logger.info(f"Finished detecting fovea and recalculating stats.")

        # Mark as done
        f = open(self.project.downloaded, 'w')  # MODIFY: Change it to .done
        f.close()
        pbar.close()
        for workbook in self.project.children:
            workbook.downloaded.unlink(missing_ok=True)
        msg = f'You can find the cohort in {self.project.folder.as_uri()}.'
        logger.info(msg)
        print(msg)

class Uploader:
    """
    This class deals with uploading ....
    ...
    # TODO: ...
    """

    def __init__(self, configs: dict, instances: list[str], project: str, workbooks: list[str]):
        # Store the arguments
        self.instances = instances
        self.project = project
        self.workbooks = workbooks
        # Store the configs
        self.configs: dict = self._check_configs(configs)

        # Instantiate attributes
        now = datetime.now(pytz.timezone('Europe/Zurich')).strftime('%Y%m%d.%H%M%S.%f')
        #: Directory for keeping the copied files
        self.CACHE_DIR = pathlib.Path(Parser.settings['general']['cache_large']) / 'upload' / now
        self.CACHE_DIR.mkdir(parents=True)
        #: Flag for signaling successful upload
        self.success = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Delete the caches if the process is done successfully
        if self.success:
            for file in self.CACHE_DIR.iterdir():
                file.unlink(missing_ok=True)
            self.CACHE_DIR.rmdir()
            msg = f'The copied files are removed.'
            print(msg)
            logger.info(msg)


    def _catch_errors(func: Callable[[], None]) -> Callable[[], None]:
        """Decorator for handling errors in the uploader."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Execute the function
            try:
                return func(self, *args, **kwargs)
            except InvalidConfigsException as e:
                logger.warning(e.message)
                print(f'PROBLEM in the configs file: {e.message}')
                print(f'You can change this configuration by changing '
                    f'"{e.field}" in {(Parser.args.configs_dir / Parser.args.configs).absolute().as_posix()}.')
                exit()
            except Exception as e:
                raise e

        return wrapper

    @ _catch_errors
    def _check_configs(self, configs: dict) -> dict:
        """
        Gets the configuration dictionary and checks the sanity of the configurations.
        Some fields will be modified in the returned configurations.

        Args:
            configs: The original configuration dictionary.

        Raises:
            InvalidConfigsException: If there is a problem with one of the configurations.

        Returns:
            dict: The modified configuration dictionary.
        """

        # Check the configs purpose
        if configs['purpose'] != 'upload':
            raise InvalidConfigsException(
                    message=f'The provided configs file ({Parser.args.configs_dir / Parser.args.configs}) does not have proper format.',
                    field='purpose',
            )


        return configs

    @_catch_errors
    def get_upload_files(
        self, pids: list[Union[str, int]] = None, pidsfile: Union[pathlib.Path, str] = None,
        updatemetadata: bool = False, copyfiles: bool = False
    ) -> list[DicomFile]:
        """
        Prepares the files related to a set of patients and returns the list of the prepared files.

        Args:
            pids: list of patient identifiers. Defaults to ``None``.
            pidsfile: Path to a json file containing a list of patient identifiers. Defaults to ``None``.
            updatemetadata: If ``True``, the metadata of the scans on the image pools are updated before the related files are fetched. Defaults to `False`.
            copyfiles: If ``True``, the files are copied from the image pools to a safe directory. Defaults to `False`.

        Returns:
            A list of the prepared files.
        """

        # Get the patients
        if not pids:
            pids = []
        if pidsfile:
            pids.extend([str(pid) for pid in read_json(pidsfile)])
        patients = [PatientManager(pid=pid) for pid in set(pids)]

        # Find the files on image pools and copy them
        heyexfiles = HeyexMetadataManager.explore(patients=patients, update=updatemetadata)
        if copyfiles and heyexfiles:
            copied_mb = 0
            desc = 'Copying files'
            pbar = tqdm.tqdm(
                total=len(heyexfiles),
                desc=(desc + ' (0 MiB)').ljust(Parser.settings['progress_bar']['description']),
                ncols=get_terminal_size().columns,
                leave=None,
            )
            for idx, file in enumerate(heyexfiles):
                path: pathlib.Path = HeyexMetadataManager.DIR / file['path']
                try:
                    shutil.copyfile(src=path, dst=(self.CACHE_DIR / path.name))
                except Exception as e:
                    msg = f'Failed to copy {path.name}'
                    logger.warning(f'{msg}: {path.as_posix()}')
                    logger.exception(f'{msg}: {e}')
                    continue
                copied_mb += path.stat().st_size / (1024 ** 2)
                pbar.update()
                if ((idx + 1) % 100) == 0 or idx == (len(heyexfiles) - 1):
                    if copied_mb < 1024:
                        pbar.desc = (desc + f' ({copied_mb:.1f} MiB)').ljust(Parser.settings['progress_bar']['description'])
                    else:
                        copied_gb = copied_mb / 1024
                        pbar.desc = (desc + f' ({copied_gb:.1f} GiB)').ljust(Parser.settings['progress_bar']['description'])
                    pbar.refresh()
            pbar.close()
            msg = f'{len(heyexfiles)} files are copied to {self.CACHE_DIR.as_posix()}.'
            print(msg)
            logger.info(msg)
            uploadfiles = [
                DicomFile(path=file, mode='local')
                for file in self.CACHE_DIR.iterdir()
                if (file.is_file() and file.suffix == '.dcm')
            ]
        else:
            uploadfiles = [
                DicomFile(path=(HeyexMetadataManager.DIR / file['path']), mode='local')
                for file in heyexfiles
            ]

        return uploadfiles

    def filter_files(self, uploadfiles: list[DicomFile]) -> list[DicomFile]:
        """
        Filter a given set of files based on the configurations.

        Args:
            uploadfiles: Files to be filtered.

        Returns:
            The filtered files.
        """

        msg = f'Filtering the detected files..'
        logger.info(msg)
        print(msg)
        filtered = []
        for file in uploadfiles:
            # NOTE: Takes too long, we can instead add a column to the metadata
            # TODO: Need to have MediaStorageSOPClassUID in metadata to do this
            # if file.ispdf():
            #     continue
            filtered.append(file)
        msg = f'{len(filtered)}/{len(uploadfiles)} files passed the filters.'
        logger.info(msg)
        print(msg)

        return filtered

    @_catch_errors
    def upload(self, uploadfiles: list[DicomFile], better_anonymisation: bool = False) -> None:
        """
        Upload a given set of files.

        Args:
            uploadfiles: Files to be uploaded.
        """

        # Upload the given files
        for workbook in self.workbooks:
            with DiscoveryManager(
                instances=self.instances,
                projectname=self.project,
                workbookname=workbook,
                permission='write',
            ) as manager:
                manager.upload(
                    files=uploadfiles,
                    anonymize=[
                        instance for instance in self.instances
                        if Parser.settings['api'][instance]['anonymize']
                    ],
                    better_anonymisation=better_anonymisation
                )

        # Flag as success
        self.success = True

class InvalidConfigsException(Exception):
    """
    Exception for handling problems in the configs file.

    Args:
        message: The message of the exception. Defaults to ``None``.
        field: The name of the field that creates the problem. Defaults to ``None``.
    """

    def __init__(self, message: str = None, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)
