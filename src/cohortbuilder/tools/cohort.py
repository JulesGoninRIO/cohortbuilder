"""
This module includes the objects for browsing and manipulating the downloaded
(built) cohorts.
"""

import json
import pathlib
from typing import Callable, Union

import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
from pydicom import dcmread, Dataset

from src.cohortbuilder.tools.definitions import Biomarker, Region, RetinalLayer
from src.cohortbuilder.tools.parameters import calculate_cvi, calculate_tvs, detect_fovea, calculate_cvi_line, calculate_disruption, calculate_pachy_area, get_reference_coordinates, get_position_tag
from src.cohortbuilder.utils.helpers import read_img, read_json, list2str, is_notebook


class Cohort:
    """
    This class creates a cohort instance from a downloaded cohort.

    Args:
        path: The root directory to the downloaded cohort.

    Examples:
        >>> from src.cohortbuilder.tools.cohort import Cohort
        >>>
        >>> # Browse through the cohort
        >>> AOSLO = Cohort('../cohorts/AOSLO/')
        >>> AOSLO.browse(fovea=False, cvi=False, tvs=False, file='../cohorts/AOSLO/output.json', indent=None, prec='.4e')

        >>> from src.cohortbuilder.tools.cohort import Cohort
        >>>
        >>> # Load the output of a previous browse
        >>> AOSLO = Cohort('../cohorts/AOSLO/')
        >>> AOSLO.load(file='../cohorts/AOSLO/output.json')
        >>>
        >>> # Define the builder function
        >>> def builder(dataset):
        >>>     if 'OCT_CUBE' in dataset['info']['layerVariants']:
        >>>         row = [
        >>>             dataset['info']['patient']['name'],
        >>>             dataset['info']['patient']['birthdate'],
        >>>             dataset['info']['laterality'],
        >>>         ]
        >>>
        >>>     else:
        >>>         row = None
        >>>
        >>>     return row
        >>>
        >>> # Set the names of the columns
        >>> columns = [
        >>>     'Patient ID',
        >>>     'Study Date',
        >>>     'Laterality',
        >>> ]
        >>>
        >>> # Create a dataframe
        >>> df = AOSLO.export(builder=builder, columns=columns)
    """

    def __init__(self, path: Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        assert path.exists()

        #: The root directory to the downloaded cohort
        self.dir: pathlib.Path = path
        #: Dictionary of the patients in the cohort
        self.patients: dict = None

    def browse(self, fovea: bool = False, cvi_cube: bool = False, cvi_line: bool = False, tvs: bool = False, pachy: bool = False,
               dscore: bool = False, dicom_retina_ONH_tag: bool = False, dicom_coord: bool = False,
               file: Union[str, pathlib.Path] = None, checkpoint: bool = True,
               indent: int = None, prec: str = ''):
        """
        Loops over the directory and stores data of each dataset in the patients attribute.

        Args:
            fovea: Wether or not to calculate location of fovea. Defaults to ``False``.
            cvi_cube: Wether or not to calculate cvi on cube. Defaults to False.
            cvi_line: Wether or not to calculate cvi on line. Defaults to False.
            tvs: Wether or not to calculate thickness vectors. Defaults to False.
            pachy: Wether or not to calculate pachy area. Defaults to False.
            dscore: Wether or not to calculate disruption score. Defaults to False.
            dicom_retina_ONH_tag: Wether or not to extract the dicom Retina ONH tag. Defaults to False.
            dicom_coord: Wether or not to extract the dicom coordinates. Defaults to False.
            file: Path to save the result in a json file.
            checkpoint: create a checkpoint file after each patient. Defaults to True
            indent: Number of spaces to use for indentation. Defaults to ``None``.
            prec: The format of the floats in the downloaded jsons. Defaults to empty string.
        """

        # Check the inputs
        assert all([
            isinstance(cvi_cube, bool),
            isinstance(cvi_line, bool),
            isinstance(tvs, bool),
            isinstance(pachy, bool),
            isinstance(dscore, bool),
            isinstance(dicom_retina_ONH_tag, bool),
            isinstance(dicom_coord, bool)
        ])
        if file:
            assert isinstance(file, (str, pathlib.Path))
            file = pathlib.Path(file)
            assert file.suffix == '.json'
            assert file.parent.exists()
            if file.exists():
                print(f'File already exists. Loading "{file}" instead of browsing.')
                self.load(file)
                return

        if checkpoint:
            file_checkpoint = file.parent / (file.stem + "-checkpoint" + file.suffix)
            if file_checkpoint.exists():
                print(f'Checkpoint File exists. Loading "{file_checkpoint}" and browse on top of it.')
                self.load(file_checkpoint)

        # Set the progress bar settings
        desc_width = 50
        ncols = 120

        # Loop over the cohort entities
        patient_folders = [patient for patient in self.dir.iterdir() if patient.is_dir()]
        patient_pbar = tqdm.tqdm(
                    total=len(patient_folders),
                    desc=f'Cohort: {self.dir.name}'.ljust(desc_width),
                    ncols=ncols,
                    leave=None,
                )

        if not isinstance(self.patients, dict):
            self.patients = dict()

        for patient in patient_folders:
            if patient.name in self.patients.keys():
                if patient_pbar: patient_pbar.update(1)
                continue
            studies = dict()
            study_folders = [study for study in patient.iterdir() if study.is_dir()]
            study_pbar = tqdm.tqdm(
                        total=len(study_folders),
                        desc=f'Patient: {patient.name}'.ljust(desc_width),
                        ncols=ncols,
                        leave=None,
                    ) if not is_notebook() else None
            for study in study_folders:
                datasets = list()
                dataset_folders = [dataset for dataset in study.iterdir() if dataset.is_dir()]
                dataset_pbar = tqdm.tqdm(
                            total=len(dataset_folders),
                            desc=f'Study: {study.name.split(" ")[1]}'.ljust(desc_width),
                            ncols=ncols,
                            leave=None,
                        ) if not is_notebook() else None
                for dataset in dataset_folders:
                    # Read dataset info
                    info = read_json(dataset / 'info.json')

                    # Read metadata info
                    _uuid, _pid, _date, _type, _laterality, _sex, _age = get_meta_info(info)

                    # Define children files
                    # TODO: Handle multiple files
                    oct_path = dataset / 'oct' / 'volume'  # TODO: Add other OCTs
                    fds_path = dataset / 'fundus'
                    seg_path = dataset / 'children' / 'segmentation_01'
                    bms_file = dataset / 'children' / 'biomarkers_01.json'
                    vls_file = dataset / 'children' / 'volume_01.json'
                    ths_file = dataset / 'children' / 'thickness_01.json'

                    # Read OCT
                    if oct_path.exists():
                        oct_info = read_json(oct_path / 'info.json')
                    else:
                        oct_path = None
                        oct_info = None

                    # Check the range of the OCT
                    # NOTE: The goal is to detect the wrong volume and thicknesses and to mask them
                    # NOTE: This is only true for OCT cubes
                    # NOTE: Currently, this method only deals with OCT cubes so we are safe to do it
                    # TODO: Distinguish between other types of OCTs
                    central_zone_valid = True
                    pericentral_zone_valid = True
                    peripheral_zone_valid = True
                    if oct_info:
                        if min(oct_info['range'][0], oct_info['range'][3]) < 1.:
                            central_zone_valid = False
                            pericentral_zone_valid = False
                            peripheral_zone_valid = False
                        elif min(oct_info['range'][0], oct_info['range'][3]) < 3.:
                            pericentral_zone_valid = False
                            peripheral_zone_valid = False
                        elif min(oct_info['range'][0], oct_info['range'][3]) < 6.:
                            peripheral_zone_valid = False

                    # Read fundus
                    if fds_path.exists():
                        fds_info = read_json(fds_path / 'info.json')
                    else:
                        fds_path = None
                        fds_info = None

                    # Read segmentation biomarkers
                    if seg_path.exists():
                        seg_biom = read_json(seg_path / 'biomarkers.json')\
                            if (seg_path / 'biomarkers.json').exists() else None
                    else:
                        seg_path = None
                        seg_biom = None

                    # Read biomarkers
                    if bms_file.exists():
                        # CHECK: Why json files are not valid sometimes?
                        with open(bms_file, 'r') as f:
                            biomarkers = f.read()
                            try:
                                biomarkers = json.loads(biomarkers)
                            except:
                                for i in range(1, 52):
                                    try:
                                        biomarkers = json.loads(biomarkers[:-i])
                                        break
                                    except:
                                        pass
                        biomarkers = biomarkers[-9:]  # CHECK: Why sometimes more than 9?
                        biomarkers = [[float(format(val, prec)) for val in bm] for bm in biomarkers]
                        biomarkers = {name: val for name, val in\
                            zip(Biomarker.get_names(), biomarkers)}
                    else:
                        biomarkers = None

                    # Read volumes
                    if vls_file.exists():
                        try:
                            volumes = read_json(vls_file)
                            volumes = np.reshape(volumes, (11, 10))
                            volumes = {name: {reg: float(format(val, prec)) for reg, val in zip(Region.get_names(), volume)}\
                                for name, volume in zip(RetinalLayer.get_names(plus=True), volumes)}
                            # Replace invalid volumes (zones out of the range of the scans)
                            invalid_zones_names = []
                            if not central_zone_valid:
                                invalid_zones_names.extend([region.name for region in Region.central()])
                            if not pericentral_zone_valid:
                                invalid_zones_names.extend([region.name for region in Region.pericentral()])
                            if not peripheral_zone_valid:
                                invalid_zones_names.extend([region.name for region in Region.peripheral()])
                            for layer in volumes.values():
                                for region_name in layer.keys():
                                    if region_name in invalid_zones_names:
                                        layer[region_name] = None
                        except:
                            volumes = None
                    else:
                        volumes = None

                    # Read thicknesses
                    if ths_file.exists():
                        try:
                            thicknesses = read_json(ths_file)
                            thicknesses = np.reshape(thicknesses, (11, 10))
                            thicknesses = {name: {reg: float(format(val, prec)) for reg, val in zip(Region.get_names(), thickness)}\
                                for name, thickness in zip(RetinalLayer.get_names(plus=True), thicknesses)}
                            # Replace invalid thicknesses (zones out of the range of the scans)
                            invalid_zones_names = []
                            if not central_zone_valid:
                                invalid_zones_names.extend([region.name for region in Region.central()])
                            if not pericentral_zone_valid:
                                invalid_zones_names.extend([region.name for region in Region.pericentral()])
                            if not peripheral_zone_valid:
                                invalid_zones_names.extend([region.name for region in Region.peripheral()])
                            for layer in thicknesses.values():
                                for region_name in layer.keys():
                                    if region_name in invalid_zones_names:
                                        layer[region_name] = None
                        except:
                            thicknesses = None
                    else:
                        thicknesses = None

                    # Read dicom informations
                    if dicom_coord or dicom_retina_ONH_tag:
                        dicom_path = dataset / "parent.dcm"
                        if dicom_path.exists():
                            dicom_info = dict()
                            dicom: Dataset = dcmread(dicom_path)
                            if dicom_coord:
                                dicom_info["coordinates"] = get_reference_coordinates(dicom, _type)
                                ### there is miss labeled OCT circle so change the label here
                                if (_type == "line") and (len(dicom_info["coordinates"]) != 4):
                                    _type = "circle"
                                    info["layerVariants"] = ["OCT_GENERAL", "OCT_CIRCLE"]
                            if dicom_retina_ONH_tag:
                                # retina/ONH tag
                                dicom_info["retina_ONH_tag"] = get_position_tag(dicom)
                        else:
                            dicom_info = None
                    else:
                        dicom_info = None


                    # Detect the presence of fovea and its position
                    if fovea and oct_path and seg_path:
                        foveas = dict()
                        for seg_file in seg_path.glob('*.svg'):
                            seg = read_img(seg_file)
                            foveas[seg_file.stem] = detect_fovea(seg, info=oct_info)
                    else:
                        foveas = None

                    # Calculate CVI on cube
                    if cvi_cube and oct_path and seg_path and (_type=="cube"):
                        cvis = dict()
                        for oct_file, seg_file in zip(oct_path.glob('*.jpg'), seg_path.glob('*.svg')):
                            oct = read_img(oct_file)
                            seg = read_img(seg_file)
                            cvis[oct_file.stem] = calculate_cvi(
                                oct=oct,
                                seg=seg,
                                fovea=foveas[seg_file.stem] if foveas else None,
                                prec=prec,
                            )
                    else:
                        cvis = None

                    # Calculate the thickness vector
                    if tvs and oct_path and seg_path:
                        tvss = dict()
                        for seg_file in seg_path.glob('*.svg'):
                            seg = read_img(seg_file)
                            tvss[seg_file.stem] = calculate_tvs(
                                seg=seg,
                                info=oct_info,
                                fovea=foveas[seg_file.stem] if foveas else None,
                                prec=prec
                                )
                    else:
                        tvss = None

                    # Caluclate cvi on OCT line
                    if cvi_line and oct_path and seg_path and (_type=="line"):
                        try:
                            cvis = dict()
                            for oct_file, seg_file in zip(oct_path.glob('*.jpg'), seg_path.glob('*.svg')):
                                oct = read_img(oct_file)
                                seg = read_img(seg_file)
                                cvis[seg_file.stem] = calculate_cvi_line(
                                    seg=seg,
                                    oct=oct
                                    )
                        except:
                            cvis = None
                    else:
                        cvis = None

                    # Caluclate pachy area on OCT line
                    if pachy and oct_path and seg_path and (_type=="line"):
                        try:
                            pachys = dict()
                            for oct_file, seg_file in zip(oct_path.glob('*.jpg'), seg_path.glob('*.svg')):
                                oct = read_img(oct_file)
                                seg = read_img(seg_file)
                                pachys[seg_file.stem] = calculate_pachy_area(
                                    oct=oct,
                                    seg=seg,
                                    spacings = oct_info['spacing']
                                    )
                        except:
                            pachys = None
                    else:
                        pachys = None

                    # Caluclate disruption score on OCT line
                    if dscore and oct_path and seg_path and (_type=="line"):
                        try:
                            dscores = dict()
                            for oct_file, seg_file in zip(oct_path.glob('*.jpg'), seg_path.glob('*.svg')):
                                oct = read_img(oct_file)
                                seg = read_img(seg_file)
                                dscores[seg_file.stem] = calculate_disruption(
                                    oct=oct,
                                    seg=seg,
                                    )
                        except:
                            dscores = None
                    else:
                        dscores = None

                    # Store the data
                    datasets.append({
                        'folder': str(dataset),
                        'info': info,
                        'oct': {'folder': str(oct_path), 'info': oct_info} if oct_path else None,
                        'fundus': {'folder': str(fds_path), 'info': fds_info} if fds_path else None,
                        'segmentation': {'folder': str(seg_path), 'biomarkers': seg_biom} if seg_path else None,
                        'biomarkers': biomarkers,
                        'volumes': volumes,
                        'thicknesses': thicknesses,
                        'foveas': foveas,
                        'cvis': cvis,
                        'tvss': tvss,
                        'pachys': pachys,
                        'dscores': dscores,
                        'dicom_info': dicom_info,
                    })
                    if dataset_pbar: dataset_pbar.update(1)

                if dataset_pbar: dataset_pbar.close()
                studies[study.name.split(' ')[1]] = datasets
                if study_pbar: study_pbar.update(1)

            if study_pbar: study_pbar.close()
            self.patients[patient.name] = studies

            if checkpoint:
                with open(file.parent / (file.stem + "-checkpoint" + file.suffix), 'w') as f:
                    json.dump(self.patients, f, indent=indent)

            if patient_pbar: patient_pbar.update(1)
        if patient_pbar: patient_pbar.close()

        # Store the browsed patients
        if file:
            with open(file, 'w') as f:
                json.dump(self.patients, f, indent=indent)
                print(f'Patients saved in {file}.')

    def load(self, file: Union[str, pathlib.Path]) -> None:
        """
        Loads the patients of the cohort from its file.

        Args:
            file: The file containing all the information of the
                patients in a cohort.
        """

        try:
            with open(file, 'r') as f:
                self.patients = json.load(f)
        except json.JSONDecodeError:
            print(f'Patients file ({file}) is corrupted.')

    # TODO: Add a progress bar
    def export(self, builder: Callable[[dict], list], columns: list = None) -> pd.DataFrame:
        """
        Loops over the datasets and extracts the values returned by a builder in a table.

        Args:
            builder: A function that gets as input a dataset
                dictionary and returns a list of desired values of that dictionary.
                It should return ``None`` if the input dataset is not wanted.
            columns: List of column names. Defaults to ``None`` .

        Raises:
            Exception: If the `self.browse <src.cohortbuilder.tools.cohort.Cohort.browse>`
                method has not been called before.

        Returns:
            Table of the results for each dataset.
        """

        # Check patients
        if not self.patients:
            raise Exception('Patients is empty. Use the browse method first.')

        # Initialize the rows
        rows = list()

        # Loop over datasets in data
        for studies in tqdm.tqdm(self.patients.values()):
            for datasets in studies.values():
                for dataset in datasets:
                    row = builder(dataset)
                    if row:
                        rows.append(row)

        # Create the results Dataframe
        results = pd.DataFrame(
            data=rows,
            columns=columns,
        )

        return results

class CohortDataset:
    """
    This class creates a dataset instance from a downloaded dataset.

    Args:
        dataset: A downloaded dataset in a cohort.
    """

    def __init__(self, dataset: dict):
        for key, val in dataset.items():
            self.__dict__[key] = val

    def plot_vectors(self, file: Union[str, pathlib.Path] = None, idx=0) -> Union[None, plt.Figure]:
        """Plots the vectors of the dataset and stores them in a file"""

        # Get the vectors
        name = str(idx).zfill(4)
        if self.tvss and name in self.tvss:
            tvs = self.tvss[name]
        else:
            tvs = None
        if self.cvis and name in self.cvis:
            cvi = self.cvis[name]
        else:
            cvi = None


        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12, 6),
        )
        settings = {
            # "xlim": (x[0] + 1, x[-1]),
        }

        # Set the title
        title_items = list()
        if self.info["laterality"] in {'R', 'L'}:
            title_items.append('OD' if self.info["laterality"] == 'R' else 'OS')
        if 'OCT_LINE' in self.info['layerVariants'] and self.info['angles']:
            angle = self.info['angles'][0]
            if angle == 0:
                title_items.append('Vertical')
            elif angle == 90:
                title_items.append('Horizontal')
            else:
                title_items.append(angle)
        fig.suptitle(list2str(title_items, delimeter=' | '))

        # Plot tvs
        ax = axes[0]
        self._plot_tvs(tvs, ax)
        if tvs: ax.legend(loc='upper right')
        # Plot cvi
        ax = axes[1]
        self._plot_cvi(cvi, ax)
        if cvi: ax.legend(loc='upper right')

        for ax in axes:
            ax.set(**settings)
            ax.grid()
            # ax.legend(loc='upper right')

        if file:
            fig.savefig(fname=file)
            plt.close(fig)
        else:
            return fig

    def _plot_tvs(self, tvs, ax) -> None:
        ax.set(
            xlabel='x (mm)',
            title='Layer thicknesses (mm)'
        )
        if not tvs:
            return
        for layer in tvs.keys():
            if not tvs[layer]:
                continue
            vector = tvs[layer]['vector']
            center = tvs[layer]['center']
            vector = np.array(vector)
            x = np.arange(len(vector), dtype=float)
            if center:
                x -= center
            x *= self.oct['info']['spacing'][3]
            ax.scatter(x, vector, s=2, label=layer)
        if center:
            ax.axvline(x=0, linestyle='--', color='black', label='Fovea')

    def _plot_cvi(self, cvi, ax) -> None:
        ax.set(
            xlabel='x (mm)',
            title='CVI'
        )
        if not cvi:
            return
        vector = cvi['vector']
        center = cvi['center']
        cvi_avg = cvi['average']
        vector = np.array(vector, dtype=float)
        x = np.arange(len(vector), dtype=float)
        if center:
            x -= center
        x *= self.oct['info']['spacing'][3]
        ax.scatter(x, vector, s=2, label='CVI')
        ax.vlines(x, ymin=0, ymax=vector, linewidth=.1, label='_nolegend_')
        ax.axhline(y=cvi_avg, linestyle=':', color='black', label='Average CVI')
        if center:
            ax.axvline(x=0, linestyle='--', color='black', label='Fovea')


def get_meta_info(info):
    """ Extract meta information of the scan.
    """
    _type = None
    if 'OCT_LINE' in info['layerVariants']:
        _type = 'line'
    elif 'OCT_CUBE' in info['layerVariants']:
        _type = 'cube'
    elif 'OCT_CIRCLE' in info['layerVariants']:
        _type = 'circle'

    _date = info['study']['studyDatetime'][:10]
    _laterality = info['laterality']
    _pid = info['patient']['patientId'] 
    _uuid = info['uuid']
    _sex = info['patient']['sex']
    
    if info['patient']['birthdate'] is None:
        birth = datetime.datetime.strptime('1900-01-01', '%Y-%m-%d').date()
    else:
        birth = datetime.datetime.strptime(info['patient']['birthdate'], '%Y-%m-%d').date()
    date = datetime.datetime.strptime(_date, '%Y-%m-%d').date()
    _age = date.year - birth.year - ((date.month, date.day) < (birth.month, birth.day))
    return _uuid, _pid, _date, _type, _laterality, _sex, _age