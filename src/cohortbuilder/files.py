"""
This module includes classes for managing source files of different types.
"""

from __future__ import annotations

import io
import pathlib
import shutil
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum, unique
from functools import wraps
from time import sleep
from typing import TYPE_CHECKING, Callable, Literal, TypeVar, Union, Any
from enum import Enum, EnumMeta
from abc import ABCMeta

import numpy as np
import pydicom
from loguru import logger
from oct_converter.dicom import create_dicom_from_oct
from PIL import Image
from pydicom.encaps import encapsulate, encapsulate_extended
from pydicom.filereader import dcmread
from pydicom.valuerep import DSfloat
from pydicom.uid import (JPEG2000Lossless, JPEGBaseline, JPEGExtended,
                         JPEGLosslessSV1, OphthalmicTomographyImageStorage,
                         RLELossless, generate_uid)
from typing_extensions import ParamSpec

from src.cohortbuilder.definitions import UploadPipelineFileStatus
from src.cohortbuilder.discovery.definitions import DiscoveryFileStatus
from src.cohortbuilder.discovery.exceptions import RequestMaxAttemptsReached
from src.cohortbuilder.discovery.file import DiscoveryFile, DiscoveryTaskStatus
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.tools.list_and_delete_pending import get_discovery_file_uuids_by_name, delete_acquisition_and_file, delete_discovery_file
from src.cohortbuilder.utils.helpers import create_dicom_uid_from_file

if TYPE_CHECKING:
    from src.cohortbuilder.discovery.entities import Workbook
    from src.cohortbuilder.managers import Client
    from pydicom.dataset import FileDataset, Dataset
    from pydicom.dataelem import DataElement

T = TypeVar('T')
P = ParamSpec('P')


class File:
    """
    Class for managing a file that exists on the remote
    servers and has to be processed or uploaded to Discovery.

    Args:
        path: The path of the file.
        mode: The mode (local or remote) of the file.
    """

    def __init__(self, path: Union[None, str, pathlib.Path], mode: Literal['local', 'remote'] = 'local'):
        assert mode in {'local', 'remote'}
        #: Mode of the original file, remote or local
        self.mode: Literal['local', 'remote'] = mode
        #: Original path
        self.path: Union[None, str, pathlib.Path] = pathlib.Path(path)
        #: Path of the local copy of the file
        self.copied: pathlib.Path = None
        #: File name
        self.name: str = self.path.name
        #: File details on Discovery instances
        self.discovery: dict[str, DiscoveryFile] = {}
        #: Status of the file in the pipeline
        self.status: UploadPipelineFileStatus = UploadPipelineFileStatus.DETECTED

    def _local(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator for methods that are only viable for local files."""

        @wraps(func)
        def wrapper(self: File, *args, **kwargs) -> T:
            try:
                assert self.mode == 'local' and self.path.exists()
            except:
                logger.debug(f'{repr(self)} | The file is not local or does not exist.')
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def _copied(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator for methods that are only viable when a local copy of the file is available."""

        @wraps(func)
        def wrapper(self: File, *args, **kwargs) -> T:
            try:
                assert self.copied and self.copied.exists()
            except:
                logger.debug(f'{repr(self)} | Local copy of the file is not available.')
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def exists(self, client: Client = None) -> bool:
        """
        Checks if the local or remote path of the file exists.

        Args:
            client: SSH connection manager of the server.
              Must be passed if the files is remote.

        Returns:
            ``True`` if the file exists.
        """

        if self.mode == 'local':
            return self.path.exists()
        elif self.mode == 'remote' and client:
            stdout, _ = client.run(command=f'test -f {self.path.as_posix()}')
            if stdout.channel.recv_exit_status() == 0:
                return True
            else:
                return False
        else:
            raise Exception

    @_local
    def copy(self, target: Union[str, pathlib.Path]) -> None:
        """
        Creates a local copy of a local file and stores its path.

        Arguments:
            target: The target path of the copy.
        """

        target = pathlib.Path(target)
        assert target.suffix
        shutil.copyfile(src=self.path, dst=target)
        self.copied = target

    # UNUSED
    def get(self, client: Client, target: Union[str, pathlib.Path]) -> None:
        """Fetches a remote file from a given server and stores its path as a local copy."""

        target = pathlib.Path(target)
        assert self.mode == 'remote' and client.isalive() and self.exists(client=client)
        assert target.parent.exists()

        client.sftp.get(remotepath=self.path.as_posix(), localpath=target.as_posix())
        self.copied = target

    @_copied
    def upload(self, workbook: Workbook) -> None:
        """
        Uploads the local copy of a file to a given workbook in a Discovery instance.
        If the file is uploaded in Discovery before, Discovery returns the
        UUID of the previous file and does not duplicate the file.

        Args:
            workbook: Discovery workbook to upload the file to.
        """

        # Skip if the file has been uploaded successfully before
        if workbook.discovery.instance in self.discovery:
            return

        # Try uploading the file in a loop with maximum attempts
        MAX_ATTEMPTS = Parser.settings['general']['upload_max_attempts'] if Parser.settings else 3
        for attempt in range(MAX_ATTEMPTS):
            # Upload the file and handle exceptions
            try:
                response = workbook.discovery.upload(file=self.copied, uuid=workbook.uuid)
            except RequestMaxAttemptsReached as e:
                msg = f'Attempt {attempt+1} failed: {type(e).__name__}'
                sleep(20)

            # Check for errors in the response
            if 'errors' in response:
                # Determine as pending in a particular scenario
                if 'File already exists and is being processed' in response['errors'][0]['message']:

                    msg = 'Temporarily failed because the file has been uploaded before and is processing / stuck processing.'
                    self._log_upload(workbook=workbook, message=msg)

                    uuids_of_files_with_same_name = get_discovery_file_uuids_by_name(discovery_instance=workbook.discovery, file_name=self.name)
                    files_with_same_name = [DiscoveryFile(discovery=workbook.discovery, uuid=x) for x in uuids_of_files_with_same_name]

                    # The following is the same logic as in src.cohortbuilder.managers.UploadManager, but simplified
                    for f in files_with_same_name:
                        f.fetch(acquisitions=True) # Get status of all files with same name
                        logger.debug(f'Fetched {f.acquisitions} as acquisitions for {f} (found by name) in File.Upload')
                    
                        if not f.acquisitions:
                            sleep(60) # We haven't left Discovery time to do stuff. Wait for a minute, then try again.
                            f.fetch(acquisitions=False)
                            if f.status == DiscoveryFileStatus.PARSING: # File is stuck as parsing. Delete and reupload.
                                self._log_upload(workbook=workbook, message='File is stuck in parsing state. Removing and reuploading.')
                                delete_discovery_file(discovery_instance=workbook.discovery, wb=workbook, discovery_file=f)
                            continue
                        for aq in f.acquisitions:
                            task_status = {k: v[0]['status'] for k, v in aq.tasks.items()}
                            logger.debug(task_status)
                            pending_status = [x == DiscoveryTaskStatus.PENDING for x in task_status.values()]
                            if any(pending_status): # Delete any stuck pending files and acquisitions
                                delete_acquisition_and_file(discovery_instance=workbook.discovery, wb=workbook, aq=aq)

                    msg = f'Finished clearing pending files with name {self.name} from Discovery'
                    self._log_upload(workbook=workbook, message=msg)

                # Log the error messages if its unknown and try again
                else:
                    msg = f'Attempt {attempt+1} failed. Encountered errors: {[error["message"] for error in response["errors"]]}'
                    self._log_upload(workbook=workbook, message=msg)

                sleep(15) # Give a good amount of time for the file to be properly deleted / cleared if necessary
                continue # Proceed to retry

            # Check the status if there are no errors
            else:
                uuid = response['data']['upload']['uuid']
                status = DiscoveryFileStatus[response['data']['upload']['status']]
                # Finish as successful
                if status.isuploaded:
                    msg = 'Successful.'
                    break
                # Log and try again
                else:
                    msg = f'Failed: {status.name}'
                    self._log_upload(workbook=workbook, message=msg)
                    continue

        # Log and return if MAX_ATTEMPTS is reached
        else:
            msg = 'Failed. Reached maximum tries for uploading the file.'
            self._log_upload(workbook=workbook, message=msg)
            return

        # Register the uploaded discovery file
        dfile = DiscoveryFile(discovery=workbook.discovery, uuid=uuid)
        dfile.status = status
        self.discovery[workbook.discovery.instance] = dfile
        self._log_upload(workbook=workbook, message=msg)

    @_copied
    def remove(self) -> None:
        """Removes the local copy of the file if it exists."""

        self.copied.unlink()
        self.copied = None

    @property
    def cache(self) -> dict:
        """The jason-serializable summary of the file to be stored."""

        cache = {
            instance: {
                'uuid': dfile.uuid,
                'status': dfile.status.name,
                'successful': dfile.issuccessful,
            }
            for instance, dfile in self.discovery.items()
        }
        cache['heyex'] = self.path.relative_to(Parser.settings['heyex']['root']).as_posix()

        return cache

    def log_status_changed(self, message: str = None) -> None:
        """Logs the status change of the file with a message."""

        logger.trace(f'{repr(self)} | Status changed to {self.status.name}.')
        if message:
            logger.trace(f'{repr(self)} | Message: {message}')

    def _log_upload(self, workbook: Workbook, message: str) -> None:
        """Logs the upload result of the file."""

        logger.trace(f'{repr(self)} | Upload to "{repr(workbook)}": {message}')

    def __str__(self) -> str:
        return self.path.stem

    def __repr__(self) -> str:
        return self.path.stem

class DicomTagEnumMeta(ABCMeta, EnumMeta):
    """Metaclass for abstract DicomTagEnum classes."""

    def __new__(mcls, *args, **kw):
        cls = super().__new__(mcls, *args, **kw)
        # Update member dictionaries
        _value2member_map_ = {
            mem.id if isinstance(val, EnumMeta) else val: mem
            for val, mem in cls._value2member_map_.items()
        }
        cls._value2member_map_ = _value2member_map_

        return cls

class DicomTagEnum(Enum, metaclass=DicomTagEnumMeta):
    """Abstract class for DicomTag enumerations."""


    def __init__(self, *args):
        _parent: DicomTagEnum = None
        # Add the child tags as attributes
        if len(args) == 1 and isinstance(args[0], EnumMeta):
            enm = args[0]
            for name, val in enm._member_map_.items():
                self.__setattr__(name, val)
                val.__setattr__('_parent', self)

    @property
    def id(self):
        """Returns the identifier of the tag."""

        return self.value._id.value if isinstance(self.value, EnumMeta) else self.value

    @property
    def parents(self) -> list[DicomTagEnum]:
        """Returns the hierarchical parents of the tag."""

        p = self
        parents = []
        while not isinstance(p, DicomTag):
            p = p._parent
            parents.append(p)
        parents.reverse()

        return parents

    @property
    def path(self) -> str:
        """Returns the hierarchical path of the tag."""

        return '.'.join([p.name for p in self.parents] + [self.name])

@unique
class DicomTag(DicomTagEnum):
    """
    Enumeration for tags of a dicom file.

    References:

    - `Description of the tags <https://dicom.innolitics.com/ciods/ophthalmic-photography-8-bit-image>`_
    - The summary of the investigation on some dicom samples: ``T://Studies/CohortBuilder/data/dcm samples``
    - `Dicom file format standards <https://dicom.nema.org/medical/dicom/current/output/chtml/part10/chapter_7.html>`_
    - `Metadata samples <https://tableau-qua.fhv.ch/#/site/HOJG/views/AnalysedeschampsDicomdescastests/SOPClass>`_
    - `Transfer Syntax UIDs <https://pydicom.github.io/pydicom/stable/old/image_data_handlers.html#supported-transfer-syntaxes>`_
    """

    # Meta-data
    FileMetaInformationGroupLength = (0x0002, 0x0000)
    FileMetaInformationVersion = (0x0002, 0x0001)
    #: Unique for modalities
    MediaStorageSOPClassUID = (0x0002, 0x0002)
    #: Should be generated with pydicom.uid.generate_uid for each file
    MediaStorageSOPInstanceUID = (0x0002, 0x0003)
    #: Image compression method
    TransferSyntaxUID = (0x0002, 0x0010)
    ImplementationClassUID = (0x0002, 0x0012)
    ImplementationVersionName = (0x0002, 0x0013)

    # Plain tags
    #: It will be generated automatically by pydicom
    SpecificCharacterSet = (0x0008,0x0005)
    ImageType = (0x0008,0x0008)
    InstanceCreationDate = (0x0008,0x0012)
    InstanceCreationTime = (0x0008,0x0013)
    #: Same as MediaStorageSOPClassUID
    SOPClassUID = (0x0008,0x0016)
    #: Same as MediaStorageSOPInstanceUID
    SOPInstanceUID = (0x0008,0x0018)
    #: Datetime string format '%Y%m%d'
    StudyDate = (0x0008,0x0020)
    #: Datetime string format '%Y%m%d'
    SeriesDate = (0x0008,0x0021)
    #: Datetime string format '%Y%m%d'
    AcquisitionDate = (0x0008,0x0022)
    #: Datetime string format '%Y%m%d'
    ContentDate = (0x0008,0x0023)
    #: Datetime string format '%Y%m%d%H%M%S.%f'
    AcquisitionDateTime = (0x0008,0x002A)
    #: String format 'H%M%S'
    StudyTime = (0x0008,0x0030)
    #: String format 'H%M%S'
    SeriesTime = (0x0008,0x0031)
    #: String format 'H%M%S'
    AcquisitionTime = (0x0008,0x0032)
    #: String format 'H%M%S'
    ContentTime = (0x0008,0x0033)
    AccessionNumber = (0x0008,0x0050)
    # 'OP': Ophthalmic Photography, 'OPT': Ophthalmic Tomography
    Modality = (0x0008,0x0060)
    #: Required, empty if unknown
    Manufacturer = (0x0008,0x0070)
    #: Optional
    InstitutionName = (0x0008,0x0080)
    #: Optional
    ReferringPhysicianName = (0x0008,0x0090)
    #: Optional
    StationName = (0x0008,0x1010)
    #: Institution-generated description
    StudyDescription = (0x0008,0x1030)
    #: Description of the series
    SeriesDescription = (0x0008,0x103E)
    OperatorsName = (0x0008,0x1070)
    #: Optional
    ManufacturerModelName = (0x0008,0x1090)
    DerivationDescription = (0x0008,0x2111)
    PatientName = (0x0010, 0x0010)
    PatientID = (0x0010, 0x0020)
    #: Optional
    IssuerOfPatientID = (0x0010, 0x0021)
    #: Datetime string format '%Y%m%d'
    PatientBirthDate = (0x0010, 0x0030)
    #: 'M', 'F', 'O'
    PatientSex = (0x0010, 0x0040)
    PatientComments = (0x0010, 0x4000)
    #: Optional
    DeviceSerialNumber = (0x0018, 0x1000)
    SoftwareVersions = (0x0018, 0x1020)
    SynchronizationTrigger = (0x0018, 0x106A)
    AcquisitionTimeSynchronized = (0x0018, 0x1800)
    DetectorType = (0x0018, 0x7004)
    AcquisitionDuration = (0x0018, 0x9073)
    #: Unique identifier of the study
    StudyInstanceUID = (0x0020, 0x000D)
    #: Unique identifier of the series
    SeriesInstanceUID = (0x0020, 0x000E)
    #: User or equipment generated study identifier
    StudyID = (0x0020, 0x0010)
    #: A number that identifies the seriesy, empty if unknown
    SeriesNumber = (0x0020, 0x0011)
    AcquisitionNumber = (0x0020, 0x0012)
    InstanceNumber = (0x0020, 0x0013)
    FrameOfReferenceUID = (0x0020, 0x0052)
    ImageLaterality = (0x0020, 0x0062)
    SynchronizationFrameOfReferenceUID = (0x0020, 0x0200)
    SOPInstanceUIDOfConcatenationSource = (0x0020, 0x0242)
    PositionReferenceIndicator = (0x0020, 0x1040)
    ConcatenationUID = (0x0020, 0x9161)
    InConcatenationNumber = (0x0020, 0x9162)
    InConcatenationTotalNumber = (0x0020, 0x9163)
    ConcatenationFrameOffsetNumber = (0x0020, 0x9228)
    EmmetropicMagnification = (0x0022, 0x000A)
    IntraOcularPressure = (0x0022, 0x000B)
    HorizontalFieldOfView = (0x0022, 0x000C)
    PupilDilated = (0x0022, 0x000D)
    LightPathFilterTypeStackCodeSequence = (0x0022, 0x0017)
    RefractiveStateSequence = (0x0022, 0x001B)
    AxialLengthOfTheEye = (0x0022, 0x0030)
    DepthSpatialResolution = (0x0022, 0x0035)
    MaximumDepthDistortion = (0x0022, 0x0036)
    AlongScanSpatialResolution = (0x0022, 0x0037)
    MaximumAlongScanDistortion = (0x0022, 0x0038)
    AcrossScanSpatialResolution = (0x0022, 0x0048)
    MaximumAcrossScanDistortion = (0x0022, 0x0049)
    IlluminationWaveLength = (0x0022, 0x0055)
    IlluminationPower = (0x0022, 0x0056)
    IlluminationBandwidth = (0x0022, 0x0057)
    #: 1 for grayscale images, 3 for RGB images
    SamplesPerPixel = (0x0028, 0x0002)
    #: Specifies the intended interpretation of the pixel data.
    PhotometricInterpretation = (0x0028, 0x0004)
    #: Number of frames in a multi-frame image
    NumberOfFrames = (0x0028, 0x0008)
    #: Height of each image
    Rows = (0x0028, 0x0010)
    #: Width of each image
    Columns = (0x0028, 0x0011)
    #: Number of bits allocated to each pixel
    BitsAllocated = (0x0028, 0x0100)
    #: Number of bits stored for each pixel
    BitsStored = (0x0028, 0x0101)
    #: Should be BitsAllocated minus 1
    HighBit = (0x0028, 0x0102)
    #: Data representation of the pixel samples
    PixelRepresentation = (0x0028, 0x0103)
    #: YES/NO
    BurnedInAnnotation = (0x0028, 0x0301)
    RescaleIntercept = (0x0028, 0x1052)
    RescaleSlope = (0x0028, 0x1053)
    #: ``"00"`` means No, ``"01"`` means YES
    LossyImageCompression = (0x0028, 0x2110)
    #: Required if LossyImageCompressionRatio is '01'
    LossyImageCompressionRatio = (0x0028, 0x2112)
    PresentationLUTShape = (0x2050, 0x0020)
    PatientEyeMovementCommanded = (0x0022, 0x0005)
    PixelSpacing = (0x0028, 0x0030)
    #: A data stream of the pixel samples that comprise the image
    PixelData = (0x7FE0, 0x0010)

    # Slices
    E2E = slice((0x0051, 0x0000), (0x0052, 0x0000))
    E2E_RTX = slice((0x5555, 0x0000), (0x5556, 0x0000))
    DeviceDetails = slice((0x0022, 0x0000), (0x023, 0x0000))

    # Sequences

    class AnatomicRegionSequence(DicomTagEnum):
        _id = (0x0008,0x2218)
        CodeValue = (0x0008,0x0100)
        CodingSchemeDesignator = (0x0008,0x0102)
        CodeMeaning = (0x0008,0x0104)

    class DimensionOrganizationSequence(DicomTagEnum):
        _id = (0x0020, 0x9221)

    class DimensionIndexSequence(DicomTagEnum):
        _id = (0x0020, 0x9222)

    class AcquisitionDeviceTypeCodeSequence(DicomTagEnum):
        _id = (0x0022, 0x0015)
        CodeValue = (0x0008, 0x0100)
        CodingSchemeDesignator = (0x0008, 0x0102)
        CodeMeaning = (0x0008, 0x0104)

    class ContrastBolusAgentSequence(DicomTagEnum):
        _id = (0x0018, 0x0012)
        CodeValue = (0x0008, 0x0100)
        CodingSchemeDesignator = (0x0008, 0x0102)
        CodeMeaning = (0x0008, 0x0104)
        ContrastBolusVolume = (0x0018, 0x1041)
        ContrastBolusIngredientConcentration = (0x0018, 0x1049)
        ContrastBolusAgentNumber = (0x0018, 0x9337)
        class ContrastBolusAdministrationRouteSequence(DicomTagEnum):
            _id = (0x0018, 0x0014)
            CodeValue = (0x0008, 0x0100)
            CodingSchemeDesignator = (0x0008, 0x0102)
            CodeMeaning = (0x0008, 0x0104)
        class ContrastAdministrationProfileSequence(DicomTagEnum):
            _id = (0x0018, 0x9340)
            ContrastBolusVolume = (0x0018, 0x1041)
            ContrastBolusStartTime = (0x0018, 0x1042)
            ContrastBolusStopTime = (0x0018, 0x1043)

    class PatientEyeMovementCommandCodeSequence(DicomTagEnum):
        _id = (0x0022, 0x0006)
        CodeValue = (0x0008, 0x0100)
        CodingSchemeDesignator = (0x0008, 0x0102)
        CodeMeaning = (0x0008, 0x0104)

    class AcquisitionContextSequence(DicomTagEnum):
        _id = (0x0040, 0x0555)

    class SharedFunctionalGroupsSequence(DicomTagEnum):
        _id = (0x5200, 0x9229)
        class ReferencedImageSequence(DicomTagEnum):
            _id = (0x0008, 0x1140)
            ReferencedSOPClassUID = (0x0008, 0x1150)
            ReferencedSOPInstanceUID = (0x0008, 0x1155)
            class PurposeOfReferenceCodeSequence(DicomTagEnum):
                _id = (0x0040, 0xA170)
                CodeValue = (0x0008,0x0100)
                CodingSchemeDesignator = (0x0008,0x0102)
                CodeMeaning = (0x0008,0x0104)
        class FrameAnatomySequence(DicomTagEnum):
            _id = (0x0020, 0x9071)
            FrameLaterality = (0x0020, 0x9072)
            class AnatomicRegionSequence(DicomTagEnum):
                _id = (0x0008, 0x2218)
                CodeValue = (0x0008,0x0100)
                CodingSchemeDesignator = (0x0008,0x0102)
                CodeMeaning = (0x0008,0x0104)
        class PlaneOrientationSequence(DicomTagEnum):
            _id = (0x0020, 0x9116)
            ImageOrientationPatient = (0x0020, 0x0037)
        class PixelMeasuresSequence(DicomTagEnum):
            _id = (0x0028, 0x9110)
            SliceThickness = (0x0018, 0x0050)
            PixelSpacing = (0x0028, 0x0030)

    class PerFrameFunctionalGroupsSequence(DicomTagEnum):
        # Identifier
        _id = (0x5200, 0x9230)
        # Sequences
        class FrameContentSequence(DicomTagEnum):
            _id = (0x0020, 0x9111)
            FrameAcquisitionDateTime = (0x0018, 0x9074)
            FrameReferenceDateTime = (0x0018, 0x9151)
            FrameAcquisitionDuration = (0x0018, 0x9220)
            ReferenceCoordinates = (0x0022, 0x0032)
            StackID = (0x0020, 0x9056)
            InStackPositionNumber = (0x0020, 0x9057)
            DimensionIndexValues = (0x0020, 0x9157)
        class PlanePositionSequence(DicomTagEnum):
            _id = (0x0020, 0x9113)
            ImagePositionPatient = (0x0020, 0x0032)
        class OphthalmicFrameLocationSequence(DicomTagEnum):
            _id = (0x0022, 0x0031)
            ReferencedSOPClassUID = (0x0008, 0x1150)
            ReferencedSOPInstanceUID = (0x0008, 0x1155)
            ReferenceCoordinates = (0x0022, 0x0032)
            OphthalmicImageOrientation = (0x0022, 0x0039)

#: Related dicom header tags
_DICOM_METADATA_TAGS: list[DicomTagEnum] = [
    # Patient
    DicomTag.PatientID,
    DicomTag.PatientName,
    DicomTag.PatientSex,
    DicomTag.PatientBirthDate,
    DicomTag.PatientComments,
    # Datetime
    DicomTag.AcquisitionDateTime,
    DicomTag.InstanceCreationDate,
    DicomTag.AcquisitionDate,
    DicomTag.StudyDate,
    DicomTag.SeriesDate,
    DicomTag.ContentDate,
    DicomTag.InstanceCreationTime,
    DicomTag.AcquisitionTime,
    DicomTag.StudyTime,
    DicomTag.SeriesTime,
    DicomTag.ContentTime,
    # Image
    DicomTag.ImageLaterality,
    DicomTag.BurnedInAnnotation,
    DicomTag.Modality,
    DicomTag.ImageType,
    DicomTag.Rows,
    DicomTag.Columns,
    DicomTag.NumberOfFrames,
    DicomTag.PixelSpacing,
    DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.PixelSpacing,
    DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.SliceThickness,
    # Study
    DicomTag.StudyID,
    DicomTag.OperatorsName,
    DicomTag.ReferringPhysicianName,
    DicomTag.StudyDescription,
    DicomTag.SeriesDescription,
    DicomTag.AccessionNumber,
    # Device
    DicomTag.Manufacturer,
    DicomTag.ManufacturerModelName,
    DicomTag.DetectorType,
    DicomTag.DeviceSerialNumber,
    DicomTag.AcquisitionDuration,
    DicomTag.AcquisitionDeviceTypeCodeSequence.CodeValue,
    DicomTag.AcquisitionDeviceTypeCodeSequence.CodingSchemeDesignator,
    DicomTag.AcquisitionDeviceTypeCodeSequence.CodeMeaning,
    # SOP
    DicomTag.SOPClassUID,
    DicomTag.SOPInstanceUID,
    DicomTag.FrameOfReferenceUID,
    # Anatomic region
    DicomTag.AnatomicRegionSequence.CodeValue,
    DicomTag.AnatomicRegionSequence.CodingSchemeDesignator,
    DicomTag.AnatomicRegionSequence.CodeMeaning,
    DicomTag.SharedFunctionalGroupsSequence.FrameAnatomySequence.AnatomicRegionSequence.CodeValue,
    DicomTag.SharedFunctionalGroupsSequence.FrameAnatomySequence.AnatomicRegionSequence.CodingSchemeDesignator,
    DicomTag.SharedFunctionalGroupsSequence.FrameAnatomySequence.AnatomicRegionSequence.CodeMeaning,
    # Other
    DicomTag.StationName,
    DicomTag.InstitutionName,
    DicomTag.HorizontalFieldOfView,
    DicomTag.IlluminationWaveLength,
    DicomTag.PhotometricInterpretation,
    DicomTag.SynchronizationFrameOfReferenceUID,
    DicomTag.PatientEyeMovementCommanded,
    DicomTag.PatientEyeMovementCommandCodeSequence.CodeValue,
    DicomTag.PatientEyeMovementCommandCodeSequence.CodingSchemeDesignator,
    DicomTag.PatientEyeMovementCommandCodeSequence.CodeMeaning,
    DicomTag.ContrastBolusAgentSequence.CodeValue,
    DicomTag.ContrastBolusAgentSequence.CodingSchemeDesignator,
    DicomTag.ContrastBolusAgentSequence.CodeMeaning,
    DicomTag.ContrastBolusAgentSequence.ContrastBolusAdministrationRouteSequence.CodeValue,
    DicomTag.ContrastBolusAgentSequence.ContrastBolusAdministrationRouteSequence.CodingSchemeDesignator,
    DicomTag.ContrastBolusAgentSequence.ContrastBolusAdministrationRouteSequence.CodeMeaning,
]

_DICOM_METADATA_TAGNAMES: dict[DicomTagEnum, str] = {
    tag: tag.path.replace('.', '_')
    for tag in _DICOM_METADATA_TAGS
}

#: Type for efficient storage of dicom metadata
DicomMetadata = namedtuple(
    typename='DicomMetadata',
    field_names=_DICOM_METADATA_TAGNAMES.values(),
)

class DicomFile(File):
    """
    Class for managing a dicom file.

    Args:
        path: The path of the file.
        mode: The mode (local or remote) of the file.
        replace_phrase: Phrase to be replaced for anonymization.

    .. seealso::
        `DicomTags <src.cohortbuilder.utils.files.DicomTags>`
            Relative tags of a dicom file.
    """

    def __init__(self, path: Union[str, pathlib.Path], mode: Literal['remote', 'local'], replace_phrase: str = 'NA', from_conversion: bool = False):
        path = pathlib.Path(path)
        assert path.suffix.lower() in {'.dcm', '.dicom', '', '.zip'}
        super().__init__(path=path, mode=mode)
        #: The tags of the anonymized file will be replaced by this phrase
        self.replace_phrase: str = replace_phrase
        # Whether the file has originated from a conversion.
        self.__from_conversion: bool = from_conversion

    @File._copied
    def rectify(self) -> None:
        """
        Rectifies the tags of the local copy of a Dicom files and overwrites it.

        Clearing the e2e:

        - Clears the tags associated to the e2e file inside the dicom, making the file much smaller.
        - The patient ID inside these tags creates problems in Discovery. When it does not match with
            the patient ID in the corresponding Dicom tag, Discovery replaces the PID by "1".
        """
        # NOTE: Be CAREFUL when modifying this method.
        # Changing anything in the files results in duplicating files on Discovery.

        f = dcmread(self.copied)

        # Delete the embedded e2e
        del f[DicomTag.E2E.id]
        del f[DicomTag.E2E_RTX.id]

        # Delete the operator's name
        # NOTE: This will avoid some files from getting rejected
        # NOTE: Commented temorarily to avoid duplicating old files
        # if DicomTag.OperatorsName.id in f:
        #     del f[DicomTag.OperatorsName.id]

        if self.__from_conversion:
            self.__content_dates_to_acquisition_dates(f)

        f.save_as(self.copied)

    def __content_dates_to_acquisition_dates(self, f) -> None:
        """
        Sets the content date to the acquisition date of the dicom file.
        This is especially useful for converted files (eg. FDA or E2E) where the content date and time fields are those of conversion.
        """

        acquisition_datetime = DicomFile.readtag(f, DicomTag.AcquisitionDateTime)
        acquisition_date = acquisition_datetime[:-13] # 13 because 6 digits for time, 1 for space, 6 for precision time
        acquisition_time = acquisition_datetime[-13:]

        DicomFile.settag(f, DicomTag.ContentDate, acquisition_date)
        DicomFile.settag(f, DicomTag.ContentTime, acquisition_time)

    @File._copied
    def anonymize(self, hide_patient_sex: bool = True, new_anonymisation_method: bool = True) -> None:
        """
        Anonymizes a dicom file.

        Args:
            hide_patient_sex: Replaces the sex with ``"NA"``. Defaults to ``False``.
        """
        # NOTE: Be CAREFUL when modifying this method.
        # Changing anything in the files results in duplicating files on Discovery.

        f = dcmread(self.copied)

        # Remove the e2e tags
        # NOTE: They can be used for patient identification
        del f[DicomTag.E2E.id]
        del f[DicomTag.E2E_RTX.id]

        # Remove the operator name
        # NOTE: Combined with the acquisition date, it can be used for patient identification
        if DicomTag.OperatorsName.id in f:
            del f[DicomTag.OperatorsName.id]

        # Hide the pateint ID
        DicomFile.settag(ds=f, tag=DicomTag.PatientID, value=self.replace_phrase)

        # Hide the patient name
        DicomFile.settag(ds=f, tag=DicomTag.PatientName, value=f'{self.replace_phrase}^{self.replace_phrase}')

        # Hide the day and month from the patient birthdate
        dob = DicomFile.readtag(f, DicomTag.PatientBirthDate)
        year = dob[:4]
        try:
            if not dob: raise AttributeError('No DOB.')
            DicomFile.settag(f, DicomTag.PatientBirthDate, f'{year}0101')
        except (AttributeError, TypeError):
            logger.warning(f'{self.path.stem} | Failed to set PatientBirthDate attribute while anonymising DICOM file. Value {"is empty" if not dob else "has stayed as " + dob}.')
            # Continue on

        # Hide the patient sex
        if hide_patient_sex:
            DicomFile.settag(f, DicomTag.PatientSex, 'U')

        # TODO: Make sure that this is actually accepted by Discovery. Right now, errors or rejected files can happen about 50% of the time.
        if new_anonymisation_method:
            # We add this option in order to use the "old" anonymisation option if wanted.
            # This may be desired in order to not duplicate the already numerous pre-existing files on discovery.
            
            acquisition_date = datetime.strptime(DicomFile.readtag(f, DicomTag.ContentDate), '%Y%m%d')
            fake_generic_date = datetime( # Need to keep exact date for certain types of studies. Time in blanked out.
                acquisition_date.year,
                acquisition_date.month,
                acquisition_date.day,
                second=1 # One second so that written ContentTime is not invalid.
            )

            # Unique number for this file. Should remove.
            DicomFile.settag(f, tag=DicomTag.AccessionNumber, value=self.replace_phrase)
            # DicomFile.settag(f, tag=DicomTag.SOPInstanceUID, value=self.replace_phrase) # Changing this might cause DICOM rejection!
            instance_uid = DicomFile.readtag(f, tag=DicomTag.SOPInstanceUID) # Grab the UID (which is important), and copy it to all less important UIDs
            DicomFile.settag(f, tag=DicomTag.MediaStorageSOPInstanceUID, value=instance_uid)
            DicomFile.settag(f, tag=DicomTag.StudyInstanceUID, value=instance_uid)
            DicomFile.settag(f, tag=DicomTag.SeriesInstanceUID, value=instance_uid)
            DicomFile.settag(f, tag=DicomTag.FrameOfReferenceUID, value=instance_uid)


            # We give a fake date, year only.
            DicomFile.settag(f, tag=DicomTag.InstanceCreationDate, value=datetime.strftime(fake_generic_date, '%Y%m%d'))
            DicomFile.settag(f, tag=DicomTag.StudyDate, value=datetime.strftime(fake_generic_date, '%Y%m%d'))
            DicomFile.settag(f, tag=DicomTag.SeriesDate, value=datetime.strftime(fake_generic_date, '%Y%m%d'))
            DicomFile.settag(f, tag=DicomTag.ContentDate, value=datetime.strftime(fake_generic_date, '%Y%m%d'))
            DicomFile.settag(f, tag=DicomTag.AcquisitionDateTime, value=datetime.strftime(fake_generic_date, '%Y%m%d%H%M%S.%f'))
            DicomFile.settag(f, tag=DicomTag.ContentTime, value=datetime.strftime(fake_generic_date, '%H%M%S.%f'))

            DicomFile.settag(f, tag=DicomTag.StudyTime, value=datetime.strftime(fake_generic_date, '%H%M%S'))
            DicomFile.settag(f, tag=DicomTag.SeriesTime, value=datetime.strftime(fake_generic_date, '%H%M%S')) # These are slightly different, but we blank them out too.
            
            # Remove information which identifies via the institution or treatment practitioner
            DicomFile.settag(f, tag=DicomTag.ReferringPhysicianName, value=self.replace_phrase)
            DicomFile.settag(f, tag=DicomTag.InstitutionName, value=self.replace_phrase)


        f.save_as(self.copied)
        # Save to the location of the copy.
        # self.copy() must have been called first to create this copy and set the path.

    @File._local
    def ispdf(self) -> bool:
        """Checks if the file content is a pdf."""

        tag = DicomTag.MediaStorageSOPClassUID
        f = dcmread(self.path, specific_tags=[tag.id])
        if f.file_meta[tag.id].repval.lower() == 'encapsulated pdf storage':
            return True
        else:
            return False

    @File._local
    def template(self, pixeldata: bool = True, patientinfo: bool = True) -> pydicom.FileDataset:
        """
        Clears the tags of the local file and returns it.

        Args:
            pixeldata: If ``True``, pixel data and its associated tags will be cleared. Defaults to ``True``.
            patientinfo: If ``True``, patient data will be cleared. Defaults to ``True``.
        """
        # NOTE: Tags commented with NTBR have to be replaced later.

        # Read the local file
        f = dcmread(self.path)

        # Clear or replace metadata
        for tag, val in {
            DicomTag.MediaStorageSOPClassUID: OphthalmicTomographyImageStorage,
            DicomTag.MediaStorageSOPInstanceUID: None,  # NTBR
            DicomTag.ImplementationClassUID: None,
            DicomTag.ImplementationVersionName: None,
            DicomTag.TransferSyntaxUID: None,  # NTBR
        }.items():
            DicomFile.settag(f.file_meta, tag, val)

        # Clear or replace general tags
        for tag, val in {
            DicomTag.ImageType: None,
            DicomTag.SpecificCharacterSet: '',
            DicomTag.SOPClassUID: OphthalmicTomographyImageStorage,
            DicomTag.SOPInstanceUID: None,  # NTBR
            DicomTag.SeriesNumber: 1,
            DicomTag.AccessionNumber: None,
            DicomTag.InstitutionName: None,
            DicomTag.ReferringPhysicianName: None,
            DicomTag.StationName: None,
            DicomTag.StudyDescription: None,
            DicomTag.SeriesDescription: None,
            DicomTag.OperatorsName: None,
            DicomTag.DerivationDescription: None,
            DicomTag.DeviceSerialNumber: None,
            DicomTag.SoftwareVersions: None,
            DicomTag.DetectorType: None,
            DicomTag.AcquisitionDuration: None,
            DicomTag.AcquisitionNumber: None,
            DicomTag.InstanceNumber: 1,
            DicomTag.FrameOfReferenceUID: None,
            DicomTag.SynchronizationFrameOfReferenceUID: None,
            DicomTag.SynchronizationTrigger: None,
            DicomTag.AcquisitionTimeSynchronized: None,
            DicomTag.SOPInstanceUIDOfConcatenationSource: None,
            DicomTag.PositionReferenceIndicator: None,
            DicomTag.ConcatenationUID: None,
            DicomTag.InConcatenationNumber: None,
            DicomTag.InConcatenationTotalNumber: None,
            DicomTag.ConcatenationFrameOffsetNumber: None,
            DicomTag.SamplesPerPixel: 1,
            DicomTag.PhotometricInterpretation: 'MONOCHROME2',
            DicomTag.BitsAllocated: 8,
            DicomTag.BitsStored: 8,
            DicomTag.HighBit: 7,
            DicomTag.PixelRepresentation: 0,
            DicomTag.BurnedInAnnotation: None,
            DicomTag.RescaleIntercept: None,
            DicomTag.RescaleSlope: None,
            DicomTag.LossyImageCompression: None,
            DicomTag.LossyImageCompressionRatio: None,
            DicomTag.PresentationLUTShape: None,
            DicomTag.StudyInstanceUID: None,
            DicomTag.SeriesInstanceUID: None,
            DicomTag.StudyID: None,
        }.items():
            DicomFile.settag(f, tag, val)

        # Remove detailed sequences
        for tag in [
            DicomTag.AnatomicRegionSequence,
            DicomTag.DimensionOrganizationSequence,
            DicomTag.AcquisitionContextSequence,
            DicomTag.DimensionIndexSequence,
        ]:
            if tag.id in f:
                del f[tag.id]

        # Remove slices
        del f[DicomTag.E2E.id]
        del f[DicomTag.E2E_RTX.id]
        del f[DicomTag.DeviceDetails.id]

        # Remove detailed shared functional groups tags
        for tag in [
            DicomTag.SharedFunctionalGroupsSequence.ReferencedImageSequence,
            DicomTag.SharedFunctionalGroupsSequence.FrameAnatomySequence,
            # DicomTag.SharedFunctionalGroupsSequence.PlaneOrientationSequence,
        ]:
            shared = DicomFile.gettag(f, DicomTag.SharedFunctionalGroupsSequence)
            if tag in shared:
                del shared[tag]

        # Remove detailed per frame functional groups tags
        for frame in DicomFile.gettag(f, DicomTag.PerFrameFunctionalGroupsSequence):
            # Clear frame content
            for tag, val in {
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.FrameAcquisitionDateTime: None,
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.FrameReferenceDateTime: None,
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.FrameAcquisitionDuration: None,
                DicomTag.PerFrameFunctionalGroupsSequence.OphthalmicFrameLocationSequence.ReferencedSOPClassUID: None,
                DicomTag.PerFrameFunctionalGroupsSequence.OphthalmicFrameLocationSequence.ReferencedSOPInstanceUID: None,
                DicomTag.PerFrameFunctionalGroupsSequence.OphthalmicFrameLocationSequence.ReferenceCoordinates: [0, 0, 0, 500],
            }.items():
                DicomFile.settag(frame, tag, val, base=DicomTag.PerFrameFunctionalGroupsSequence)

        if pixeldata:
            # Clear pixel data
            DicomFile.settag(f, DicomTag.PixelData, b'')

            # Keep only one per frame item
            tag = DicomTag.PerFrameFunctionalGroupsSequence
            DicomFile.settag(
                ds=f,
                tag=tag,
                value=DicomFile.readtag(f, tag)[:1]
            )


            # Clear tags
            date = '19000101'
            time = '000000'
            for tag, val in {
                # Tags associated to pixels
                DicomTag.Rows: None,  # NTBR
                DicomTag.Columns: None,  # NTBR
                DicomTag.NumberOfFrames: None,  # NTBR
                DicomTag.ImageType: ['DERIVED', 'PRIMARY'],
                DicomTag.ImageLaterality: None,
                # Datetimes
                DicomTag.SeriesDate: date,
                DicomTag.SeriesTime: time,
                DicomTag.StudyDate: date,
                DicomTag.StudyTime: time,
                DicomTag.ContentDate: date,
                DicomTag.ContentTime: time,
                DicomTag.AcquisitionDateTime: date + time + '.0000',
                DicomTag.InstanceCreationDate: date,
                DicomTag.InstanceCreationTime: time,
                # Manufacturer
                DicomTag.Manufacturer: None,
                DicomTag.ManufacturerModelName: None,
                # Spacings
                DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.SliceThickness: None,
                DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.PixelSpacing: None,
            }.items():
                DicomFile.settag(f, tag, val)

        # Clear patient info
        if patientinfo:
            for tag, val in {
                DicomTag.PatientName: None,
                DicomTag.PatientID: None,
                DicomTag.PatientBirthDate: None,
                DicomTag.PatientSex: None,
            }.items():
                DicomFile.settag(f, tag, val)

        # Replace the pixel data with random noise
        frames = np.random.randint(size=(2, 500, 500), low=0, high=255, dtype='uint8')
        self._replace(f=f, frames=frames)

        return f

    # TODO: Keep the tag/settings that correspond to the icons in Discovery
    # NOTE: Files get rejected by Discovery if manufacturer is "Topcon".
    # TODO: Generalize the method to other imaging modalities.
    # TODO: Create from scratch instead of using a template.
    def create(
        self,
        images: list[np.ndarray],
        info: dict,
        template: Union[str, pathlib.Path] = None,
    ) -> pydicom.FileDataset:
        """
        Creates a dicom file by filling in the necessary values from a template.
        This method is only tested for creating multi-frame OCT cubes.

        Args:
            images: Frames of the dicom file.
            info: Metadata and patient information.
            template: If given, the template is read from this file.
              If ``None``, file should exist locally and a template is created it.
              Defaults to ``None`` .

        Raises:
            Exception: If neither a template is given nor the file is available locally.

        Returns:
            The generated dicom file.
        """

        if template and pathlib.Path(template).exists():
            # Read the template file
            f = dcmread(fp=template)
        elif self.path and self.path.exists():
            # Create a template from the local file
            f = self.template()
        else:
            raise Exception

        # Set unique SOP Instance UID
        instance_uid = generate_uid()
        DicomFile.settag(f.file_meta, DicomTag.MediaStorageSOPInstanceUID, instance_uid)
        DicomFile.settag(f, DicomTag.SOPInstanceUID, instance_uid)

        # Set the datetimes
        now = datetime.now()
        if 'study' in info and 'studyDatetime' in info['study']:
            datetime_study = datetime.strptime(info['study']['studyDatetime'], '%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            datetime_study = now
        for tag, val in {
            DicomTag.SeriesDate: datetime_study.strftime('%Y%m%d'),
            DicomTag.SeriesTime: datetime_study.strftime('%H%M%S'),
            DicomTag.StudyDate: datetime_study.strftime('%Y%m%d'),
            DicomTag.StudyTime: datetime_study.strftime('%H%M%S'),
            DicomTag.AcquisitionDateTime: datetime_study.strftime('%Y%m%d%H%M%S.%f'),
            DicomTag.ContentDate: datetime_study.strftime('%Y%m%d'),
            DicomTag.ContentTime: datetime_study.strftime('%H%M%S'),
            DicomTag.InstanceCreationDate: now.strftime('%Y%m%d'),
            DicomTag.InstanceCreationTime: now.strftime('%H%M%S'),
        }.items():
            DicomFile.settag(f, tag, val)

        # Set unique instance UIDs
        prefix = '2.16.756.5.30.1.328.50'  # HOJG prefix
        DicomFile.settag(f, DicomTag.StudyInstanceUID, f'{prefix}.1.1.1.{datetime_study.strftime("%Y%m%d%H%M%S%f")}')
        DicomFile.settag(f, DicomTag.SeriesInstanceUID, f'{prefix}.1.2.1.{datetime_study.strftime("%Y%m%d%H%M%S%f")}')

        # Set the device and manufacturer
        if 'manufacturer' in info:
            DicomFile.settag(f, DicomTag.Manufacturer, info['manufacturer'])
        if 'device' in info:
            DicomFile.settag(f, DicomTag.ManufacturerModelName, info['device'])

        # Set the patient info
        if 'patient' in info:
            for tag, val in {
                DicomTag.PatientName: f'{info["patient"]["surname"]}^{info["patient"]["name"]}',
                DicomTag.PatientID: info['patient']['patientId'],
                DicomTag.PatientBirthDate: datetime.strptime(info['patient']['birthdate'], '%Y-%m-%d').strftime('%Y%m%d')
                    if info['patient']['birthdate'] else None,
                DicomTag.PatientSex: info['patient']['sex'],
            }.items():
                DicomFile.settag(f, tag, val)

        # Set the accession number
        if 'patient' in info:
            DicomFile.settag(f, DicomTag.AccessionNumber, info['patient']['patientId'])

        # Replace the pixel data
        DicomFile._replace(
            f=f,
            frames=np.stack(images),
            spacing=(info['spacing'] if 'spacing' in info else None),
            thickness=(info['thickness'] if 'thickness' in info else None),
            laterality=(info['laterality'] if 'laterality' in info else None),
            compression='JPEGBaseline',
        )

        return f

    @staticmethod
    def _replace(f: pydicom.FileDataset, frames: np.ndarray,
            spacing: list[float] = None, thickness: float = None, laterality: Literal['R', 'L'] = None,
            compression: Literal['RLELossless', 'JPEGBaseline', 'JPEGExtended', 'JPEG2000Lossless'] = 'JPEG2000Lossless'
            ) -> None:
        """
        Replaces the pixel data of a file dataset and updates its associated tags.

        Args:
            f: File dataset read with pydicom.dcmread
            frames: Frames to be replaced, could be 2D or 3D numy array.
            spacing: The spacings (H, W) in mm/px. Defaults to ``None``.
            thickness: The spacing between the B-scans (depth) in mm. Defaults to ``None``.
            laterality: Laterality of the scan. Defaults to ``None``.
            compression: The image compression protocol. Defaults to 'JPEG2000Lossless'.
        """

        # Update shape tags
        DicomFile.settag(f, DicomTag.NumberOfFrames, (frames.shape[-3] if len(frames.shape) == 3 else 1))
        DicomFile.settag(f, DicomTag.Rows, frames.shape[-2])
        DicomFile.settag(f, DicomTag.Columns, frames.shape[-1])

        # Create per frame items
        # TODO: Create the perframe_item from scratch instead of reading it
        acquisitiondatetime = datetime.strptime(DicomFile.readtag(f, DicomTag.AcquisitionDateTime), '%Y%m%d%H%M%S.%f')
        perframe_item = DicomFile.gettag(f, DicomTag.PerFrameFunctionalGroupsSequence)[0].copy()
        perframe = [deepcopy(perframe_item) for _ in range(len(frames))]
        for idx, item in enumerate(perframe):
            for tag, val in {
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.InStackPositionNumber: idx + 1,
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.DimensionIndexValues: [1, idx + 1],
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.FrameAcquisitionDateTime:
                    datetime.strftime(acquisitiondatetime + (idx + 1) * timedelta(seconds=2), '%Y%m%d%H%M%S.%f'),
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.FrameReferenceDateTime:
                    DicomFile.readtag(f, DicomTag.AcquisitionDateTime),
                DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.FrameAcquisitionDuration: 2.0,
            }.items():
                DicomFile.settag(item, tag, val, base=DicomTag.PerFrameFunctionalGroupsSequence)
            if len(frames.shape) == 3:
                h = frames.shape[-3] - idx - 1
                for tag, val in {
                    DicomTag.PerFrameFunctionalGroupsSequence.FrameContentSequence.ReferenceCoordinates: [h, 0, h, frames.shape[-1]],
                    DicomTag.PerFrameFunctionalGroupsSequence.PlanePositionSequence.ImagePositionPatient: [0, 0, idx * (thickness or .1)]
                }.items():
                    DicomFile.settag(item, tag, val, base=DicomTag.PerFrameFunctionalGroupsSequence)
        DicomFile.settag(f, DicomTag.PerFrameFunctionalGroupsSequence, perframe)

        # Update spacings and laterality if given
        if laterality:
            DicomFile.settag(f, DicomTag.ImageLaterality, laterality)
        if thickness:
            DicomFile.settag(f, DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.SliceThickness, thickness)
        if spacing:
            DicomFile.settag(f, DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.PixelSpacing, spacing)

        # Set the plane orientation
        DicomFile.settag(f, DicomTag.SharedFunctionalGroupsSequence.PlaneOrientationSequence.ImageOrientationPatient, [1, 0, 0, 0, 1, 0])

        # Compress the images with RLELossless
        if compression == 'RLELossless':
            f.file_meta.TransferSyntaxUID = RLELossless
            f.compress(RLELossless, arr=frames)

        elif compression in {'JPEGBaseline', 'JPEGExtended', 'JPEG2000Lossless'}:
            # Compress the images using PIL.Image
            frames_compressed = []
            assert frames.dtype == 'uint8'
            for frame in frames:
                img = Image.fromarray(frame)
                img_bytes = io.BytesIO()
                img.save(
                    fp=img_bytes,
                    format='JPEG2000' if compression == 'JPEG2000Lossless' else 'JPEG',
                    irreversible=False,  # Only works for JPEG2000Lossless
                )
                frames_compressed.append(img_bytes.getvalue())

            # Update metadata
            DicomFile.settag(f, DicomTag.PhotometricInterpretation, 'MONOCHROME1')
            DicomFile.settag(f, DicomTag.HighBit, 7)

            # Basic encapsulation
            if compression == 'JPEGBaseline':
                encapsulation = encapsulate(frames_compressed)
                f.file_meta.TransferSyntaxUID = JPEGBaseline
                f.PixelData = encapsulation

            # Extended encapsulation
            elif compression == 'JPEGExtended':
                encapsulation = encapsulate_extended(frames_compressed)
                f.file_meta.TransferSyntaxUID = JPEGExtended
                f.PixelData = encapsulation[0]
                f.ExtendedOffsetTable = encapsulation[1]
                f.ExtendedOffsetTableLengths = encapsulation[2]

            elif compression == 'JPEG2000Lossless':
                encapsulation = encapsulate_extended(frames_compressed)
                f.file_meta.TransferSyntaxUID = JPEG2000Lossless
                f.PixelData = encapsulation[0]
                f.ExtendedOffsetTable = encapsulation[1]
                f.ExtendedOffsetTableLengths = encapsulation[2]

    @property
    @File._local
    def metadata(self) -> DicomMetadata:
        """The jason-serializable metadata of the file to be stored."""

        f = dcmread(self.path, stop_before_pixels=True)

        # Get the values of all tags
        metadata = {
            _DICOM_METADATA_TAGNAMES[tag]: DicomFile.readtag(f, tag)
            for tag in _DICOM_METADATA_TAGS
        }

        # PersonName
        for tag in [
            DicomTag.PatientName,
            DicomTag.OperatorsName,
            DicomTag.ReferringPhysicianName,
        ]:
            val = DicomFile.readtag(f, tag)
            metadata[_DICOM_METADATA_TAGNAMES[tag]] = ' '.join([
                val.given_name,
                val.middle_name,
                val.family_name,
            ]) if val else None
        # UID
        for tag in [
            DicomTag.SOPClassUID,
            DicomTag.SOPInstanceUID,
            DicomTag.FrameOfReferenceUID,
            DicomTag.SynchronizationFrameOfReferenceUID,
        ]:
            val = DicomFile.readtag(f, tag)
            metadata[_DICOM_METADATA_TAGNAMES[tag]] = str(val) if val else None
        # DSfloat
        for tag in [
            DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.SliceThickness,
        ]:
            val = DicomFile.readtag(f, tag)
            metadata[_DICOM_METADATA_TAGNAMES[tag]] = float(val) if val else None
        # MultiValue
        for tag in [
            DicomTag.ImageType,
            DicomTag.PixelSpacing,
            DicomTag.SharedFunctionalGroupsSequence.PixelMeasuresSequence.PixelSpacing,
        ]:
            vals = DicomFile.readtag(f, tag)
            metadata[_DICOM_METADATA_TAGNAMES[tag]] = [
                float(item) if isinstance(item, DSfloat) else str(item)
                for item in vals
            ] if vals else None
        # IS
        for tag in [
            DicomTag.NumberOfFrames,
        ]:
            val = DicomFile.readtag(f, tag)
            metadata[_DICOM_METADATA_TAGNAMES[tag]] = int(val) if val else None

        return DicomMetadata(**metadata)

    @staticmethod
    def gettag(ds: Dataset, tag: DicomTagEnum, base: DicomTagEnum = None) -> DataElement:
        """
        Gets a (nested) tag from a file or a dicom dataset.
        When a sequence is encountered, the first dataset is reached.

        Args:
            ds: Dicom file or internal dataset.
            tag: The tag of interest
            base: The base tag of the input dataset.
              For the case of dicom file, it should be ``None``.
              Defaults to ``True``.

        Returns:
            The data element in the specified tag.
            If the specified tag is not found, ``None`` is returned.
        """

        # Remove the base from the parents
        if base:
            for p in base.parents:
                assert p in tag.parents
            parents = [p for p in tag.parents if p not in base.parents + [base]]
        else:
            parents = tag.parents

        # Read the parent tags sequentially
        for p in parents:
            if p.id in ds:
                seq = ds[p.id]
                if len(seq.value):
                    ds = seq[0]
                else:  # NOTE: Sometimes the sequence is empty
                    ds = None
                    break
            else:
                ds = None
                break

        # Read the tag
        if ds is not None:
            ds = ds[tag.id] if tag.id in ds else None

        return ds

    @staticmethod
    def readtag(ds: Dataset, tag: DicomTagEnum, base: DicomTagEnum = None) -> Any:
        """
        Reads the value of a (nested) tag from a file or a dicom dataset.
        When a sequence is encountered, the first dataset is reached.

        Args:
            ds: Dicom file or internal dataset.
            tag: The tag of interest
            base: The base tag of the input dataset.
              For the case of dicom file, it should be ``None``.
              Defaults to ``True``.

        Returns:
            The value of the data element in the specified tag.
            If the specified tag is not found, ``None`` is returned.
        """

        d = DicomFile.gettag(ds, tag, base)

        return d.value if d is not None else None

    @staticmethod
    def settag(ds: Dataset, tag: DicomTagEnum, value: Any, base: DicomTagEnum = None) -> None:
        """
        Sets a (nested) tag to a value.
        When a sequence is encountered, the first dataset is reached.
        If the tag does not exist, the method does not do anything.

        Args:
            ds: Dicom file or internal dataset.
            tag: The tag of interest
            base: The base tag of the input dataset.
              For the case of dicom file, it should be ``None``.
              Defaults to ``True``.

        Returns:
            The value of the data element in the specified tag.
            If the specified tag is not found, ``None`` is returned.
        """

        d = DicomFile.gettag(ds, tag, base)
        if d is not None:
            d.value = value

class GenericFileUploadWrapper(File):
    '''
    Class which wraps a file for upload to Discovery. Calls to DICOM-specific methods will do nothing, but write messages to the logger.
    '''
    def anonymize(self, *args, **kwargs):
        logger.trace(f"Attempted to anonymise a non-DICOM file: {repr(self)}, which is not supported.")
    def rectify(self, *args, **kwargs):
        logger.trace(f"Attempted to rectify a non-DICOM file: {repr(self)}, which is not supported.")

def process_fda_or_e2e_file(path_to_file: Union[str, pathlib.Path], mode: Literal['remote', 'local']) -> list[DicomFile]:
        '''
        This is a function for processing an E2E file, by converting it into multiple DICOM files.
        These files are returned as a list of DicomFile instances.

        Args:
            path: The path of the file.
            mode: The mode (local or remote) of the file.
        '''
        assert mode == 'local'
        cache_dir = pathlib.Path(Parser.settings['general']['cache']) / 'tmp' / 'convert'
        cache_dir.mkdir(parents=True, exist_ok=True)
        path_of_converted_to_dicom = create_dicom_from_oct(path_to_file.as_posix(), cache_dir) # One dicom for each fundus image
        dcms = [DicomFile(x, mode='local', from_conversion=True) for x in path_of_converted_to_dicom]

        first_conversion_uid = create_dicom_uid_from_file(path_of_converted_to_dicom[0])
        first_study_instance_uid = '1.2.826.0.1.3680043.8.498.' + first_conversion_uid
        for f in dcms:
            dicom_read = dcmread(f.path)
            DicomFile.settag(dicom_read, DicomTag.StudyInstanceUID, first_study_instance_uid) # Make all the files (which are of the same study) have the same StudyInstanceUID

            # Set UIDs of instance UID fields
            dicom_read.file_meta['MediaStorageSOPInstanceUID'].value = '1.2.826.0.1.3680043.8.498.' + first_conversion_uid # This needs to be set in a special way
            DicomFile.settag(dicom_read, DicomTag.SOPInstanceUID, '1.2.826.0.1.3680043.8.498.' + first_conversion_uid)
            DicomFile.settag(dicom_read, DicomTag.SeriesInstanceUID, '1.2.826.0.1.3680043.8.498.' + first_conversion_uid)

            dicom_read.save_as(f.path)

        return dcms
