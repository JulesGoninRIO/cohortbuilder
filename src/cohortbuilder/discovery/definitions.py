"""
This module includes the names and the definitions that exist in Discovery.
"""

from __future__ import annotations

from enum import Enum


class DiscoveryFileStatus(Enum):
    """
    Status values that a file in Discovery can get.
    It shows the status of the upload and the parsing of the file.
    """

    #: The file has been created
    CREATED = -1
    #: The file is in the pre-processing queue
    PENDING = 0
    #: The file is uploaded and is being pre-processed
    PARSING = 1
    #: There has been an error uploading the file
    ERROR = 1.1
    #: Timeout occured while uploading the file
    TIMEOUT = 1.2
    #: The file is uploaded and pre-processed
    READY = 2
    #: The file is rejected
    REJECTED = 2.1
    #: The file has been deleted
    DELETED = 4

    @property
    def isuploaded(self) -> bool:
        """
        Indicates whether the file has been uploaded successfully.
        """

        if self in {
            DiscoveryFileStatus.ERROR,
            DiscoveryFileStatus.TIMEOUT,
            DiscoveryFileStatus.DELETED,
        }:
            return False
        if self in {
            DiscoveryFileStatus.PENDING,
            DiscoveryFileStatus.PARSING,
            DiscoveryFileStatus.READY,
            DiscoveryFileStatus.REJECTED,
        }:
            return True
        else:
            raise NotImplementedError

    @property
    def isended(self) -> bool | None:
        """
        Indicates whether processing the file has ended.
        If the status is not representative, ``None`` is returned.
        """

        if self in [
            DiscoveryFileStatus.ERROR,
            DiscoveryFileStatus.TIMEOUT,
            DiscoveryFileStatus.REJECTED,
            DiscoveryFileStatus.DELETED,
        ]:
            return True
        elif self in [
            DiscoveryFileStatus.PENDING,
            DiscoveryFileStatus.PARSING,
        ]:
            return False
        elif self in [
            DiscoveryFileStatus.READY,
        ]:
            return None
        else:
            raise NotImplementedError

class DiscoveryDatasetStatus(Enum):
    """
    Status values that a dataset in Discovery can get.
    General status of the pre- and post-processing of the dataset.
    It does not get updated by Discovery if a task is manually reprocessed in a new job.
    """

    #: The status of the dataset is not accessible because it is owned by another user
    NONE = -9
    #: The dataset is drafted
    DRAFT = -1
    #: Pre-processing the dataset is pending
    PENDING = 0
    #: The dataset is being pre-processed
    PROCESSING = 1
    #: The dataset has been pre-processed
    READY = 2
    #: Pre-processing the dataset has had errors
    ERROR = 2.1
    #: Pre-processing the dataset has been timed out
    TIMEOUT = 2.2
    #: Post-processing the dataset has been successful
    POSTPROCESSED = 3
    #: The dataset has been deleted
    DELETED = 9

    @property
    def isended(self) -> bool | None:
        """
        Indicates whether processing the dataset has ended.
        If the status is not representative, ``None`` is returned.
        """

        if self in [
            DiscoveryDatasetStatus.DRAFT,
            DiscoveryDatasetStatus.TIMEOUT,
            DiscoveryDatasetStatus.DELETED,
        ]:
            return True
        elif self in [
            DiscoveryDatasetStatus.PENDING,
            DiscoveryDatasetStatus.PROCESSING,
            DiscoveryDatasetStatus.READY,
        ]:
            return False
        elif self in [
            DiscoveryDatasetStatus.POSTPROCESSED,
            DiscoveryDatasetStatus.ERROR,
            DiscoveryDatasetStatus.NONE,
        ]:
            return None
        else:
            raise NotImplementedError

    @property
    def issuccessful(self) -> bool | None:
        """
        Indicates whether processing the dataset has been successful.
        If the status is not representative, ``None`` is returned.
        """

        if self in [
            DiscoveryDatasetStatus.PENDING,
            DiscoveryDatasetStatus.PROCESSING,
            DiscoveryDatasetStatus.READY,
            DiscoveryDatasetStatus.DRAFT,
            DiscoveryDatasetStatus.TIMEOUT,
            DiscoveryDatasetStatus.DELETED,
        ]:
            return False
        elif self in [
            DiscoveryDatasetStatus.POSTPROCESSED,
            DiscoveryDatasetStatus.ERROR,
            DiscoveryDatasetStatus.NONE,
        ]:
            return None
        else:
            raise NotImplementedError

class DiscoveryJobStatus(Enum):
    """
    Status values that a job in Discovery can get.
    """

    #: The job has not started yet
    PENDING = 0
    #: The job has started but is not yet done
    STARTED = 1
    #: The job has ended but there has been an ERROR in one of the tasks
    ENDED = 2
    #: The job had ended successfully
    SUCCESS = 3

class DiscoveryTaskStatus(Enum):
    """
    Status values that a task in Discovery can get.
    """

    #: The task depends on another incomplete task
    PENDING = 0
    #: Received, rare
    RECEIVED = 0.1
    #: Retry, rare
    RETRY = 0.2
    # Dispatched, rare. Probably that the task is starting up.
    DISPATCHED = 0.3
    #: The task has started but is not done
    STARTED = 1
    #: The task is not relative
    REJECTED = 1.1
    #: Revoked, rare
    REVOKED = 1.2
    #: The task has ended successfully
    SUCCESS = 2
    #: The task has ended because it is not possible
    ENDED = 2.1
    #: The task has failed
    ERROR = 2.2
    #: Timeout, rare
    TIMEOUT = 2.3
    #: Failure, rare
    FAILURE = 2.4

    @property
    def isultimate(self):
        """Indicates whether the status is ultimate or it might get improved by relaunching the task."""

        return self in [
            DiscoveryTaskStatus.REJECTED,
            DiscoveryTaskStatus.SUCCESS,
        ]

    @property
    def isnegative(self):
        '''Indicates whether the status is non-desirable / negative, i.e., not neutral (waiting / running), nor positive (success).'''

        return self in [
            DiscoveryTaskStatus.REJECTED,
            DiscoveryTaskStatus.REVOKED,
            DiscoveryTaskStatus.ENDED,
            DiscoveryTaskStatus.ERROR,
            DiscoveryTaskStatus.TIMEOUT,
            DiscoveryTaskStatus.FAILURE
        ]

class DiscoveryDatasetPurpose(Enum):
  """Purpose values that a dataset in Discovery might have."""

  #: Indicates dataset was imported from a file
  IMPORT = 'IMPORT'
  #: Indicates dataset was generated by a processor
  PROCESSOR = 'PROCESSOR'
  #: Indicates dataset was user-generated as a correction to a PROCESSOR dataset
  MANUAL = 'MANUAL'
  #: Indicates dataset was user-generated as an eCRF linked to a patient study
  ECRF = 'ECRF'
  #: Indicates dataset was user-generated as an annotation linked to an IMPORT dataset
  ANNOTATION = 'ANNOTATION'
  #: Indicates dataset was user-generated as a custom dataset by a script
  CUSTOM = 'CUSTOM'

class DiscoveryProcessor(Enum):
    """Discovery processors and their corresponding name on Discovery."""

    PREPROCESSOR = 'Dataset Preprocessor'
    LFSEGMENTATION = 'Layer and Fluid Segmentation Processor'
    GAPROGRESSION = 'GA Progression Prediction Processor'
    BIOMARKERS = 'Macular Biomarkers Processor'
    GASEGMENTATION = 'Geographic Atrophy Segmentation'
    POSTPROCESSOR = 'Dataset Postprocessor'

class DiscoveryTask(Enum):
    """Discovery tasks in RetinAI default processing."""

    # The values are the paths in the Default RetinAI job (the first job when an acquisition is created)
    # The order of the tasks must be such that the dependencies are met
    PREPROCESSOR = '1'  # Dataset Preprocessor
    POSTPROCESSOR = '1.4'  # Dataset Postprocessor
    LFSEGMENTATION = '1.1'  # Layer and Fluid Segmentation Processor
    LFSEGMENTATION_POSTPROCESSOR = '1.1.1'  # Dataset Postprocessor
    BIOMARKERS = '1.2'  # Macular Biomarkers Processor
    BIOMARKERS_POSTPROCESSOR = '1.2.1'  # Dataset Postprocessor
    GASEGMENTATION = '1.3'  # Geographic Atrophy Segmentation
    GASEGMENTATION_POSTPROCESSOR = '1.3.1'  # Dataset Postprocessor
    GAPROGRESSION = '1.1.2'  # Geographic Atrophy Progression Prediction Processor
    GAPROGRESSION_POSTPROCESSOR = '1.1.2.1'  # Dataset Postprocessor

    @property
    def processor(self) -> str:
        """Returns the name of the corresponding Discovery processor."""

        if self is DiscoveryTask.PREPROCESSOR:
            return DiscoveryProcessor.PREPROCESSOR
        if self is DiscoveryTask.LFSEGMENTATION:
            return DiscoveryProcessor.LFSEGMENTATION
        if self is DiscoveryTask.GAPROGRESSION:
            return DiscoveryProcessor.GAPROGRESSION
        if self is DiscoveryTask.BIOMARKERS:
            return DiscoveryProcessor.BIOMARKERS
        if self is DiscoveryTask.GASEGMENTATION:
            return DiscoveryProcessor.GASEGMENTATION
        elif self in [
            DiscoveryTask.POSTPROCESSOR,
            DiscoveryTask.LFSEGMENTATION_POSTPROCESSOR,
            DiscoveryTask.GAPROGRESSION_POSTPROCESSOR,
            DiscoveryTask.BIOMARKERS_POSTPROCESSOR,
            DiscoveryTask.GASEGMENTATION_POSTPROCESSOR
        ]:
            return DiscoveryProcessor.POSTPROCESSOR
        else:
            raise NotImplementedError

    @property
    def level(self) -> int:
        """Returns the level of the task."""

        return len(self.value.split('.')) - 1

    @property
    def postprocess(self) -> DiscoveryTask:
        """Returns the corresponding post-processor, if any."""

        if self.processor is DiscoveryProcessor.POSTPROCESSOR:
            return None
        elif self.processor is DiscoveryProcessor.PREPROCESSOR:
            return DiscoveryTask.POSTPROCESSOR
        elif self.processor is DiscoveryProcessor.LFSEGMENTATION:
            return DiscoveryTask.LFSEGMENTATION_POSTPROCESSOR
        elif self.processor is DiscoveryProcessor.BIOMARKERS:
            return DiscoveryTask.BIOMARKERS_POSTPROCESSOR
        elif self.processor is DiscoveryProcessor.GASEGMENTATION:
            return DiscoveryTask.GASEGMENTATION_POSTPROCESSOR
        elif self.processor is DiscoveryProcessor.GAPROGRESSION:
            return DiscoveryTask.GAPROGRESSION_POSTPROCESSOR
        else:
            raise NotImplementedError

    @property
    def parent(self) -> DiscoveryTask:
        """Returns the parent processor, if any."""

        if self in [
            DiscoveryTask.PREPROCESSOR,
            DiscoveryTask.LFSEGMENTATION,
            DiscoveryTask.BIOMARKERS,
            DiscoveryTask.GASEGMENTATION,
        ]:
            return None
        elif self is DiscoveryTask.POSTPROCESSOR:
            return DiscoveryTask.PREPROCESSOR
        elif self is DiscoveryTask.GAPROGRESSION:
            return DiscoveryTask.LFSEGMENTATION_POSTPROCESSOR
        elif self is DiscoveryTask.LFSEGMENTATION_POSTPROCESSOR:
            return DiscoveryTask.LFSEGMENTATION
        elif self is DiscoveryTask.BIOMARKERS_POSTPROCESSOR:
            return DiscoveryTask.BIOMARKERS
        elif self is DiscoveryTask.GAPROGRESSION_POSTPROCESSOR:
            return DiscoveryTask.GAPROGRESSION
        elif self is DiscoveryTask.GASEGMENTATION_POSTPROCESSOR:
            return DiscoveryTask.GASEGMENTATION
        else:
            raise NotImplementedError

    def subset(levels: list[int]) -> list[DiscoveryTask]:
        """Returns a list of tasks with indicated levels."""

        return [task for task in DiscoveryTask if task.level in levels]

class LayerType(Enum):
    """
    A dictionary for translating the layer types to their corresponding
    values in Discovery.
    """

    RAW = 'ScanType.Raw'
    FUNDUS = 'ScanType.Fundus'
    OCT = 'ScanType.OCT'
    OTHER = 'LayerType.Other'
    UNKNOWN = 'ScanType.Unknown'

class LayerVariant(Enum):
    """
    A dictionary for translating the layer variants to their corresponding
    values in Discovery.
    """

    # Fundus
    #: Color fundus image
    F_CFI = 'ScanVariantFundus.Color'
    #: Multicolor fundus image
    F_MFI = 'ScanVariantFundus.MultiColor'
    #: SLO fundus image (Present in all fundus images)
    F_SLO = 'ScanVariantFundus.SLO'
    #: Reconstruction fundus image
    F_RECONSTRUCTION = 'ScanVariantFundus.Reconstruction'
    #: Angiography fundus image
    F_ANGIO = 'ScanVariantFundus.Angio'
    #: ICG angiography fundus image
    F_ANGIOICG = 'ScanVariantFundus.AngioICG'
    #: Fluorescent angiography fundus image
    F_ANGIOFLUO = 'ScanVariantFundus.AngioFluo'
    #: Autofluorescent fundus image
    F_AUTOFLUO = 'ScanVariantFundus.Autofluo'
    #: Infrared fundus image
    F_INFRARED = 'ScanVariantFundus.Infrared'
    #: Autofluorescence or infrared fundus image
    F_AFIR = 'ScanVariantFundus.AF&IR'
    #: Infrared fundus image (not properly identified)
    F_IR = 'ScanVariantFundus.IR'
    #: Autofluorescence fundus image (not properly identified)
    F_AF = 'ScanVariantFundus.AF'
    #: Fundus image of eye (!)
    F_EYE = 'ScanVariantFundus.Eye'
    #: Low quality fundus image
    F_LQ = 'ScanVariantFundus.LQ'
    #: Fundus image centered on optic nerve head
    F_ONH = 'ScanVariantFundus.ONH'
    #: Fundus image centered on retina
    F_RETINA = 'ScanVariantFundus.Retina'
    #: Fundus image of other variants
    F_OTHER = 'ScanVariantFundus.Other'
    #: Grayscale fundus image
    F_GRAYSCALE = 'ScanVariantFundus.Grayscale'
    #: ICGA fundus image
    F_ICGA = 'ScanVariantFundus.ICGA'

    # OCT
    #: Structural OCT (present in all OCT images)
    OCT_STRUCTURAL = 'ScanVariantOCT.Structural'
    #: OCTA
    OCTA = 'ScanVariantOCT.Angiography'
    #: Cube OCT (C-scan)
    OCT_CUBE = 'ScanVariantOCT.CubeScan'
    #: Line OCT (B-scan)
    OCT_LINE = 'ScanVariantOCT.LineScan'
    #: Circle OCT
    OCT_CIRCLE = 'ScanVariantOCT.CircularScan'
    #: Radial OCT
    OCT_RADIAL = 'ScanVariantOCT.RadialScan'
    #: Star OCT
    OCT_STAR = 'ScanVariantOCT.StarScan'
    #: Cross OCT
    OCT_CROSS = 'ScanVariantOCT.CrossScan'
    #: Anterior segment OCT
    OCT_ANTSEG = 'ScanVariantOCT.AnteriorSegment'
    #: Infrared 800 OCT
    OCT_IR_800 = 'ScanVariantOCT.IR_800'

    # Other
    RAW = 'ScanVariant.Raw'
    RAW_IMAGE = 'ScanVariantRaw.Image'
    PDF = 'ScanVariantRaw.PDF'
    FORM = 'ScanVariant.Form'
    ECRF = 'ScanVariant.ECRF'
    SECONDARY = 'ScanVariant.Secondary'

    @classmethod
    def fundus_variants(cls) -> set[LayerVariant]:
        """Returns the layer variants containing the word 'Fundus'."""

        return {variant for variant in LayerVariant if 'Fundus' in variant.value}

    @classmethod
    def fundus_variants_added(cls) -> set[LayerVariant]:
        """Filters the original fundus variants from the list of the fundus variants."""

        variants_all = cls.fundus_variants()
        variants_original = {
            LayerVariant.F_SLO,
            LayerVariant.F_CFI,
            LayerVariant.F_RECONSTRUCTION,
            LayerVariant.F_MFI,
        }
        variants_computed = variants_all.difference(variants_original)
        return variants_computed

    @classmethod
    def OCT_variants(cls) -> set[LayerVariant]:
        """Returns the layer variants containing the word 'OCT'."""

        return {variant for variant in LayerVariant if 'OCT' in variant.value}
