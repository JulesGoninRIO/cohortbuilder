"""
This module includes the classes for taxonomy of ophthalmic scans.
As of now, the classes of the module are not tested and are not well-documented.

.. note::
    This module is not being maintained. The documentation is not updated.
"""

# TODO: Update the docstrings
# TODO: Update the type annotations

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, Literal
from zipfile import ZipFile

import cv2
import numpy as np
import pydicom as dicom
import skimage
import skimage.feature
from loguru import logger
from scipy.signal import savgol_filter

from src.cohortbuilder.discovery.definitions import LayerType, LayerVariant
from src.cohortbuilder.utils.helpers import read_img
from src.cohortbuilder.utils.imageprocessing import (apply_threshold, filter_ridges,
                                       get_ridges, is_round, rdg2drdg)

if TYPE_CHECKING:
    from src.cohortbuilder.discovery.entities import Layer


class FundusClassifier:
    """Layer variant classifier for fundus layers."""

    # NOTE: General idea for differenciating between the different image types:
    #     - Angios are white blood vessels on dark background, AF&IR are the opposite.
    #     - Infrared shows 'shallow' vessels, so each ridge would be two distinct ridges, separated by a few pixels.
    #     - Autofluo has 'full' vessels, ridges are not duplicated.
    #     - TODO: differentiate angio fluo and ICG.


    def __init__(self, layer: Layer):
        self.layer = layer
        self.n_images: int = len(layer.attributes['content'])
        if (
            'parentFile' in layer.parent.attributes
            and layer.parent.attributes['parentFile']
            and layer.parent.attributes['parentFile']['signedUrl']
        ):
            self.parent_file: str = layer.parent.attributes['parentFile']
        else:
            self.parent_file = None

    def classify(self) -> list[str]:
        """
        Classfies ...

        Returns:
            The classified layer variants.
        """

        # Assert the layer is a fundus and it's not a CFI
        assert self.layer.attributes['scanType'] == LayerType.FUNDUS.value,\
            'Layer is not a fundus. All the information are available in Discovery.'
        assert LayerVariant.F_CFI.value not in self.layer.attributes['scanVariant'],\
            'Layer is a CFI. Use the classifier corresponding to CFIs.'

        # Get the variants from dicom
        # NOTE: No extension means Dicom file
        if self.parent_file:
            ext = self.parent_file['extension']
            # TODO: use MIME type
            if (not ext) or (ext.lower() in {'dcm', 'zip'}):
                return self.extract_from_dcm()

        # Detect Angio movie
        # NOTE: Angio movie is not exported in Dicom
        # CHECK: Why 10?
        if self.n_images > 10:
            return [LayerVariant.F_ANGIO.value]

        # Detect ANGIO, IR, AF
        # TODO: Refresh the URL before downloading?
        if self.n_images != 1:
            logger.debug(f'{repr(self.layer)} | Assumed all fundus except movie \
                had exactly one image, this one has {self.n_images}.')
        url_uuid = self.layer.uuid + '-' + str(0).zfill(4)
        url = self.layer.parent.get_url(uuid=url_uuid)
        f = download(url, out=None)  # TODO: Use Discovery.download()
        img = read_img(f)
        if is_round(img=img, mrg=10):
            img = img[100:-100, 100:-100]
        variant = self.get_angio_variant(img)
        if variant == LayerVariant.F_AFIR:
            variant = self.infra_or_auto(img)
        return [variant.value]

    def extract_from_dcm(self) -> list[str]:
        """
        Downloads a parent file, extracts it if it's zip, and reads the variants.

        Returns:
            List of the variants of the fundus layer.
        """

        # Download the parent file
        url_uuid = self.layer.parent.uuid + '-' + '0002'
        url = self.layer.parent.get_url(uuid=url_uuid)
        file = download(url)  # TODO: Use Discovery.download()

        # Zip file
        if self.parent_file['extension'] == 'ZIP':
            with ZipFile(file, mode='r') as zipfile:
                names = zipfile.namelist()
                for name in names:
                    with zipfile.open(name) as f:
                        # NOTE: There will be only one fundus, others layers should be ignored.
                        variants = self.get_variant_from_dcm(file=f)
                        if variants:
                            return variants
                else:
                    return [LayerVariant.F_OTHER.value]
        # Dicom file
        else:
            return self.get_variant_from_dcm(file=file)

    # NOTE: Only tested with dicoms I had (from heyex). Might need to extend that method with data from other dicoms.
    def get_variant_from_dcm(self, file: BinaryIO) -> list[LayerVariant]:
        """Extract the layer variants from a dicom file."""

        data = dicom.read_file(file)

        if data['00220015'][0]['00080104'] == 'Optical Coherence Tomography Scanner':
            return []
        elif 'Raw' in data.file_meta.MediaStorageSOPClassUID.name:
            logger.debug(f'{repr(self.layer)} | RAW? Check.')
            return [LayerVariant.RAW.value]
        elif 'Secondary' in data.file_meta.MediaStorageSOPClassUID.name :
            logger.debug(f'{repr(self.layer)} | Secondary? Check.')
            return [LayerVariant.SECONDARY.value]

        type_tag = data.ImageType[-1]
        if type_tag == 'FA':
            return [LayerVariant.F_ANGIO.value, LayerVariant.F_ANGIOFLUO.value]
        elif type_tag == 'ICG':
            return [LayerVariant.F_ANGIO.value, LayerVariant.F_ANGIOICG.value]
        elif type_tag == 'MC':
            return [LayerVariant.F_MFI.value]
        elif type_tag == 'AF':
            loc = LayerVariant.F_ONH if data['00082218'][0]['00080104'].value == 'Optic nerve head' else LayerVariant.F_RETINA # Should work
            return [LayerVariant.F_AUTOFLUO.value, loc.value]
        elif type_tag == 'RED':
            loc = LayerVariant.F_ONH if data['00082218'][0]['00080104'].value == 'Optic nerve head' else LayerVariant.F_RETINA
            return [LayerVariant.F_INFRARED.value, loc.value]

        if data.AnatomicRegionSequence[0].CodeMeaning == 'Eye':
            return [LayerVariant.F_EYE.value]
        elif data.AcquisitionDeviceTypeCodeSequence[0].CodeMeaning == 'Fundus Camera':
            logger.debug(f'{repr(self.layer)} | This should be a CFI. Check.')
            return [LayerVariant.F_CFI.value]

        logger.debug(f'{repr(self.layer)} | Unknown Dicom type')
        return [LayerVariant.F_OTHER.value]

    def get_angio_variant(self, img: np.ndarray, threshold: float = .02)\
            -> Literal[LayerVariant.F_AFIR, LayerVariant.F_ANGIO, LayerVariant.F_OTHER]:
        """
        Classifies the fundus between AFIR, ANGIO, and other.

        Args:
            img: Fundus image read by read_img.
            threshold: Threshold for masking the ridges. Defaults to .02.

        Returns:
            LayerVariant: A layer variant between AFIR, ANGIO, and FUNDUS_OTHER.
        """

        # Create image crops
        n_tiles = 3
        M = img.shape[0] // n_tiles
        N = img.shape[1] // n_tiles
        tiles = [
            img[x:x+M, y:y+N]
            for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)
        ]

        n_black = 0
        n_white = 0
        for tile in tiles:
            # Get white and black ridges with thresholding (uint8)
            white = get_ridges(tile, black=False)
            black = get_ridges(tile, black=True)

            # Apply the threshold on black and white ridges
            white = apply_threshold(rdg=white, threshold=threshold)
            black = apply_threshold(rdg=black, threshold=threshold)

            # Get masks for white and black long ridges
            mask_white, length_white, _ = filter_ridges(rdg=white, length_threshold=70)
            mask_black, length_black, _ = filter_ridges(rdg=black, length_threshold=70)

            # Check the average lengths
            if length_white < 500 or length_black < 500:
                continue

            # Apply the masks on the tile
            tile_white = mask_white * tile
            tile_black = mask_black * tile

            # Calculate the mean pixel values for bright pixels
            dark_threshold = 2  # 0-255
            mean_white = tile_white[np.where(tile_white > dark_threshold)].mean()
            mean_black = tile_black[np.where(tile_black > dark_threshold)].mean()
            mean_tile = tile[np.where(tile > dark_threshold)].mean()

            # Calculate the differences and compare
            diff_a = (mean_tile - mean_black) / mean_black * 100
            diff_b = (mean_white - mean_tile) / mean_tile * 100
            if diff_a > 1.5 * diff_b:
                n_black += 1
            elif diff_b > 1 * diff_a:
                n_white += 1

        # Compare the numbers
        if n_black + n_white <= 2 or n_black == n_white:
            return LayerVariant.F_OTHER
        elif n_black > n_white:
            return LayerVariant.F_AFIR
        elif n_black < n_white:
            return LayerVariant.F_ANGIO

    def infra_or_auto(self, img: np.ndarray) -> Literal[LayerVariant.F_AUTOFLUO, LayerVariant.F_INFRARED]:
        """
        Distinguishes between autofluorescent and infrared fundus images.

        Args:
            img: Fundus image.

        Returns:
            Layer variant.
        """

        # Create image crop indexes
        n_tiles = 3
        M = img.shape[0] // n_tiles
        N = img.shape[1] // n_tiles
        slices = [
            (slice(x, x+M, None), slice(y, y+N, None))
            for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)
        ]

        # Instantiate ridges
        rdg_black = np.empty(img.shape, dtype=np.float64)
        rdg_blurred = np.empty(img.shape, dtype=np.float64)
        rdg_black_mask = np.empty(img.shape, dtype=bool)

        for s in slices:
            # Get the tile
            tile = img[s]

            # Get black and blurred black ridges
            black = get_ridges(tile, sigmas=range(1, 2, 2), black=True)
            blurred = get_ridges(tile, sigmas=range(4, 6, 2), black=True)
            rdg_black[s] = black
            rdg_blurred[s] = blurred

            # Apply threshold on black ridges and filter it
            black = apply_threshold(
                rdg=black,
                threshold=.9*np.quantile(black, 0.80),  # NOTE: Used to be .008
            )
            black_mask, _, _ = filter_ridges(black, length_threshold=500)
            rdg_black_mask[s] = black_mask

        # Get the double ridges
        rdg_mask = (rdg_blurred > .01) * rdg_black_mask
        drdg = rdg2drdg(mask=rdg_mask)

        # Get the maximum length of the double ridges
        _, _, length_max = filter_ridges(drdg)
        if length_max < 150:
            return LayerVariant.F_AUTOFLUO
        else:
            return LayerVariant.F_INFRARED

class CFIClassifier:
    """Layer variant classifier for CFI layers."""

    def __init__(self, layer: Layer):
        self.layer = layer
        self.n_images: int = len(layer.attributes['content'])

    def classify(self) -> list[str]:
        """
        Classfies Retina and ONH fundus images if the quality is adequate.

        Args:
            layer: The layer to be classified.

        Returns:
            The classified layer variants.
        """

        # Check inputs
        assert LayerVariant.F_CFI.value in self.layer.attributes['scanVariant'],\
            'Layer is not a CFI. Use another classifier.'
        assert self.n_images == 1,\
            f'Assumed all CFI had exactly one image, this one has {self.n_images}.'

        # Download the image and get ridges
        url_uuid = self.layer.uuid + '-' + str(0).zfill(4)
        url = self.layer.parent.get_url(uuid=url_uuid)
        f = download(url, out=None)  # TODO: Use Discovery.download()
        img = read_img(f)
        rdg = get_ridges(img, circle=True)
        rdg = apply_threshold(rdg=rdg, threshold=.006)

        # Get the variant if the quality is acceptable
        _, _, length_max = filter_ridges(rdg)
        if length_max > 5500:
            variant = SideClassifier.side_predict(img, rdg)
        else:
            variant = LayerVariant.F_LQ

        return [variant.value]

class SideClassifier:

    @classmethod
    def side_predict(cls, img: np.ndarray, ridges: np.ndarray) -> Literal[LayerVariant.F_RETINA, LayerVariant.F_ONH]:
        params = cls._get_parameters(img, ridges)
        pred = cls.clf.predict(np.reshape(params, (1, -1)))

        if pred[0]:
            return LayerVariant.F_RETINA
        else:
            return LayerVariant.F_ONH

    @classmethod
    def _get_parameters(cls, image: np.ndarray, ridges: np.ndarray) -> list[float]:
        f_arg = cls._get_relative_brightness(image)
        s_arg = cls._mean_brightness_center_ridges(ridges)

        return [f_arg, s_arg]

    @staticmethod
    def _get_relative_brightness(image: np.ndarray) -> float:
        half_height = int(image.shape[0] / 2)
        half_width = int(image.shape[1] / 2)

        gray_img = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        middle_slice = gray_img[half_height, :]
        y_filtered = savgol_filter(middle_slice, 151, 3)
        mean_brightness_of_center = np.mean(y_filtered[half_width - 75: half_width + 75])
        mean_brightness_of_image = np.median(y_filtered)

        relative_brightness = ((mean_brightness_of_center - mean_brightness_of_image) / mean_brightness_of_image) * 100

        return relative_brightness

    @staticmethod
    def _mean_brightness_center_ridges(rdg: np.ndarray) -> float:
        half_height = int(rdg.shape[0] / 2)
        half_width = int(rdg.shape[1] / 2)
        rr, cc = skimage.draw.disk((half_height, half_width),
                                   0.10 * min(half_width, half_height))
        mask = np.zeros(rdg.shape, dtype=np.uint8)
        mask[rr, cc] = 1
        rdg_mask = rdg * mask

        a = np.mean(rdg_mask)
        b = np.mean(rdg)

        return (a / b) * 100
