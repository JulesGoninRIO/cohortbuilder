"""
This module includes the methods for extracting new parameters from the scans.
"""

from typing import Tuple

import cv2
import numpy as np
from lxml import etree
import pathlib
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.ndimage.interpolation import rotate
from scipy.signal import argrelextrema
from skimage.filters import thresholding

from src.cohortbuilder.tools.definitions import RetinalLayer
from src.cohortbuilder.utils.helpers import recover_cor, rotate_cor
from src.cohortbuilder.utils.imageprocessing import bnd2curve, get_angle, mask2bnd, seg2mask


# TODO: Needs to be tested on more scans, has some bugs
# TODO: Combine relative thickness of GCL_IPL and PR_RPE at fovea location
# TODO: Combine fovea location of neighbouring scans
def detect_fovea(seg, info: dict, slope_thresh: float = 0.08,
    thickness_thresh: float = 0.050, separation_thresh: float = None
) -> Tuple[np.ndarray, float]:
    """
    Detects the position and the intensity (average of the slopes) of the fovea.

    Args:
        seg: Segmentation SVG read with `read_img <src.cohortbuilder.utils.helpers.read_img>`.
        info: Information dictionary of the corresponding oct.
        slope_thresh: Slope threshold for detecting fovea. Defaults to 0.08.
        thickness_thresh: Average thickness threshold for detecting fovea. Defaults to 0.050.
        separation_thresh: Minimum thickness threshold for detecting separation. Defaults to ``None`` .

    Returns:
        The coordinates of the fovea and its intensity.
        If ``fovea_cor`` and ``fovea_slope`` are ``None``, nothing has been detected.
        If ``fovea_cor`` is not ``None`` but fovea_slope is ``None``, it's not sure that the detected mask is fovea.

    .. seealso::
        `read_img <src.cohortbuilder.utils.helpers.read_img>`
            Reads an image from a file or a binary IO object.
    """

    # Get the thickness vectors
    tvs = calculate_tvs(seg, info, layers=['RNFL', 'GCL_IPL'])

    # Filter 1: RNFL layer should be thin
    # TODO: Tune the thickness threshold
    if thickness_thresh:
        for tv in [tvs['RNFL']]:
            marg = len(tv['vector']) // 10
            if np.asarray(tv['vector'][marg:-marg]).mean() > thickness_thresh:
                return None, None

    # Filter 2: There should not be any separation in RNFL or GCL_IPL
    # TODO: Tune the separation threshold
    # CHECK: Is this condition necessary?
    # CHECK: cohorts/AOSLO healthy images/AHOC_47/2021-11-01 8cdc0018-dd9a-425d-bc23-fb176ee2e101/0032436f-4edc-4c3d-9fb5-a4b4db33639f, idx = 48.
    if separation_thresh:
        for tv in [tvs['RNFL'], tvs['GCL_IPL']]:
            marg = len(tv['vector']) // 10
            if np.any(tv['vector'][marg:-marg] < separation_thresh):
                return None, None

    # Get the mask and its shape
    mask = seg2mask(seg, layers='RNFL')
    shape = mask.shape

    # Rotate the mask and get the upper boundary
    angle = np.degrees(get_angle(mask))
    mask = rotate(mask, angle=angle, order=0)
    upper, _ = mask2bnd(mask)
    h, w = bnd2curve(upper)

    # Get smoothed spline of the upper boundary
    h_ = UnivariateSpline(w, h, k=2)(w)

    # Get the extrema around the fovea and their slopes
    fovea_cor = None
    fovea_slope = None
    extrema = np.union1d(argrelextrema(h_, np.greater), argrelextrema(h_, np.less))
    # In case of only one minimum
    if len(extrema) == 1 and extrema == argrelextrema(h_, np.greater)[0]:
        fovea_cor = (h[extrema[0]], w[extrema[0]])
    # In case of the pattern of 3 extrema
    else:
        for i in range(len(extrema)-2):
            ids = extrema[i:i+3]
            slopes = np.diff(-h_[ids]) / np.diff(w[ids])
            if (
                slopes[0] < 0 and slopes[1] > 0\
                and np.abs(slopes).mean() > (fovea_slope if fovea_slope else slope_thresh)
            ):
                fovea_cor = (h[ids[1]], w[ids[1]])
                fovea_slope = np.abs(slopes).mean().item()

    # Recover the fovea coordinates in original frames
    if fovea_cor:
        fovea_cor = recover_cor(
            cor=fovea_cor,
            angle=angle,
            shape=shape,
        )

    return fovea_cor, fovea_slope

# TODO: Tune window_size and k
# TODO: Implement Dynamic Time Warping for pairing pixels at the upper and lower boundary
def calculate_cvi(oct: np.ndarray, seg: etree._Element, fovea: np.ndarray = None,
    window_size: int = 35, k: float = 0.001, prec: str = ''
) -> dict:
    """
    Calculates the CVI (choroidal vascularity index) vector and its area-averaged
    value. The vector values are calculated by the ratio of vascular areas to the
    total choroidal areas (vascular and stromal areas).

    Args:
        oct: OCT file read by `read_img <src.cohortbuilder.utils.helpers.read_img>`.
        seg: Segmentation SVG file read by `read_img <src.cohortbuilder.utils.helpers.read_img>`.
        fovea: Detected fovea using `detect_fovea <src.cohortbuilder.tools.parameters.detect_fovea>`.
            If passed, the center of the vector will be returned.
        window_size: Window size for Niblack local threshold. Defaults to 35.
        k: k-value for Niblack local threshold. Defaults to 0.001.

    Returns:
        The CVI vector, its center index, and average CVI.

    .. seealso::
        `read_img <src.cohortbuilder.utils.helpers.read_img>`
            Reads an image from a file or a binary IO object.
    """

    # Get rotation angle
    angle = np.degrees(get_angle(seg2mask(seg)))

    # Get the mask and rotate
    shape = oct.shape
    oct = rotate(oct, angle=angle, order=0)
    mask = seg2mask(seg, layers='CC_CS')
    mask = rotate(mask, angle=angle, order=0)

    # Get the fovea coordinates if passed
    fovea_cor = fovea[0] if fovea else None
    # Get the coordinates of the rotated fovea
    if fovea_cor is not None:
        _, w_fovea = rotate_cor(
            fovea_cor,
            angle=angle,
            shape=shape,
        )

    # Mask vascular areas in OCT
    oct_smoothed = thresholding.threshold_niblack(oct, window_size=window_size, k=k)
    vascular = np.logical_and(oct > oct_smoothed, mask)

    # Calculate CSI and CVI vectors
    with np.errstate(invalid='ignore'):
        csi = vascular.sum(axis=0) / mask.sum(axis=0)
    cvi = [None if np.isnan(csi) else float(format(1-csi.item(), prec)) for csi in csi]
    center = w_fovea if (fovea_cor is not None) else None

    # Calculate average CSI and CVI
    csi_avg = vascular.sum() / mask.sum()
    cvi_avg = None if np.isnan(csi_avg) else float(format(1 - csi_avg.item(), prec))

    return {'vector': cvi, 'center': center, 'average': cvi_avg}

# TODO: Implement Dynamic Time Warping for pairing pixels at the upper and lower boundary
def calculate_tvs(seg: etree._Element, info: dict, layers: list[str] = None,
    fovea: tuple = None, prec: str = ''
) -> dict[str, dict]:
    """
    Calculates the thickness vectors of the input layers; using the given segmentation file.

    Args:
        seg: Segmentation SVG read with `read_img <src.cohortbuilder.utils.helpers.read_img>`.
        info: Information dictionary of the corresponding OCT.
        layers: List of desired layers. If ``None``, all layers will be considered. Defautls to ``None``.
        fovea: Detected fovea using `detect_fovea <src.cohortbuilder.tools.parameters.detect_fovea>`.
            If passed, the center of the vector will be returned.

    Returns:
        The thickness vector for each specified layer and its center index.

    .. seealso::
        `read_img <src.cohortbuilder.utils.helpers.read_img>`
            Reads an image from a file or a binary IO object.
    """

    # Check inputs
    if not layers:
        layers = [layer for layer in RetinalLayer.get_names() if layer != 'BG']
    assert all(layer in RetinalLayer.get_names() for layer in layers)

    # Get the rotation angle and mask shape
    mask = seg2mask(seg)
    shape = mask.shape
    angle = np.degrees(get_angle(mask))

    # Get the fovea coordinates if passed
    fovea_cor = fovea[0] if fovea else None
    # Get the coordinates of the rotated fovea
    if fovea_cor is not None:
        _, w_fovea = rotate_cor(
            fovea_cor,
            angle=angle,
            shape=shape,
        )

    # NOTE: A basic calculation shows that the spacings do not change with rotation
    # CHECK: Is this the right spacing to use?
    spacing = info['spacing'][2]

    # Calculate layer thicknesses
    tvs = dict()
    for layer in layers:
        # Get the layer mask
        mask = seg2mask(seg, layers=layer)
        # Skip if the mask is empty
        if not mask.any():
            tvs[layer] = None
            continue

        # Rotate and build boundary masks
        mask = rotate(mask, angle=angle, order=0)
        upper, lower = mask2bnd(mask)
        h_upper, w_upper = bnd2curve(upper)
        h_lower, w_lower = bnd2curve(lower)

        # TODO: Include None values if no overlap (like CVI)
        # Caclulate the intersection of curves
        intersection, ids_upper, ids_lower = np.intersect1d(w_upper, w_lower, return_indices=True)
        # Skip the layer if no intersection
        if len(intersection) == 0:
            tvs[layer] = None
            continue

        # Calculate the thickness in intersection
        tv = (h_lower[ids_lower] - h_upper[ids_upper]) * spacing
        center = int(w_fovea - max(ids_upper[0], ids_lower[0]))\
            if (fovea_cor is not None) else None

        # Make list and apply the precision
        tv = [float(format(val, prec)) for val in tv]
        # Store the tv
        tvs[layer] = {'vector': tv, 'center': center}

    return tvs


ws = 45
er_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
di_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

gamma = 0.9
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

def calculate_pachy_area(oct, seg, spacings):
    # Get choroidal layer mask
    mask = seg2mask(seg, ['CC_CS']).astype(int)

    # Normalize the OCT image using a lookup table
    oct_img = (oct/oct.max() * 255.0).astype('uint8')
    oct_img = cv2.LUT(oct_img, lookUpTable)

    # Mask vascular areas in OCT
    thresh_niblack = thresholding.threshold_niblack(oct_img, window_size=ws, k=0.01)
    binary_niblack = oct_img > thresh_niblack
    binary_niblack = (1-binary_niblack) * mask

    # Use morphological operators to find large vascular area (pachyvessels) from mask
    binary = cv2.medianBlur(binary_niblack.astype('uint8'), 9)
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, er_kernel)
    binary = cv2.medianBlur(binary.astype('uint8'), 15)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, di_kernel)
    # In case no large vessels were detected return zero
    if binary.sum() == 0:
        return [0]
    # Segment the mask from OCT
    masked = oct_img * binary
    # Get average of pixel intensities of the pachyvessels.
    # it is used to get more accurate segmentation of pachyvessels
    # using their pixels intensities (since vascular pixels tend to be darker)
    avg = masked[binary != 0].mean()
    masked[masked == 0] = 255
    masked = (masked < 1.5 * avg) * 255
    masked = cv2.medianBlur(masked.astype('uint8'), 9)
    # In case no large vessels were detected return zero
    if masked.sum() == 0:
        return [0]
    # Canny edge detaction to detect the edges of the segmentation
    # It is used to calculate the area of pachyvessels
    edged = cv2.Canny(masked, threshold1=1, threshold2=1)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = []
    for i in range(len(contours)):
        # Calculates are for each pachyvessel
        area = cv2.contourArea(contours[i])
        # The area is calculated in number of pixels. This line uses the spacing information
        # to scale it into the right metric
        area = area * spacings[2] * spacings[3]
        areas.append(area)

    # Sort the result list so the largest area would be tha last one on the list
    areas = sorted(areas)
    return areas if areas != [] else [float('nan')]


n_row = 50
thresh = 0.65
kernel_size = 15
kpad = kernel_size//2

def mapp(col):
    """ This function is used to squash the layer and normalize the pixel values and match them with 
        the template's mean and std. This function is applied per column of the image
    """
    # Selects only the segmented part in each column.
    col_ = col[col>0]

    # Normalizes the column and matchs it with the template's mean and std.
    col_ = (col_ - col_.mean())/(col_.std() + 1e-5) * template_normed.std() + template_normed.mean()

    # Fixes the size of segmented part of the column to n_row to squash the layer.
    if col_.size < n_row:
        res = np.zeros(n_row)
        pad = (n_row - col_.size)//2
        res[pad:pad+col_.size] = col_
    else:
        pad = (col_.size - n_row)//2
        res = col_[pad:pad+n_row]
    return res

def cutout_zero_cols(stacked):
    """ Cuts out empty column edges after rotation.
    """
    sum_per_col = np.sum(stacked[:,:,0], axis=0)
    idx = np.argwhere(sum_per_col <= 5).squeeze()
    if idx.size > 0:
        if idx.size == 1:
            idx = np.array([0, idx, sum_per_col.shape[0]])
        else:
            idx = np.concatenate(([0],idx,[sum_per_col.shape[0]]))
        max_dist_pair = (0, None)
        for i in range(1, len(idx)):
            if idx[i] - idx[i-1] > max_dist_pair[0]:
                max_dist_pair = (idx[i] - idx[i-1], i)
        i = max_dist_pair[1]
        return stacked[:, idx[i-1]+10:idx[i]-10, :]
    else:
        return stacked

def rotate_stacked(stacked, theta):
    """ Rotates a stack/batch of images by the given theta.
    """
    rotated = ndimage.rotate(stacked, theta, reshape=True)
    rotated[:,:,:2] = np.round(rotated[:,:,:2])
    rotated[:,:,2][rotated[:,:,2] < 0] = 0
    rotated = cutout_zero_cols(rotated)

    return rotated

def apply_matching(scan, template, thickness):
    """ Squashes the layer in scan and applies OpenCV template matching to detec mismatches and disruptions.
    """
    # Calls mapp function per each column to squash the layer
    flattened = np.apply_along_axis(mapp, 0, scan).squeeze()
    # Resizes the template height to match it with the thickness of the layer.
    adjusted_temp = ndimage.zoom(template[kpad:-kpad], thickness/(template.shape[0]))
    # Extends the template horizontally
    adjusted_temp = np.stack([adjusted_temp]*kernel_size, axis=-1)
    # Apply template matching
    res = cv2.matchTemplate(flattened.astype('float32'), adjusted_temp.astype('float32'), cv2.TM_SQDIFF_NORMED)
    # Normalize the result
    return (res - res.min())/(res.max()-res.min())

kernel = np.ones((kernel_size, 1), np.uint8)
template_normed = np.load(pathlib.Path(__file__).parent / 'disruption_template.npy')

def calculate_disruption(oct, seg):
    # Gets the IZ, EZ and RPE mask from segmentation file.
    # This mask is used to segment these layers for finding disruption
    mask = seg2mask(seg, ["PR_RPE"]).astype('uint8')
    # Gets the SRF and PED mask from segmentation file.
    # This mask is used to detect artifacts caused by srf or ped
    anomaly_mask = seg2mask(seg, ['SRF', 'PED']).astype('uint8')
    # Get the angle of rotation to make the layer horizontal
    theta = get_angle(mask)
    # Stack the masks and the scan to rotate them
    stacked = np.stack([mask, anomaly_mask, oct], axis=-1)
    rotated = rotate_stacked(stacked, theta)
    rotated_mask = rotated[:,:,0]
    # Dilate the mask just to have better coverage of IZ, EZ and RPE layers
    rotated_mask_dilated = cv2.dilate(rotated_mask, kernel, iterations=1)
    # Using dilate can also cover some parts of SRF or PED. This line uses
    # anomlay_mask to remove those coverages
    rotated_mask_dilated = rotated_mask_dilated  - rotated_mask_dilated * rotated[:,:,1]
    # Segment the layers mask from the oct scan
    segmented_scan = rotated[:,:,2] * rotated_mask_dilated
    # Get median thickness of the layers. Will be used in template matching for resizing
    # the template size. Also will be used as another metric to find disruptions.
    thickness = np.median(rotated_mask.sum(axis=0))
    # Apply template matching
    matching_score = apply_matching(segmented_scan, template_normed, thickness)
    # Use thickness as an another metric to find disruptions
    thickness_variance_score = (thickness - rotated_mask.sum(axis=0))/thickness
    # Sum up template matching score and thicness score to get the final score
    final_score = matching_score.mean(axis=0) + 0.01 * thickness_variance_score[kpad:-kpad]
    # This part reduces scores by a threshold (to remove bias). Then sets negative values to zero.
    values = final_score - thresh
    max_ = values.max()
    if max_ < 0:
        max_ = 0
    values = np.clip(values, a_min=0, a_max=max_)

    # Sum up the values to get the final score
    return values.sum()


def calculate_cvi_line(oct, seg, window_size=35, k=0.001):
    # Load the choroidal layer mask from the segmentation svg
    mask = seg2mask(seg, layers='CC_CS')
    # Mask vascular areas in OCT
    oct_smoothed = thresholding.threshold_niblack(oct, window_size=window_size, k=k)
    vascular = np.logical_and(oct > oct_smoothed, mask)
    # Calculate CVI
    csi_avg = vascular.sum() / mask.sum()
    cvi_avg = None if np.isnan(csi_avg) else float(1 - csi_avg.item())

    return cvi_avg


def get_reference_coordinates(dicom, _type):
    """Extracted the reference coordinates tag from the Dicom file
        - coordinates of both edges of the first and the last bscan
        - eye fundus of the patient as reference
        - locate at : PerFrameFunctionalGroupsSequence -> OphthalmicFrameLocationSequence-> ReferenceCoordinates
        - A first bscan, B last bsan. For each two point 1 and 2 with two coordinates x and y.
    Args:
        dicom (_type_): dicom file of the OCT
        _type (str): type of the OCT

    Returns:
        tupple: ([Ax1, Ay1, Ax2, Ay2], [Bx1, By1, Bx2, By2])
    """
    PerFrameFunctionalGroupsSequence = dicom.get("PerFrameFunctionalGroupsSequence")
    if PerFrameFunctionalGroupsSequence is not None:
        if _type == "cube":
            # Extract first bscan coordinates
            coord1 = PerFrameFunctionalGroupsSequence[0].get("OphthalmicFrameLocationSequence")[0].get("ReferenceCoordinates")
            # Extract middle bscan coordinates
            coord2 = PerFrameFunctionalGroupsSequence[len(PerFrameFunctionalGroupsSequence)//2].get("OphthalmicFrameLocationSequence")[0].get("ReferenceCoordinates")
            # Extract last bscan coordinates
            coord3 = PerFrameFunctionalGroupsSequence[-1].get("OphthalmicFrameLocationSequence")[0].get("ReferenceCoordinates")
            return (coord1, coord2, coord3)
        if _type == "line":
            coord1 = PerFrameFunctionalGroupsSequence[0].get("OphthalmicFrameLocationSequence")[0].get("ReferenceCoordinates")
            return coord1
    else:
        return None


def get_position_tag(dicom):
    """extract ONH and Macula tag
    Args:
        dicom_file (_type_): dicom_file

    Returns:
        int: 1: macula, 0: ONH, -1: no tag
    """
    shared_functional_groups_sequence = dicom.get('SharedFunctionalGroupsSequence')
    if shared_functional_groups_sequence is not None:
        frame_anatomy = shared_functional_groups_sequence[0].get('FrameAnatomySequence')
        if frame_anatomy is not None and frame_anatomy[0].AnatomicRegionSequence is not None:
            anatomic_region_sequence = frame_anatomy[0].AnatomicRegionSequence[0]
            if anatomic_region_sequence.CodeMeaning == 'Retina':
                return 1
            elif anatomic_region_sequence.CodeMeaning == 'Optic nerve head':
                return 0
            else: return -1
