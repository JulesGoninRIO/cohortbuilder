"""
This module includes the image processing utility functions.
"""

import io
import itertools
from typing import Tuple, Union

import cv2
import numpy as np
import skimage
from lxml import etree
from PIL import Image
from reportlab.graphics import renderPM, shapes
from skimage.color import rgb2gray
from skimage.filters.ridges import sato
from svglib.svglib import SvgRenderer
# from joblib import delayed, Parallel
# from multiprocessing import cpu_count
from tqdm.auto import tqdm
from imageio.v2 import imread
import json
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import disk
from skimage.segmentation import find_boundaries
from skimage.filters import threshold_minimum
import scipy.ndimage as ndi
from joblib import Parallel, delayed
from multiprocessing import cpu_count

EIGHT_DISK = disk(8)
SIXTEEN_DISK = disk(16)

from src.cohortbuilder.tools.definitions import RetinalLayer

# NOTE: correct libraries versions are reportlab==3.6.11 and svglib==1.4.1 (the recent versions do not work)


def draw2img(drawing: shapes.Drawing, **kwargs) -> io.BytesIO:
    """
    Creates a pixmap and draw drawing to it then return it as a BytesIO.

    .. seealso::
        `Drawing <https://pyng.tech/docs/reportlab/graphics/shapes.m.html#reportlab.graphics.shapes.Drawing>`_
            Drawing objects in the reportlab documentation.
        `renderPM.drawToFile <https://pyng.tech/docs/reportlab/graphics/renderPM.m.html#reportlab.graphics.renderPM.drawToFile>`_
            Base method in the reportlab documentation.
    """

    img = io.BytesIO()
    renderPM.drawToFile(drawing, img, **kwargs)

    return img

def seg2mask(svg: etree._Element, layers: Union[str, list[str], None] = None) -> np.ndarray:
    """Returns the mask of the specified layers in a segmentation.

    Args:
        svg: Segmentation svg.
        layers: The layer(s) to be masked.
            If ``None``, it will return all the layers except the background. Defaults to ``None`` .

    Returns:
        The mask array of the specified layers.

    .. seealso::
        `mask2bnd <src.cohortbuilder.utils.imageprocessing.mask2bnd>`
            Returns the boundary of a mask.
        `bnd2curve <src.cohortbuilder.utils.imageprocessing.mask2bnd>`
            Extracts the indices of the boundary pixels.
    """

    # Interpret the layers
    if not layers:
        layers = [layer for layer in RetinalLayer.get_names() if layer != 'BG']
    elif isinstance(layers, str):
        layers = [layers]

    # Check layer names
    assert all(layer in RetinalLayer.get_names() for layer in layers),\
        f'Layers should be a subset of: {RetinalLayer.get_names()}'

    # Convert inputs
    layer_tags = [RetinalLayer[layer].value['tag'] for layer in layers]

    # Make other layers transparent
    for p in svg[1]:
        if p.attrib['class'] not in layer_tags:
            p.attrib['fill-opacity'] = '0'  # transparent
        if p.attrib['class'] in layer_tags:
            p.attrib['fill-opacity'] = '1'  # opaque

    svgRenderer = SvgRenderer('')
    drawing = svgRenderer.render(svg)

    img = draw2img(drawing, fmt='PNG')

    mask = np.array(Image.open(img))
    mask = np.round(np.amax(1 - mask/255, axis=2))
    mask = np.array(mask, dtype=bool)

    return mask

def mask2bnd(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a mask of the boundaries of a given mask.

    Returns:
        Masks of the upper and the lower boundaries of the input mask.

    .. seealso::
        `seg2mask <src.cohortbuilder.utils.imageprocessing.seg2mask>`
            Returns the mask of a segmented layer.
        `bnd2curve <src.cohortbuilder.utils.imageprocessing.mask2bnd>`
            Extracts the indices of the boundary pixels.
    """

    # Instantiate the masks
    mask_upper = np.zeros(shape=mask.shape, dtype=bool)
    mask_lower = np.zeros(shape=mask.shape, dtype=bool)

    # Mark lower and upper boundaries on the masks
    for w in range(mask.shape[1]):
        col = mask[:, w]
        if col.any():
            mask_upper[np.where(col)[0].min(), w] = True
            mask_lower[np.where(col)[0].max(), w] = True

    return mask_upper, mask_lower

def bnd2curve(bnd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the indexes of a boundary mask.

    Args:
        bnd: Boundary mask.

    .. seealso::
        `seg2mask <src.cohortbuilder.utils.imageprocessing.seg2mask>`
            Returns the mask of a segmented layer.
        `mask2bnd <src.cohortbuilder.utils.imageprocessing.mask2bnd>`
            Returns the boundary of a mask.
    """

    heights, widths = np.where(bnd)
    args = widths.argsort()
    heights = heights[args]
    widths = widths[args]

    return heights, widths

def get_angle(mask: np.ndarray) -> float:
    """Returns the average slope of a given mask by fitting a line to it."""

    upper, lower = mask2bnd(mask)
    h_upper, w_upper = bnd2curve(upper)
    h_lower, w_lower = bnd2curve(lower)

    w_mid, ids_upper, ids_lower = np.intersect1d(w_upper, w_lower, return_indices=True)
    h_mid = (h_lower[ids_lower] + h_upper[ids_upper]) / 2

    coeffs = np.polyfit(w_mid, h_mid, deg=1)[::-1]
    slope = coeffs[1]

    return slope

def is_round(img: np.ndarray, mrg: int = 10) -> bool:
    """
    Checks if the corners of the image are dark.

    Args:
        img: Image read by read_img.
        mrg: The margin size in pixels. Defaults to 10.

    Returns:
        Wether the image is round.
    """

    a = img[:mrg, :mrg]
    b = img[:mrg, -mrg:]
    c = img[-mrg:, :mrg]
    d = img[-mrg:, -mrg:]

    return np.all(a < 2) and np.all(b < 2) and np.all(c < 2) and np.all(d < 2)

def get_ridges(img: np.ndarray, sigmas: range = range(1, 10, 2), black: bool = True, circle: bool = False) -> np.ndarray:
    """
    Applies the sato filter to get the ridges in the image. Converts RGB to grayscale.

    Args:
        img: Input image read by read_img
        sigmas: Sigmas used as scales of filter. Defaults to range(1, 10, 2).
        black: When True, the filter detects black ridges;
            when False, it detects white ridges. Defaults to ``True``.
        circle: When True, applies a circle mask in the center of the image.
            Defaults to ``False``.

    Returns:
        Ridges of the image.
    """

    # Convert RGB to grayscale
    if len(img.shape) > 2:
        img = rgb2gray(img)

    # Apply the filter
    rdg = sato(img, sigmas=sigmas, black_ridges=black, mode='reflect')

    # Apply the centered circle mask
    if circle:
        c = (
            rdg.shape[0] // 2,
            rdg.shape[1] // 2,
        )
        h, w = skimage.draw.disk(c, .95 * min(c))
        mask = np.zeros(rdg.shape, dtype=bool)
        mask[h, w] = True
        rdg[np.where(~mask)] = 0

    return rdg

def filter_ridges(rdg: np.ndarray, length_threshold: float = None) -> tuple:
    """
    Keeps the long ridges, calculates the total length and the maximum length of them.

    Args:
        rdg: Input ridges (``uint8``) or ridges mask (``boolean``).
        length_threshold: If given, only contours longer
            than this threshold would be kept and passed as the first result.
            Total lengths would also be the total length of the long contours.
            Defaults to ``None`` .

    Returns:
        A tuple containing: Mask of long ridges, total length of the long ridges,
        and maximum length of the ridges.
    """

    # Convert boolean to uint8
    if rdg.dtype.type is np.bool_:
        rdg = rdg.astype(np.uint8) * 255

    # Determine contour of all blobs found
    contours, _ = cv2.findContours(
        image=rdg,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_NONE,
    )

    # Calculate the lengths of the contours
    lengths = [cv2.arcLength(cnt, True) for cnt in contours]
    lengths = np.array(lengths, dtype=float)
    length_max = np.max(lengths) if len(lengths) else 0

    if length_threshold:
        length_total = np.sum(lengths[np.where(lengths > length_threshold)])
        mask = np.zeros(shape=rdg.shape, dtype=np.uint8)
        cv2.drawContours(
            image=mask,
            contours=[contours[i] for i in range(len(lengths)) if lengths[i] > length_threshold],
            contourIdx=-1,
            color=255,
            thickness=-1,
        )
        mask.dtype = bool
    else:
        mask = np.zeros(shape=rdg.shape, dtype=bool)
        mask[np.where(rdg)] = True
        length_total = np.sum(lengths)

    return mask, length_total, length_max

# TODO: Remove this function (after replacing its usages with the same line)
def apply_threshold(rdg: np.ndarray[float], threshold: float) -> np.ndarray:
    """Applies a threshold on a ridges mask and returns it as ``uint8``."""

    return  np.array(rdg > threshold, dtype=np.uint8) * 255

# TODO: Unify with in_double_ridge
def rdg2drdg(mask: np.ndarray) -> np.ndarray:
    """
    Returns a mask of pixels that are in a double ridge.

    Args:
        mask: Mask of the ridges.

    Returns:
        Mask of the double ridges.
    """

    h, w = mask.shape
    rdg_double = np.zeros(shape=mask.shape, dtype=bool)
    for i, j in itertools.product(range(h), range(w)):
        if in_double_ridge(rdg=mask, ids=(i, j)):
            rdg_double[i, j] = True

    return rdg_double

def in_double_ridge(rdg: np.ndarray, ids: Tuple[int, int], marg: int = 20) -> bool:
    """
    Determines wether a pixel is located in a double ridge.

    Args:
        rdg: Mask of the ridges.
        ids: Tuple of indexes of the pixel.
        marg: Margin to look for another ridge. Defaults to 20.

    Returns:
        True if the pixel is in a double ridge.
    """

    # Read the inputs and check the pixel
    i, j = ids
    if not rdg[i, j]:
        return False

    # Define the neighbouring indexes
    idx_u = max(i - marg, 0)
    idx_d = min(i + marg, rdg.shape[0] - 1)
    idx_l = max(j - marg, 0)
    idx_r = min(j + marg, rdg.shape[1] - 1)

    # TODO: Improve the logic. There could be some corner cases!
    # Define the neighbours
    neighbours = [
        rdg[idx_u:i, j].astype(bool),
        rdg[i, idx_l:j].astype(bool),
        rdg[i:idx_d, j].astype(bool),
        rdg[i, j:idx_r].astype(bool),
    ]

    # Check the neighbours one by one
    current = True
    for i, neighbour in enumerate(neighbours):
        lengths = list()
        length = 0
        for n in neighbour:
            if n is not current:
                current = not current
                lengths.append(length)
                length = 0
            else:
                length += 1

        if (
            len(lengths) >= 3
            # and (ls[0] <= 4 and ls[2] <= 4)
        ):
            return True

    return False

# EDTRS grid creation, overlay, and statistics computation
# --------------------------------------------------------

def get_clean_oct_segmentation(location: str, segmentation_index: int = 1, filled_segmentation: bool = False, show_progress: bool = False) -> dict[str, np.array]:
    '''
    Extracts clean `np.array` cubes from SVG segmentations downloaded from Discovery, located at `location`.
    These contain continuous lines are are much easier to work with downstream.

    Arguments:
     - `location`: the base path towards the acquisition on disk. This folder contains both a subfolder for OCTs and segmentations.
     - `segmentation_index`: the index given by Discovery of the segmentation to process (eg. labelled segmentation_01, segmentation_02, etc.)
     - `filled_segmentation`: whether to return the filled-out segmentation (boolean mask), rather than just the borders
     - `show_progress`: whether to create a TQDM bar for progression.
    
    Returns: a dict which is of type layer name -> np.array cube (same dimension as OCT images). Key "all" provides all segmentations together.

    '''
    octs = tuple(location.rglob('oct/*/*.jpg'))
    template_img = np.asarray(imread(octs[0]))
    
    def get_slice_segmentation(svg_location: str) -> np.array:
        '''
        Internal function
        '''
        layer_selection = RetinalLayer.get_names()
        layer_selection = [[x] for x in layer_selection] + [tuple(layer_selection)]
        output_dict = dict()
        with open(svg_location, 'r') as f:
            svg_as_string = f.read()

        for idx in range(len(layer_selection) - 1): # Iterate over possible selections of layers from segmentation (inclusion)

            slice_output_segmentation = np.zeros_like(template_img).T
            
            selected_layers = layer_selection[idx]
            svgobj = etree.fromstring(svg_as_string, parser=etree.XMLParser()) # Re-initalise the tree at each iteration

            if 'BG' in selected_layers and len(selected_layers) == 1:
                continue
            layer_tags = [RetinalLayer[layer].value['tag'] for layer in selected_layers]

            # Make other layers transparent
            for p in svgobj[1]:
                if p.attrib['class'] not in layer_tags:
                    p.attrib['fill-opacity'] = '0'  # transparent
                if p.attrib['class'] in layer_tags:
                    p.attrib['fill-opacity'] = '1'  # opaque

            svgRenderer = SvgRenderer('')
            drawing = svgRenderer.render(svgobj)

            img = draw2img(drawing, fmt='PNG')

            mask = np.array(Image.open(img))
            assert np.array_equal(mask[..., 0], mask[...,1])
            assert np.array_equal(mask[..., 1], mask[...,2])
            mask = mask[...,0]
            for i,col in enumerate(mask.T):
                slice_output_segmentation[i, np.where(np.diff(col))[0]] = 1 # Put in a 1 when there is a jump in values: border in segmentation
            output_dict[layer_selection[idx][0]] = slice_output_segmentation.T > 0 if not filled_segmentation else mask
        output_dict['all'] = np.stack(list(output_dict.values())).sum(axis=0) > 0 # Add into 3D array, sum along first dimension, then re-binarise
    
        return output_dict
    
    svgs = tuple(location.rglob(f'children/segmentation_*{segmentation_index}/*.svg'))
    output_segmentation = [get_slice_segmentation(svg) for svg in tqdm(svgs, desc='Getting clean Discovery OCT segmentations', disable = not show_progress, leave=False)]
    # output_segmentation = tuple(
    #     tqdm(
    #         Parallel(n_jobs = cpu_count(), return_as='generator', verbose=0)(delayed(get_slice_segmentation)(svg) for svg in svgs),
    #         total=len(svgs),
    #         desc='Getting clean Discovery OCT segmentations',
    #         disable = not show_progress
    #     )
    # )

    final_dict = dict()
    if output_segmentation:
        for key in output_segmentation[0].keys():
            final_dict[key] = np.stack([x[key] for x in output_segmentation])

    return final_dict

# Heavy-lifters of the scan stat computation code
# -----------------------------------------------

def get_oct_scale_information(location: str) -> Tuple[float, float]:
    '''
    Extracts the scale information from the OCT images in the Discovery acquisition.

    Arguments:
     - `location`: the base path towards the acquisition on disk. This folder contains both a subfolder for OCTs and segmentations.
    
    Returns: a list of scales for pixels -> milimeters for [depth, channels, height, width]
    '''

    with open(tuple(location.rglob('oct/*/info.json'))[0], 'r') as f:
        oct_info = json.load(f)
    return oct_info['scale']

def rebuild_thickness_map(clean_segmentations: dict[str, np.array], scaling_factor: int = None, show_progress: bool = False) -> Tuple[dict[str, list], int]:
    '''
    Rebuilds the thickness maps from the clean segmentations, by summing the number of pixels in each column of the segmentation, on each slice.

    Arguments:
        - `clean_segmentations`: the clean segmentations, as returned by `get_clean_oct_segmentation`
        - `scaling_factor`: the factor by which to scale the thickness maps. If not provided, it will be inferred from the segmentation shape.
        - `show_progress`: whether to show a TQDM bar for progression.
    
    Returns: a dict of layer name -> thickness map, and the scaling factor used.
    '''
    layer_thicknesses = dict()
    for layer_name, segmentation in tqdm(clean_segmentations.items(), desc='Rebuilding thickness maps', disable = not show_progress, leave=False):
        if layer_name == 'all':
            continue
        scaling_factor = segmentation.shape[2] // segmentation.shape[0] if scaling_factor is None else scaling_factor
        enface_view = np.zeros((segmentation.shape[0] * scaling_factor, segmentation.shape[2]))

        for i, sli in enumerate(segmentation):
            sli_fixed = (sli == 0).T # Invert
            for j, column in enumerate(sli_fixed):
                enface_view[np.arange(scaling_factor) + i * scaling_factor, j] += column.sum()
        
        enface_view = np.flip(enface_view, axis=0) # Flip so that image coordinates correspond to visualisation
        layer_thicknesses[layer_name] = gaussian_filter(enface_view, sigma=scaling_factor // 1.8)
        # layer_thicknesses[layer_name] = enface_view
    
    return layer_thicknesses, scaling_factor

# Visualisation functions
# -----------------------

def create_circular_line(h: int, w: int, center: tuple = None, radius: int = None) -> np.ndarray:
    '''
    Create a circular line mask. This is a mask that is 1 pixel wide and follows the circumference of a circle.
    (Mostly used for visualisation purposes)

    Args:
        `h`: height of the mask
        `w`: width of the mask
        `center`: center of the circle
        `radius`: radius of the circle
    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.isclose(dist_from_center, radius, atol=1)
    return mask

def quadrant_line(h: int, w: int, center: tuple = None, quadrant: str = 'top') -> np.ndarray:
    '''
    Create a line mask that is 1 pixel wide and follows the circumference of a quadrant of a circle.
    (Mostly used for visualisation purposes)

    Args:
        `h`: height of the mask
        `w`: width of the mask
        `center`: center of the circle
        `radius`: radius of the circle
    '''

    if center is None: # use the middle of the image
        center = ((w/2), (h/2))
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = y - center[1], x - center[0]

    if quadrant == 'bottom':
        return np.isclose(x, np.abs(y), rtol=1e-3)
    if quadrant == 'top':
        return np.isclose(-x, np.abs(y), rtol=1e-3)
    if quadrant == 'right':
        return np.isclose(np.abs(x), y, rtol=1e-3)
    if quadrant == 'left':
        return np.isclose(np.abs(x), -y, rtol=1e-3)

def overlay_edtrs_lines(thickness_map: np.array, centre_point: np.array, xscale: int) -> np.ndarray:
    '''
    Create a line that divides the image into sectors, following the EDTRS grid.
    The sectors are every 90 degrees, starting at 45 degree angles. Delimited from 1, 3, 6mm diameter, and centre is not divided.

    Args:
        `thickness_map`: the thickness map to overlay the grid on
        `centre_point`: the centre of the image
        `xscale`: the scaling factor for the image

    Returns: a binary mask of the lines
    '''


    # Sectors are every 90 degrees, starting at 45 degree angles. Delimited from 1, 3, 6mm diameter, and centre is not divided
    sector_thickness = [list] * 9

    one_mm_in_px = 1/xscale
    one_mm_in_px = one_mm_in_px / 2 # Radius -> Diameter
    circ_mask = create_circular_line(*thickness_map.shape, center=centre_point, radius=one_mm_in_px)
    top_quadrant_mask = quadrant_line(*thickness_map.shape, center=centre_point, quadrant='top')
    right_quadrant_mask = quadrant_line(*thickness_map.shape, center=centre_point, quadrant='right')
    bottom_quadrant_mask = quadrant_line(*thickness_map.shape, center=centre_point, quadrant='bottom')
    left_quadrant_mask = quadrant_line(*thickness_map.shape, center=centre_point, quadrant='left')

    sector_thickness[0] = circ_mask

    three_mm_in_px = one_mm_in_px * 3
    circ_mask = create_circular_line(*thickness_map.shape, center=centre_point, radius=three_mm_in_px)
    sector_thickness[1] = np.bitwise_or(circ_mask, top_quadrant_mask)
    sector_thickness[2] = np.bitwise_or(circ_mask, right_quadrant_mask)
    sector_thickness[3] = np.bitwise_or(circ_mask, bottom_quadrant_mask)
    sector_thickness[4] = np.bitwise_or(circ_mask, left_quadrant_mask)

    six_mm_in_px = one_mm_in_px * 6
    circ_mask = create_circular_line(*thickness_map.shape, center=centre_point, radius=six_mm_in_px)
    sector_thickness[5] = np.bitwise_or(circ_mask, top_quadrant_mask)
    sector_thickness[6] = np.bitwise_or(circ_mask, right_quadrant_mask)
    sector_thickness[7] = np.bitwise_or(circ_mask, bottom_quadrant_mask)
    sector_thickness[8] = np.bitwise_or(circ_mask, left_quadrant_mask)

    return np.stack(sector_thickness).sum(axis=0) > 0

# Visualisation function, high-level interface for external use
# --------------------------------------------------------------

def generate_thickness_map_visualisation(layer_thicknesses: dict[str, np.array], oct_info: dict[str, list], scaling_factor: int, centre_point: np.array, output_file_path: Union[str, Path]) -> None:
    '''
    Generate a visualisation of the thickness maps, with the EDTRS grid overlaid.

    Arguments:
        - `layer_thicknesses`: the thickness maps for each layer (output of `rebuild_thickness_map`)
        - `oct_info`: the scale information for the OCT images
        - `scaling_factor`: the scaling factor used to generate the thickness maps
        - `centre_point`: the centre of the EDTRS grid
        - `output_file_path`: the path to save the visualisation to
    '''
    num_items = len(layer_thicknesses)
    num_cols = int(np.ceil(np.sqrt(num_items)))
    num_rows = int(np.ceil(num_items / num_cols))

    centre_point = (centre_point[0] * scaling_factor, centre_point[1]) # Scale the centre point

    # Create a grid of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.suptitle('Thickness maps with EDTRS grid overlaid\n(1mm scale in red at bottom-right of images)', fontsize=16)

    # Flatten the axs array in case num_rows or num_cols is 1
    axs = axs.flatten()

    # Iterate over layer_thicknesses and plot each image on a subplot
    for i, (key, value) in enumerate(layer_thicknesses.items()):
        xscale = oct_info[-1]
        overlay_lines = overlay_edtrs_lines(value, centre_point, xscale)
        value_mod = np.copy(value)
        value_mod += overlay_lines * value_mod.max() / 3
        axs[i].imshow(value_mod, cmap='inferno')
        xscale = np.arange(1/xscale)
        axs[i].plot(50 + xscale, np.ones_like(xscale) * 50, c='r', label='1mm scale')
        axs[i].set(title=key)
        axs[i].axis('off')
        # axs[i].legend()

    # Hide empty subplots if any
    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    fig.savefig(str(output_file_path))
    plt.close(fig)

# Masks for statistic computation
# -------------------------------

def create_circular_mask(h: int, w: int, center: tuple = None, radius: int = None) -> np.ndarray:
    '''
    Creates a binary mask of a disk. It is filled.

    Args:
        `h`: height of the mask
        `w`: width of the mask
        `center`: center of the circle
        `radius`: radius of the circle
    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def quadrant_mask(h: int, w: int, center: tuple = None, quadrant: str = 'top') -> np.ndarray:
    '''
    Create a mask of a quadrant of a circle. It's a triangle extending from the center to the edge of the image. It is filled.

    Args:
        `h`: height of the mask
        `w`: width of the mask
        `center`: center of the circle
        `radius`: radius of the circle
    '''
    
    if center is None: # use the middle of the image
        center = ((w/2), (h/2))
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = y - center[1], x - center[0]

    if quadrant == 'bottom':
        return x >= np.abs(y)
    if quadrant == 'top':
        return -x >= np.abs(y)
    if quadrant == 'right':
        return np.abs(x) <= y
    if quadrant == 'left':
        return np.abs(x) <= -y
    
def overlay_edtrs_grid(thickness_map: np.array, centre_point: np.array, xscale: int) -> list[np.array]:
    '''
    Create a ensemble of thickness maps, combined with masks that divide the image into sectors.
    The sectors are every 90 degrees, starting at 45 degree angles. Delimited from 1, 3, 6mm diameter, and centre is not divided.

    Args:
        `thickness_map`: the thickness map to overlay the grid on
        `centre_point`: the centre of the image
        `xscale`: the scaling factor for the image
    
    Returns: a list of thickness maps, each corresponding to a sector
    '''


    # Sectors are every 90 degrees, starting at 45 degree angles. Delimited from 1, 3, 6mm diameter, and centre is not divided
    sector_thickness = [list] * 9

    one_mm_in_px = 1/xscale
    one_mm_in_px = one_mm_in_px / 2 # Radius -> Diameter
    circ_mask = create_circular_mask(*thickness_map.shape, center=centre_point, radius=one_mm_in_px)
    top_quadrant_mask = quadrant_mask(*thickness_map.shape, center=centre_point, quadrant='top')
    right_quadrant_mask = quadrant_mask(*thickness_map.shape, center=centre_point, quadrant='right')
    bottom_quadrant_mask = quadrant_mask(*thickness_map.shape, center=centre_point, quadrant='bottom')
    left_quadrant_mask = quadrant_mask(*thickness_map.shape, center=centre_point, quadrant='left')

    sector_thickness[0] = thickness_map * circ_mask

    three_mm_in_px = one_mm_in_px * 3
    circ_mask = np.bitwise_xor(create_circular_mask(*thickness_map.shape, center=centre_point, radius=three_mm_in_px), circ_mask) # Remove previous circle: makes donut
    sector_thickness[1] = thickness_map * np.bitwise_and(circ_mask, top_quadrant_mask)
    sector_thickness[2] = thickness_map * np.bitwise_and(circ_mask, right_quadrant_mask)
    sector_thickness[3] = thickness_map * np.bitwise_and(circ_mask, bottom_quadrant_mask)
    sector_thickness[4] = thickness_map * np.bitwise_and(circ_mask, left_quadrant_mask)

    six_mm_in_px = one_mm_in_px * 6
    circ_mask = np.bitwise_xor(create_circular_mask(*thickness_map.shape, center=centre_point, radius=six_mm_in_px), create_circular_mask(*thickness_map.shape, center=centre_point, radius=three_mm_in_px))
    sector_thickness[5] = thickness_map * np.bitwise_and(circ_mask, top_quadrant_mask)
    sector_thickness[6] = thickness_map * np.bitwise_and(circ_mask, right_quadrant_mask)
    sector_thickness[7] = thickness_map * np.bitwise_and(circ_mask, bottom_quadrant_mask)
    sector_thickness[8] = thickness_map * np.bitwise_and(circ_mask, left_quadrant_mask)

    return sector_thickness

# Statistics computation, high-level interface for external use
# ------------------------------------------------------------

def compute_quadrant_stats(layer_thicknesses: dict[str, np.array], oct_info: dict[str, list], scaling_factor: int, centre_point: np.array) -> dict[str, list]:
    '''
    Computes the statistics for each quadrant of the thickness maps. (Like Discovery)

    Arguments:
        - `layer_thicknesses`: the thickness maps for each layer
        - `oct_info`: the scale information for the OCT images
        - `scaling_factor`: the scaling factor used to generate the thickness maps
        - `centre_point`: the centre of the EDTRS grid
    
    Returns: a dict of layer name -> list of quadrant statistics. Each quadrant's statistics is a list ordered in the EDTRS grid order.
    '''
    full_quadrant_stats = dict()

    centre_point = (centre_point[0] * scaling_factor, centre_point[1]) # Scale the centre point

    for key, value in layer_thicknesses.items():

        sectored_thickness = overlay_edtrs_grid(value, centre_point=centre_point, xscale=oct_info[-1])

        depth_scale = oct_info[-2] # Formerly y (but this is for thickness in px -> thickness in mm)
        width_scale = oct_info[-1] # x is conserved
        height_scale = oct_info[0] # formerly z, now y. 
        volume_scale = height_scale * width_scale * depth_scale * 1e3 # Convert from mm^3 to nL

        quadrant_stats = [None] * 9
        for i in range(9):
            results = dict()
            mean_thickness_in_region = sectored_thickness[i].sum() / (sectored_thickness[i] != 0).sum() if (sectored_thickness[i] != 0).any() else 0
            if np.isnan(mean_thickness_in_region): continue
            results['mean_thickness_in_px'] = mean_thickness_in_region
            results['mean_thickness_in_um'] = mean_thickness_in_region * depth_scale * 1000 # Convert from mm to um
            for k in results.keys():
                results[k] = int(np.round(results[k]))
            results['mean_volume_in_nL'] = np.round(sectored_thickness[i].sum() * volume_scale, 4)
            quadrant_stats[i] = results

        full_quadrant_stats[key] = quadrant_stats
    return full_quadrant_stats

# Foveal detection
# ----------------

def get_slope_in_slice(image_slice: np.array, interpolation_method: str = 'quadratic') -> Tuple[float, int]:
    '''
    Create a binary mask for the upper layer of a given image, and calculate the deviation from an interpolation of this mask.
    This method serves to determine whether or not a given slice shows a fovea.

    Arguments:
        - `image_slice`: the slice to process
        - `interpolation_method`: the method to use for interpolation. Can be 'linear' or 'quadratic' (default).

    Returns: a tuple of the deviation score and the centre of the fovea.
    '''
    def fit_curve_to_mask(x_and_y_coords: np.array):
        coefs = np.polyfit(*x_and_y_coords, deg=3)
        preds = np.polyval(coefs, x_and_y_coords[0])
        return preds

    out = np.copy(image_slice)

    out = ndi.grey_opening(out, footprint=EIGHT_DISK)
    
    out = ndi.grey_closing(out, footprint=SIXTEEN_DISK)

    thresh = threshold_minimum(out)
    out = out > thresh

    out = find_boundaries(out)
    
    tmp = np.zeros_like(out)
    out = np.argmax(out, axis=0, keepdims=False)
    for i in range(tmp.shape[1]):
        tmp[out[i], i] = 1

    out = tmp
    del tmp

    BOUNDARY_SIZE = 80
    x_and_y = sorted([(b, a) for a, b in zip(*np.nonzero(out))]) # This is an x, y array of non-zero points from the segmentation
    x_and_y = x_and_y[BOUNDARY_SIZE:-BOUNDARY_SIZE] # Crop to boundary size on x axis
    heights = np.array([y[1] for y in x_and_y])

    to_del = []
    big_jumps = np.abs(np.diff(heights))
    where_big_jumps = np.where(big_jumps > 20)[0]
    continuous_segments = np.concatenate([[0], where_big_jumps, [heights.shape[0]]])
    largest_segment = np.diff(continuous_segments).argmax()
    to_del = list(range(continuous_segments[largest_segment] + 1)) + list(range(continuous_segments[largest_segment + 1], heights.shape[0]))

    x_and_y = np.delete(np.array(x_and_y), to_del, axis=0)
    heights = np.array([y[1] for y in x_and_y])
    out = x_and_y.T[::-1] # Dark magic ;)

    START_MEAN_HEIGHT = heights[:10].mean()
    END_MEAN_HEIGHT = heights[-10:].mean()
    if interpolation_method == 'linear':
        expected_slope_per_pixel = (END_MEAN_HEIGHT - START_MEAN_HEIGHT) / heights.shape[0] # We take the mean of the 10 extrema, to improve accuracy
        height_curve_expected = START_MEAN_HEIGHT + expected_slope_per_pixel * np.arange(heights.shape[0])
    elif interpolation_method == 'quadratic':
        height_curve_expected = fit_curve_to_mask(x_and_y.T)

    score = heights - height_curve_expected
    centre_of_fovea = score.argmin() # Point of maximal negative deviation: lowest point of fovea
    score[score > 0] = score[score > 0] / 1.2 # Penalise positive deviations (convex) a lot more than concave ones
    score = (np.abs(score)).sum() # Get the deviation (absolute value), and slice off the edge cases (at boundaries)

    return score, centre_of_fovea

def get_slope_in_pre_segmented_slice(image_slice_pre_segmented: np.array, interpolation_method: str = 'quadratic') -> Tuple[float, int]:
    '''
    Take a binary mask for a given image, and calculate the deviation from an interpolation of this mask.
    This method serves to determine whether or not a given slice shows a fovea.

    Arguments:
        - `image_slice_pre_segmented`: the slice to process, from the output of `get_clean_oct_segmentation`
        - `interpolation_method`: the method to use for interpolation. Can be 'linear' or 'quadratic' (default).

    Returns: a tuple of the deviation score and the centre of the fovea.
    '''
    def fit_curve_to_mask(x_and_y_coords: np.array):
        coefs = np.polyfit(*x_and_y_coords, deg=3)
        preds = np.polyval(coefs, x_and_y_coords[0])
        return preds

    out = np.copy(image_slice_pre_segmented)
    
    tmp = np.zeros_like(out)
    out = np.argmax(out, axis=0, keepdims=False)
    for i in range(tmp.shape[1]):
        tmp[out[i], i] = 1

    out = tmp
    del tmp

    BOUNDARY_SIZE = 80
    x_and_y = sorted([(b, a) for a, b in zip(*np.nonzero(out))]) # This is an x, y array of non-zero points from the segmentation
    x_and_y = x_and_y[BOUNDARY_SIZE:-BOUNDARY_SIZE] # Crop to boundary size on x axis
    heights = np.array([y[1] for y in x_and_y])

    to_del = []
    big_jumps = np.abs(np.diff(heights))
    where_big_jumps = np.where(big_jumps > 20)[0]
    continuous_segments = np.concatenate([[0], where_big_jumps, [heights.shape[0]]])
    largest_segment = np.diff(continuous_segments).argmax()
    to_del = list(range(continuous_segments[largest_segment] + 1)) + list(range(continuous_segments[largest_segment + 1], heights.shape[0]))

    x_and_y = np.delete(np.array(x_and_y), to_del, axis=0)
    heights = np.array([y[1] for y in x_and_y])
    out = x_and_y.T[::-1] # Dark magic ;)

    START_MEAN_HEIGHT = heights[:10].mean()
    END_MEAN_HEIGHT = heights[-10:].mean()
    if interpolation_method == 'linear':
        expected_slope_per_pixel = (END_MEAN_HEIGHT - START_MEAN_HEIGHT) / heights.shape[0] # We take the mean of the 10 extrema, to improve accuracy
        height_curve_expected = START_MEAN_HEIGHT + expected_slope_per_pixel * np.arange(heights.shape[0])
    elif interpolation_method == 'quadratic':
        height_curve_expected = fit_curve_to_mask(x_and_y.T)

    score = height_curve_expected - heights
    centre_of_fovea = BOUNDARY_SIZE + score.argmin() # Point of maximal negative deviation: lowest point of fovea
    score[score < 0] = score[score < 0] / 1.2 # Penalise negative deviations (convex) a lot more than concave ones
    score = (np.abs(score)).sum() # Get the deviation (absolute value), and slice off the edge cases (at boundaries)

    return score, centre_of_fovea

def find_max_slice(cube: np.array, parallel: bool = True) -> Tuple[int, float, int]:
    '''
    Find the slice with the highest foveal score in a given cube.

    Arguments:
        - `cube`: the numpy array of the cube to process, of shape (slices, height, width)
        - `parallel`: whether to run the computation in parallel
    
    Returns: a tuple of the slice index, the foveal score, and the index of the centre of the fovea.
    '''

    if parallel:
        scores_per_slice = tuple(
            tqdm(
                Parallel(n_jobs = cpu_count(), return_as='generator')(delayed(get_slope_in_slice)(sli) for sli in cube),
                total=cube.shape[0],
                desc='Iterating over slices',
                leave=False
            )
        )
    else:
        scores_per_slice = []
        for sli in tqdm(cube, desc='Iterating over slices', leave=False):
            scores_per_slice.append(get_slope_in_slice(sli))
        scores_per_slice = np.array(scores_per_slice)

    where_max_score = np.argmax([x[0] for x in scores_per_slice])
    max_score_slice = cube[where_max_score, :, :]
    return (where_max_score, *get_slope_in_slice(max_score_slice))

def find_max_slice_presegmented(prepared_cube: np.array, parallel: bool = True) -> Tuple[int, float, int]:
    '''
    Find the slice with the highest foveal score in a given cube.

    Arguments:
        - `cube`: the numpy array of the cube to process (presegmented), of shape (slices, height, width)
        - `parallel`: whether to run the computation in parallel
    
    Returns: a tuple of the slice index, the foveal score, and the index of the centre of the fovea.
    '''

    if parallel:
        scores_per_slice = tuple(
            tqdm(
                Parallel(n_jobs = cpu_count(), return_as='generator')(delayed(get_slope_in_pre_segmented_slice)(sli) for sli in prepared_cube),
                total=prepared_cube.shape[0],
                desc='Iterating over slices',
                leave=False
            )
        )
    else:
        scores_per_slice = []
        for sli in tqdm(prepared_cube, desc='Iterating over slices', leave=False):
            scores_per_slice.append(get_slope_in_pre_segmented_slice(sli))
        scores_per_slice = np.array(scores_per_slice)

    where_max_score = np.argmax([x[0] for x in scores_per_slice])
    max_score_slice = prepared_cube[where_max_score, :, :]
    return (where_max_score, *get_slope_in_pre_segmented_slice(max_score_slice))
