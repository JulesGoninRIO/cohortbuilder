"""
This module includes the names and the definitions that are hard-coded in the
downloaded files from Discovery.
"""

from __future__ import annotations

from enum import Enum


class RetinalLayer(Enum):
    """
    The title, the SVG tag (in segmentations), and the type of retinal layers.
    """

    # NOTE: The order is important.
    BG = {
        'title': 'Background',
        'tag': 'layer layer-0',
        'type': None,
        }
    RNFL = {
        'title': 'Retinal Nerve Fibre Layer',
        'tag': 'layer layer-1',
        'type': 'layer',
        }
    GCL_IPL = {
        'title': 'Ganglion Cell Layer and Inner Plexiform Layer',
        'tag': 'layer layer-2',
        'type': 'layer',
        }
    INL_OPL = {
        'title': 'Inner Nuclear Layer and Outer Plexiform Layer',
        'tag': 'layer layer-3',
        'type': 'layer',
        }
    ONL = {
        'title': 'Outer Nuclear Layer',
        'tag': 'layer layer-4',
        'type': 'layer',
        }
    PR_RPE = {
        'title': 'Photoreceptors and Retinal Pigment Epithelium',
        'tag': 'layer layer-5',
        'type': 'layer',
        }
    CC_CS = {
        'title': 'Choriocapillaris and Choroidal Stroma',
        'tag': 'layer layer-6',
        'type': 'layer',
        }
    IRF = {
        'title': 'Intraretinal Fluid',
        'tag': 'layer layer-7',
        'type': 'fluid',
        }
    SRF = {
        'title': 'Subretinal Fluid',
        'tag': 'layer layer-8 fluid',
        'type': 'fluid',
        }
    PED = {
        'title': 'Pigment Epithelial Detachment',
        'tag': 'layer layer-9 fluid',
        'type': 'fluid',
        }

    # CHECK: What is this layer?
    # NOTE: Extra number in the thickness or volume json files that
    # does not match with anything on Discovery UI
    Unknown = {
        'title': None,
        'tag': None,
        'type': None,
        }

    @classmethod
    def get_names(cls, plus: bool = False) -> list[str]:
        """
        Returns a list of the names.
        If ``plus`` is True, the underscore in the names will be replaced by ``"+"``.
        """

        return [layer.name.replace('_', '+') if plus else layer.name for layer in cls]

    @classmethod
    def get_names_layers(cls, plus: bool = False) -> list[str]:
        """
        Returns a list of the names (only layers).
        If ``plus`` is True, the underscore in the names will be replaced by ``"+"``.
        """

        return [layer.name.replace('_', '+') if plus else layer.name for layer in cls if layer.value['type'] == 'layer']

    @classmethod
    def get_names_fluids(cls, plus: bool = False) -> list[str]:
        """
        Returns a list of the names (only fluids).
        If ``plus`` is True, the underscore in the names will be replaced by ``"+"``.
        """

        return [layer.name.replace('_', '+') if plus else layer.name for layer in cls if layer.value['type'] == 'fluid']

    @classmethod
    def get_titles(cls) -> list[str]:
        """Returns a list of the titles."""

        return [layer.value['title'] for layer in cls]

    @classmethod
    def get_tags(cls) -> list[str]:
        """Returns a list of the tags."""

        return [layer.value['tag'] for layer in cls]

class Biomarker(Enum):
    """The title and the type of retinal biomarkers."""

    # NOTE: The order is important.
    SRF = {
        'title': 'Subretinal Fluid',
        'tag': None,
        'type': 'biomarker'
        }
    IRF = {
        'title': 'Intraretinal Fluid',
        'tag': None,
        'type': 'biomarker'
        }
    HF = {
        'title': 'Hyper Reflective Foci',
        'tag': None,
        'type': 'biomarker'
        }
    DRUSEN = {
        'title': 'Drusuen',
        'tag': None,
        'type': 'biomarker'
        }
    RPD = {
        'title': 'Reticular Pseudo Drusen',
        'tag': None,
        'type': 'biomarker'
        }
    ERM = {
        'title': 'Epiretinal Membrane',
        'tag': None,
        'type': 'biomarker'
        }
    GA = {
        'title': 'Geographic Atrophy',
        'tag': None,
        'type': 'biomarker'
        }
    ORA = {
        'title': 'Outer Retinal Atrophy',
        'tag': None,
        'type': 'biomarker'
        }
    FPED = {
        'title': 'Fibrovascular Pigment Epithelial Detachement',
        'tag': None,
        'type': 'biomarker'
        }

    @classmethod
    def get_names(cls) -> list[str]:
        """Returns a list of the names."""

        return [layer.name for layer in cls]

    @classmethod
    def get_titles(cls) -> list[str]:
        """Returns a list of the titles."""

        return [layer.value['title'] for layer in cls]

    @classmethod
    def get_tags(cls) -> list[str]:
        """Returns a list of the tags."""

        return [layer.value['tag'] for layer in cls]

class Region(Enum):
    """Eye regions from an enface view."""

    # NOTE: The order is important.
    BG = 'Background'
    I6 = 'I6'
    N6 = 'N6'
    S6 = 'S6'
    T6 = 'T6'
    I3 = 'I3'
    N3 = 'N3'
    S3 = 'S3'
    T3 = 'T3'
    C1 = 'C1'

    @classmethod
    def get_names(cls) -> list[str]:
        """Returns a list of the names."""

        return [region.name for region in cls]

    @classmethod
    def get_titles(cls) -> list[str]:
        """Returns a list of the titles."""

        return [region.value for region in cls]

    @classmethod
    def central(cls) -> list[Region]:
        """Returns a list of the titles."""

        return [Region.C1]

    @classmethod
    def pericentral(cls) -> list[Region]:
        """Returns a list of the titles."""

        return [Region.S3, Region.T3, Region.N3, Region.I3]

    @classmethod
    def peripheral(cls) -> list[Region]:
        """Returns a list of the titles."""

        return [Region.S6, Region.T6, Region.N6, Region.I6]
