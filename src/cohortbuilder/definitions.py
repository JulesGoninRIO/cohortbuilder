from enum import Enum
from typing import Any

class UploadPipelineFileStatus(Enum):
    """Different status values that uploading a file to Discovery might get."""

    #: There ha been an error uploading the file
    ERROR = -1
    #: The file is found on the Heyex metadata
    DETECTED = 0
    #: The file is added to the queue
    ENQUEUED = 1
    #: The file is picked by a worker
    PICKED = 2
    #: The file is available locally
    FETCHED = 3
    #: The original file is rectified
    RECTIFIED = 4
    #: The file is anonymized
    ANONYMIZED = 5
    #: The file is uploaded
    UPLOADED = 6
    #: Processing the file is finished
    ENDED = 7
    #: Processing the file is finished
    SUCCESSFUL = 8

class cnn_modality_mapping(Enum):
    CFI = 1
    FA = 2
    FAF = 3
    IR = 4
    IRIS = 5
    Multicolor_FI = 6
    ICGA = 7

def index_to_cnn_modality(idx: int) -> Enum:
    if idx == 1: return cnn_modality_mapping.CFI
    if idx == 2: return cnn_modality_mapping.FA
    if idx == 3: return cnn_modality_mapping.FAF
    if idx == 4: return cnn_modality_mapping.IR
    if idx == 5: return cnn_modality_mapping.IRIS
    if idx == 6: return cnn_modality_mapping.Multicolor_FI
    if idx == 7: return cnn_modality_mapping.ICGA

CONFIGURATION_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/schema#",
    "type": "object",
    "properties": {
        "purpose": {
            "type": "string"
        },
        "general": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "metadata": {
                    "type": "boolean"
                },
                "taxonomy": {
                    "type": "boolean"
                },
                "reidentify_modality": {
                    "type": "boolean"
                },
                "post_process_segmentations": {
                    "type": "boolean"
                },
                "detect_fovea_and_recalculate_stats": {
                    "type": "boolean"
                },
                "copy_filtered_workbook": {
                    "type": ["boolean", "null"]
                },
            },
            "required": []
        },
        "types": {
            "type": "object",
            "properties": {
                "oct": {
                    "type": "boolean"
                },
                "fundus": {
                    "type": "boolean"
                },
                "thumbnail": {
                    "type": "boolean"
                },
                "segmentation": {
                    "type": "boolean"
                },
                "biomarkers": {
                    "type": "boolean"
                },
                "thicknesses": {
                    "type": "boolean"
                },
                "volumes": {
                    "type": "boolean"
                },
                "rawimage": {
                    "type": "boolean"
                },
                "ecrf": {
                    "type": "boolean"
                },
                "pdf": {
                    "type": "boolean"
                },
                "projection_images": {
                    "type": "boolean"
                },
                "thickness_images": {
                    "type": "boolean"
                },
                "h5": {
                    "type": "boolean"
                },
                "dicom": {
                    "type": "boolean"
                },
                "e2e": {
                    "type": "boolean"
                },
                "fda": {
                    "type": "boolean"
                }
            },
            "required": [
                "biomarkers",
                "dicom",
                "e2e",
                "ecrf",
                "fda",
                "fundus",
                "h5",
                "oct",
                "pdf",
                "projection_images",
                "rawimage",
                "segmentation",
                "thickness_images",
                "thicknesses",
                "thumbnail",
                "volumes"
            ]
        },
        "filters": {
            "type": "object",
            "properties": {
                "patients": {
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": ["array", "null"],
                            "items": {
                                "type": ["integer"]
                            }
                        },
                        "birthdate_inf": {
                            "type": ["integer", "null"]
                        },
                        "birthdate_sup": {
                            "type": ["integer", "null"]
                        },
                        "sex": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": [
                        "birthdate_inf",
                        "birthdate_sup",
                        "ids",
                        "sex"
                    ]
                },
                "studies": {
                    "type": "object",
                    "properties": {
                        "uuids": {
                            "type": ["string", "null"]
                        },
                        "variants": {
                            "type": "null"
                        },
                        "date_inf": {
                            "type": ["string", "null"]
                        },
                        "date_sup": {
                            "type": ["string", "null"]
                        },
                        "patient_age_inf": {
                            "type": ["integer", "null"]
                        },
                        "patient_age_sup": {
                            "type": ["integer", "null"]
                        }
                    },
                    "required": [
                        "date_inf",
                        "date_sup",
                        "patient_age_inf",
                        "patient_age_sup",
                        "uuids",
                        "variants"
                    ]
                },
                "datasets": {
                    "type": "object",
                    "properties": {
                        "uuids": {
                            "type": ["string", "null"]
                        },
                        "variants": {
                            "type": ["string", "null"]
                        },
                        "laterality": {
                            "type": ["string", "null"]
                        },
                        "device": {
                            "type": ["string", "null"]
                        },
                        "manufacturer": {
                            "type": ["string", "null"]
                        },
                        "invisible": {
                            "type": "boolean"
                        }
                    },
                    "required": [
                        "device",
                        "invisible",
                        "laterality",
                        "manufacturer",
                        "uuids",
                        "variants"
                    ]
                }
            },
            "required": []
        },
        "status": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "boolean"
                },
                "pending": {
                    "type": "boolean"
                }
            },
            "required": [
                "pending"
            ]
        },
        "processes": {
            "type": "object",
            "properties": {
                "LFSEGMENTATION": {
                    "type": "boolean"
                },
                "GAPROGRESSION": {
                    "type": "boolean"
                },
                "BIOMARKERS": {
                    "type": "boolean"
                },
                "GASEGMENTATION": {
                    "type": "boolean"
                },
                "POSTPROCESSOR": {
                    "type": "boolean"
                }
            },
            "required": [
                "LFSEGMENTATION",
                "GAPROGRESSION",
                "BIOMARKERS",
                "GASEGMENTATION",
                "POSTPROCESSOR"
            ]
        },
        "modalities": {
            "type": "object",
            "properties": {
                "pdf": {
                    "type": "boolean"
                },
                "oct": {
                    "type": "boolean"
                },
                "cfi": {
                    "type": "boolean"
                },
                "angiography": {
                    "type": "boolean"
                }
            },
            "required": [
                "pdf",
                "oct",
                "cfi",
                "angiography"
            ]
        }
    },
    "required": [
        "general",
        "purpose"
    ]
}
