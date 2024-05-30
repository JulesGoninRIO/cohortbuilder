import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from src.cohortbuilder.discovery.queries import Q_FILE_BY_NAME
from src.cohortbuilder.discovery.discovery import Discovery
# from src.cohortbuilder.discovery.entities import Acquisition, Workbook
from src.cohortbuilder.discovery.queries import Q_DATASET_DELETE, Q_FILE_DELETE, Q_STUDY_UNLINK, Q_PATIENT_UNLINK

# The above lines are commented because of circular import problems. These lines are only used for type hinting and syntax highlighting during coding, so they were removed.
# If you would prefer syntax highlighting and suggestions, just uncomment them while you are editing this file.

def get_stuck_datasets(workbook, list_all_not_just_stuck: bool = False) -> tuple[dict[str, list], str]:
    '''
    Detect failed / pending acquisitions

    discovery_manager_instance: of type Workbook, found at src.cohortbuilder.discovery.entities

    Returns a string to easily print status of datasets, as well as an object grouping acquisitions by status
    '''
    acquisitions = workbook.get_acquisitions(separate=True, verbose=True)
    msg = f'Datasets in {repr(workbook)} :: ' + ' : '.join([
        f'{len(acquisitions["all"])} TOTAL',
        f'{len(acquisitions["pending"])} PENDING',
        f'{len(acquisitions["failed"])} UNSUCCESSFUL',
    ])
    logger.info(f"Fetched info on {len(acquisitions['all'])} acquisitions")
    acquisitions = acquisitions['all'] if list_all_not_just_stuck else acquisitions['pending']
    if not list_all_not_just_stuck:
        logger.info(f"After filtering for problematic cases {len(acquisitions)} are left.")
    return acquisitions, msg

def prompt_for_acquisiton_selection(acqs: list) -> list[int]:
    mapping = {idx: repr(aq) for idx, aq in enumerate(acqs)}
    for k, v in mapping.items():
        print(k, v)

    selection = input('Enter a space-separated list of indexes corresponding to the datasets you want to delete.\nIf you want to delete everything, write "all", for nothing, press enter immediately.\nYour answer:')

    if selection == "all":
        return np.arange(len(mapping)).tolist()
    if selection == '':
        return []

    return [int(x) for x in selection.split(' ')]

def delete_acquisition_and_file(discovery_instance: Discovery, wb, aq) -> None:
    '''
    wb: of type Workbook, found at src.cohortbuilder.discovery.entities
    aq: of type Acquisition, found at src.cohortbuilder.discovery.entities
    '''
    discovery_instance.send_query(Q_DATASET_DELETE, variables={'input': {'uuid': aq.uuid,'workbookUuid': wb.uuid}})
    logger.trace(f"Deleted acquisition with UUID: {aq.uuid} from workbook {wb} on {discovery_instance}.")
    discovery_instance.send_query(Q_FILE_DELETE, variables={'uuid': aq.file.uuid})
    logger.trace(f"Deleted file with UUID: {aq.file.uuid} from workbook {wb} on {discovery_instance}.")

def delete_discovery_file(discovery_instance: Discovery, wb, discovery_file) -> None:
    '''
    wb: of type Workbook, found at src.cohortbuilder.discovery.entities
    discovery_file: of type DiscoveryFile, found at src.cohortbuilder.discovery.entities
    '''
    discovery_instance.send_query(Q_FILE_DELETE, variables={'uuid': discovery_file.uuid})
    logger.trace(f"Deleted file with UUID: {discovery_file.uuid} from workbook {wb} on {discovery_instance}.")

def clear_empty_studies_in_workbook(discovery_instance: Discovery, wb) -> None:
    '''
    wb: of type Workbook, found at src.cohortbuilder.discovery.entities
    '''

    for patient in tqdm(wb.get_children(), desc=f'Clearing empty studies from workbook {repr(wb)}'):
        for study in patient.get_children():
            if not study.get_children(): # Empty study
                discovery_instance.send_query(Q_STUDY_UNLINK, variables = {'input': {'studyUuids': [study.uuid],'workbookUuid': wb.uuid}})
                logger.trace(f"Removed empty study {study} from workbook {wb}.")

def clear_empty_patients_from_workbook(discovery_instance: Discovery, wb) -> None:
    '''
    wb: of type Workbook, found at src.cohortbuilder.discovery.entities
    '''

    for patient in tqdm(wb.get_children(), desc=f'Clearing empty patients from workbook {repr(wb)}'):
        if not patient.get_children(): # Empty patient
            discovery_instance.send_query(Q_PATIENT_UNLINK, variables = {'input': {'patientUuids': [patient.uuid],'workbookUuid': wb.uuid}})
            logger.trace(f"Removed empty patient {patient} from workbook {wb}.")

def get_discovery_file_uuids_by_name(discovery_instance: Discovery, file_name: str) -> list[str]:
    '''
    Helper for getting a list of file UUIDs by searching Discovery for files with a corresponding name.
    You can then use these UUIDs to instantiate a DiscoveryFile.
    '''
    search_results = discovery_instance.send_query(Q_FILE_BY_NAME, variables= {
            "after": 0,
            "first": 2500,
            "input": {
                "filename": file_name
            },
            "status": None
        }
    )['data']['files']['edges']

    result_uuids = [result['node']['uuid'] for result in search_results]
    return result_uuids
