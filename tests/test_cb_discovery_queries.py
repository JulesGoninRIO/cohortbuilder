import unittest

from src.cohortbuilder.discovery.queries import *
from src.cohortbuilder.discovery.discovery import Discovery
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.utils.helpers import read_json

class test_all_individual_queries(unittest.TestCase):
    def __check_string_is_uuid(self, string: str):
        return [len(x) for x in string.split('-')] == [8, 4, 4, 4, 12]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        args = []
        settings = read_json('settings.json')

        keys: dict[str, dict] = read_json(settings['general']['keys'])
        settings['slims'].update(keys['slims'])
        for api_name in keys['api']:
            settings['api'][api_name].update(keys['api'][api_name])

        Parser.store(args=args, settings=settings)
        self.disco = Discovery(instance='fhv_jugo')

        get_projects = self.disco.send_query(Q_PROJECTS)['data']['profile']['projects']
        self.cb_project_uuid = get_projects[[x['title'] for x in get_projects].index('cohortbuilder')]['uuid']

        self.cb_test_workbook_uuid = '18cf5f3a-f4c2-4098-aba4-ecca00a8000e'

    def tearDown(self) -> None:
        del self.disco

    def test_0_q_projects(self):
        response = self.disco.send_query(Q_PROJECTS)['data']['profile']['projects']
        self.assertIsInstance(response, list)
        project_names = [x['title'] for x in response]
        # Test that a few expected projects are properly listed
        self.assertIn('cohortbuilder', project_names)
        self.assertIn('CSCR', project_names)
        self.assertIn('SOIN_LUKS', project_names)

        # Test that we get actualy discovery UUIDs
        project_uuids = [x['uuid'] for x in response]
        self.assertTrue(all([self.__check_string_is_uuid(x) for x in project_uuids]))
    
    def test_1_q_project_workbooks(self):
        # An other key is returned apart from data (errors). Here it complains about 0 not being an int, but this query gives all the data in one go, so it's nice :)
        response = self.disco.send_query(Q_PROJECT_WORKBOOKS % (self.cb_project_uuid, 0, 0))['data']['workbooks']

        self.assertGreater(response['totalCount'], 0)
        self.assertIsInstance(response['edges'], list)
        self.assertTrue(response['edges']) # We get interesting data back
        self.assertEqual(list(response['edges'][0]['node'].keys()), ['uuid', 'name', 'currentPermission'])
        self.assertTrue(self.__check_string_is_uuid(response['edges'][0]['node']['uuid']))
    
    def test_2_q_workbook_patients(self):
        num = self.disco.send_query(Q_WORKBOOK_PATIENTS % (self.cb_test_workbook_uuid, 1, 0, self.cb_test_workbook_uuid))['data']['patientSearchFeed']['totalCount']
        response = self.disco.send_query(Q_WORKBOOK_PATIENTS % (self.cb_test_workbook_uuid, num // 1000, 0, self.cb_test_workbook_uuid))['data']['patientSearchFeed'] # We get 1000x less than the max, to not overwork the server for nothing. This shows how to do stuff ;)
        self.assertGreater(response['totalPages'], 1) # Make sure that we have gotten everything :)
        self.assertGreater(response['totalCount'], 0)
        self.assertIsInstance(response['edges'], list)
        self.assertTrue(response['edges']) # We get interesting data back
        self.assertEqual(list(response['edges'][0]['node'].keys()), ['uuid', 'name', 'surname', 'patientId', 'birthdate', 'sex', 'studies'])
        self.assertTrue(self.__check_string_is_uuid(response['edges'][0]['node']['uuid']))
    
    # Queries skipped (modify the records): Q_PATIENT_MOVE, Q_PATIENT_EDIT, Q_PATIENT_MERGE, Q_PATIENT_UNLINK, Q_PATIENT_DELETE

    def test_3_q_patient_studies(self):
        # This UUID was grabbed off a random, real request made by Discovery
        random_patient_uuid = '47146b29-03c8-4917-bed9-bc52f7f6f808'

        query = Q_PATIENT_STUDIES % (random_patient_uuid, self.cb_test_workbook_uuid, 0, 0, self.cb_test_workbook_uuid)
        response = self.disco.send_query(query)['data']['patient']['studies']['edges']

        self.assertIsInstance(response, list)
        self.assertTrue(response)
        self.assertEqual(list(response[0]['node'].keys()), ['uuid', 'studyId', 'studyDatetime', 'modalities'])
        self.assertTrue(self.__check_string_is_uuid(response[0]['node']['uuid']))
    
    def test_4_q_patient_datasets_short(self):
        # This UUID was grabbed off a random, real request made by Discovery
        random_patient_uuid = '47146b29-03c8-4917-bed9-bc52f7f6f808'

        query = Q_PATIENT_DATASETS_SHORT % (random_patient_uuid, self.cb_test_workbook_uuid, 0, 0, self.cb_test_workbook_uuid)
        response = self.disco.send_query(query)['data']['patient']['datasets']

        self.assertGreater(response['totalCount'], 0)
        self.assertTrue(response['totalPages'] is None or response['totalPages'] == 1)
        self.assertIsInstance(response['edges'], list)
        self.assertEqual(list(response['edges'][0]['node'].keys()), ['uuid', 'status', 'purpose', 'parentFile'])
        self.assertTrue(self.__check_string_is_uuid(response['edges'][0]['node']['uuid']))

    # Queries skipped (modify the records): Q_STUDY_MOVE, Q_STUDY_DELETE

    def test_5_q_study_datasets(self):
        random_study_uuid = "ff600124-d58b-4186-bab7-279d991e256a"
        response = self.disco.send_query(Q_STUDY_DATASETS % (random_study_uuid, self.cb_test_workbook_uuid, 'ACQUISITION', 0, 0, self.cb_test_workbook_uuid, self.cb_test_workbook_uuid))['data']['study']['datasets']

        self.assertGreater(response['totalCount'], 0)
        self.assertTrue(response['totalPages'] is None or response['totalPages'] == 1)
        self.assertIsInstance(response['edges'], list)
        self.assertEqual(list(response['edges'][0]['node'].keys()), ['uuid', 'purpose', 'status', 'owner', 'createdAt', 'updatedAt', 'laterality', 'tags', 'device', 'manufacturer', 'acquisitionDatetime', 'seriesDatetime', 'signedUrl', 'thumbnail', 'parentFile', 'children', 'layers'])
        self.assertTrue(self.__check_string_is_uuid(response['edges'][0]['node']['uuid']))

    def test_6_q_dataset_refresh(self):
        random_dataset_uuid = "1eff123f-0408-4b5c-91ae-24bf252b85f1"
        response = self.disco.send_query(Q_DATASET_REFRESH % (random_dataset_uuid, self.cb_test_workbook_uuid, self.cb_test_workbook_uuid))['data']['dataset']
        self.assertEqual(list(response.keys()), ['signedUrl', 'thumbnail', 'parentFile', 'children', 'layers'])
        self.assertIn('https://sos-ch-gva-2.exo.io/discovery-retinai-soin/dataset/', response['signedUrl'])

    # Queries skipped (modify the records): Q_DATASET_MOVE, Q_DATASET_UNLINK, Q_DATASET_DELETE

    def test_7_q_files(self):
        response = self.disco.send_query(Q_FILES % (30, 1))['data']['files']

        self.assertGreater(response['totalCount'], 0)   
        self.assertTrue(response['totalPages'] is None or response['totalPages'] > 1)
        self.assertIsInstance(response['edges'], list)
        self.assertEqual(list(response['edges'][0]['node'].keys()), ['uuid', 'filename', 'status', 'createdAt', 'datasets'])
        self.assertTrue(self.__check_string_is_uuid(response['edges'][0]['node']['uuid']))
    
    # Queries skipped (modify the records): Q_FILE_UPLOAD

    def test_8_q_file_info(self):
        random_file_uuid = 'e76c5c35-d511-4b25-987c-9adc7bb408da'
        response = self.disco.send_query(Q_FILE_INFO % random_file_uuid)['data']['file']

        self.assertEqual(list(response.keys()), ['uuid', 'filename', 'status', 'createdAt', 'datasets'])
        self.assertTrue(self.__check_string_is_uuid(response['uuid']))
    
    def test_9_q_file_acquisitions(self):
        random_file_uuid = 'e76c5c35-d511-4b25-987c-9adc7bb408da'
        response = self.disco.send_query(Q_FILE_ACQUISITIONS % (random_file_uuid, 0, 0))['data']['file']['datasets']

        self.assertGreater(response['totalCount'], 0)   
        self.assertTrue(response['totalPages'] is None or response['totalPages'] == 1)
        self.assertIsInstance(response['edges'], list)
        self.assertEqual(list(response['edges'][0]['node'].keys()), ['uuid', 'status'])
        self.assertTrue(self.__check_string_is_uuid(response['edges'][0]['node']['uuid']))
    
    # Queries skipped (modify the records): Q_FILE_DELETE

    def test_10_q_acquisition_jobs(self):
        random_dataset_uuid = "1eff123f-0408-4b5c-91ae-24bf252b85f1"
        random_file_uuid = 'e76c5c35-d511-4b25-987c-9adc7bb408da'
        response = self.disco.send_query(Q_ACQUISITION_JOBS % (random_dataset_uuid, random_file_uuid))['data']['jobs']

        self.assertIsInstance(response, list)
        self.assertEqual(list(response[0].keys()), ['uuid', 'name', 'status', 'startedAt', 'createdAt', 'tasks']) # File status
        self.assertTrue(self.__check_string_is_uuid(response[0]['uuid']))
        self.assertIsInstance(response[0]['tasks'], list)
        self.assertEqual(list(response[0]['tasks'][0].keys()), ['uuid', 'path', 'status', 'processor', 'message', 'io']) # Job statuses
        self.assertTrue(self.__check_string_is_uuid(response[0]['tasks'][0]['uuid']))
    
    # Queries skipped (modify the records): Q_ACQUISITION_STARTJOB

    def test_11_q_processor(self):
        response = self.disco.send_query(Q_PROCESSOR % (''))['data']['processors']

        self.assertIsInstance(response, list)
        self.assertEqual(list(response[0].keys()), ['uuid', 'name', 'version'])
        self.assertTrue(self.__check_string_is_uuid(response[0]['uuid']))

    
if __name__ == "__main__":
    unittest.main()