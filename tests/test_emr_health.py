import unittest
from datetime import datetime
import pandas as pd
import hashlib
import subprocess
import numpy as np

from src.emr_health.definitions import SERVER, DATABASE, DRIVER, Q_DOCS_AND_POSITIONS, Q_DOCS_AND_POSITIONS_FULL, Q_CONSULT_DATES_AND_DOCTORS, Q_DIAGNOSES_XML, TIME_FORMAT_IN, TIME_FORMAT_QUERY, TIME_FORMAT_QUERY_FULL
from src.cohortbuilder.managers import DataBase as MediSIGHT
from tests.constants import emr_health_expected_output, emr_heath_expected_diagnosis_dates
from src.emr_health.utils import extract_from_xml

test_dates_as_args = ["01-08-2017", "15-08-2017"]

class test_emr_health(unittest.TestCase):
    def __hash_pandas_frame(self, frame):
        return hashlib.sha1(pd.util.hash_pandas_object(frame).values).hexdigest()

    def setUp(self):
        self.start_date = datetime(2017, 8, 1, 0, 0)
        self.end_date = datetime(2017, 8, 15, 0, 0)

    def test_0_establish_connection(self):
        medisight = MediSIGHT(server=SERVER, name=DATABASE, driver=DRIVER)
        self.assertIsInstance(medisight, MediSIGHT)
    
    def test_1_send_dates_and_docs_query(self):
        q_dates_and_docs = Q_CONSULT_DATES_AND_DOCTORS.format(start=datetime.strftime(self.start_date, TIME_FORMAT_QUERY), end=datetime.strftime(self.end_date, TIME_FORMAT_QUERY))

        medisight = MediSIGHT(server=SERVER, name=DATABASE, driver=DRIVER)
        consults_in_window_response = medisight.send(q_dates_and_docs)
        consults_in_window_response_hashed = self.__hash_pandas_frame(consults_in_window_response)

        self.assertEqual(consults_in_window_response_hashed, '3c42109f99a3f2e3dbce20e2531bdf3be6658dae')
    
    def test_2_doctors_and_roles_query(self):
        medisight = MediSIGHT(server=SERVER, name=DATABASE, driver=DRIVER)
        docs_and_positions_response = medisight.send(Q_DOCS_AND_POSITIONS)
        
        self.assertTrue((docs_and_positions_response['grp_description'].unique() == np.array(['infirmière: évaluation pré-op', 'chirurgien','assistant en chirurgie'], dtype=object)).all()) # Less "targets"

        docs_and_positions_full_response = medisight.send(Q_DOCS_AND_POSITIONS_FULL)

        self.assertIn('wolfensbergert', docs_and_positions_full_response['use_username'].tolist()) # Chef de clinique
        self.assertIn('schlingemannr', docs_and_positions_full_response['use_username'].tolist()) # Directeuer de recherche
    
    def test_3_patient_treatments_retrieval(self):
        from bs4 import BeautifulSoup
        patient = '6195EACE-A53E-40A1-8C28-A7AD00C3518A'
        medisight = MediSIGHT(server=SERVER, name=DATABASE, driver=DRIVER)
        diags_response = medisight.send(Q_DIAGNOSES_XML % patient)

        self.assertEqual(self.__hash_pandas_frame(diags_response), 'b8955170416102b9f9928ad64d87ff9c1ea26926') # Response was identical!

        diags_string = diags_response.iloc[0, :1][0] # Extract string from dataframe
        diags_parsed = BeautifulSoup(diags_string, 'xml')

        entries_with_dates = [x for x in diags_parsed.find_all('EyeProblemWidgetItem') if bool(x.find_all('EntireDate'))]

        dates = []
        for entry in entries_with_dates: # Date exists
            date = extract_from_xml(entry.find('EntireDate'), '<EntireDate>', '</EntireDate>')
            if 'xsi:nil' in date: # Malformed date
                continue
            date = datetime.strptime(date, TIME_FORMAT_QUERY_FULL).date() # Get the corresponding date (DAY ONLY!)

            condition = extract_from_xml(entry.find('Description'), '&lt;b&gt;', '&lt;/b&gt;')
            if 'Description' in condition: condition = extract_from_xml(condition, 'tion>', '</Description') # Temp fix for extra tags

            dates.append(date)
        
        self.assertEqual(dates, emr_heath_expected_diagnosis_dates)
    
    def test_4_final_output(self):
        output = subprocess.run(['python', "emr_health.py", *test_dates_as_args],
                                      capture_output=True,
                                      text=True,
                                      check=True).stdout
        self.assertEqual(output, emr_health_expected_output)

if __name__ == "__main__":
    unittest.main()