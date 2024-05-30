import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
from zlib import crc32
from tqdm import tqdm
from shutil import get_terminal_size
from argparse import ArgumentParser

from src.emr_health.definitions import SERVER, DATABASE, DRIVER, Q_DOCS_AND_POSITIONS, Q_DOCS_AND_POSITIONS_FULL, Q_CONSULT_DATES_AND_DOCTORS, Q_DIAGNOSES_XML, TIME_FORMAT_IN, TIME_FORMAT_QUERY, TIME_FORMAT_QUERY_FULL
from src.cohortbuilder.managers import DataBase as MediSIGHT
from src.emr_health.utils import extract_from_xml

medisight = MediSIGHT(server=SERVER, name=DATABASE, driver=DRIVER)

parser = ArgumentParser(prog='EMR Health Checker')
parser.add_argument('--start-date', type=str, dest='start_date', required=True)
parser.add_argument('--end-date', type=str, dest='end_date', required=True)
parser_careful_group = parser.add_mutually_exclusive_group(required=True)
parser_careful_group.add_argument('--be-careful', action='store_true', dest='be_careful')
parser_careful_group.add_argument('--dont-be-careful', action='store_false', dest='be_careful')
parser_careful_group.set_defaults(be_careful=True)
parser.add_argument('--no-anonymise-pids', action='store_true', dest='no_anonymise_pids')
args = parser.parse_args()

start_date = datetime.strptime(args.start_date, TIME_FORMAT_IN)
end_date = datetime.strptime(args.end_date, TIME_FORMAT_IN)

be_careful = args.be_careful

q_dates_and_docs = Q_CONSULT_DATES_AND_DOCTORS.format(start=datetime.strftime(start_date, TIME_FORMAT_QUERY), end=datetime.strftime(end_date, TIME_FORMAT_QUERY))

consults_in_window = medisight.send(q_dates_and_docs) # Already selected unique records with SQL

if be_careful:
    docs_and_roles = medisight.send(Q_DOCS_AND_POSITIONS) # Already selected unique records with SQL
else:
    docs_and_roles = medisight.send(Q_DOCS_AND_POSITIONS_FULL) # All docs in hospital

diagnoses_per_doc = {doc: 0 for doc in docs_and_roles['use_fullname'].tolist() if 'Dr' in doc or 'Prof' in doc or 'Pre' in doc} # Make sure they are a doc or more, and initialise to 0 diags
outputted_combs = set()

dates_per_patient = dict()
conditions_per_patient = dict()

for j, patient in enumerate(tqdm(consults_in_window['pat_id'], desc='Iterating over medical checkups in time window')):
    this_encounter = consults_in_window.iloc[j]
    treating_doctor = this_encounter['use_fullname']

    if be_careful and treating_doctor not in docs_and_roles['use_fullname'].tolist(): # We skip this guy
        continue

    date_time_treatment = datetime.strptime(str(this_encounter['ect_date']), TIME_FORMAT_QUERY_FULL.replace('T', ' ')).date() # Get the date of treatment (day only!)

    if patient not in dates_per_patient.keys():
        diags_response = medisight.send(Q_DIAGNOSES_XML % patient)
        if diags_response.empty:
            continue

        diags_string = diags_response.iloc[0, :1][0] # Extract string from dataframe
        diags_parsed = BeautifulSoup(diags_string, 'xml')

        entries_with_dates = [x for x in diags_parsed.find_all('EyeProblemWidgetItem') if bool(x.find_all('EntireDate'))]

        dates = []
        conditions = []
        for entry in entries_with_dates: # Date exists
            date = extract_from_xml(entry.find('EntireDate'), '<EntireDate>', '</EntireDate>')
            if 'xsi:nil' in date: # Malformed date
                continue
            date = datetime.strptime(date, TIME_FORMAT_QUERY_FULL).date() # Get the corresponding date (DAY ONLY!)

            condition = extract_from_xml(entry.find('Description'), '&lt;b&gt;', '&lt;/b&gt;')
            if 'Description' in condition: condition = extract_from_xml(condition, 'tion>', '</Description') # Temp fix for extra tags

            dates.append(date)
            conditions.append(condition)

        conditions = np.array(conditions)
        dates = np.array(dates)

        dates_per_patient[patient] = dates
        conditions_per_patient[patient] = conditions
    
    dates, conditions = dates_per_patient[patient], conditions_per_patient[patient]

    this_day_this_patient = np.where(dates == date_time_treatment)[0]
    current_conditions = []
    for idx in this_day_this_patient:
        current_conditions.append(conditions[idx])

    for c in current_conditions:
        if args.no_anonymise_pids:
            prepared_string = ' '.join([c, 'diagnosed by', treating_doctor, 'on', str(date_time_treatment), '-- PID:', patient])
        else:
            prepared_string = ' '.join([c, 'diagnosed by', treating_doctor, 'on', str(date_time_treatment), '-- Hashed (anonymised) PID:', hex(crc32(patient.encode('utf-8')) & 0xffffffff)])
        if prepared_string not in outputted_combs: # last-defense sanity-check
            outputted_combs.add(prepared_string)
            tqdm.write(prepared_string)
            try:
                diagnoses_per_doc[treating_doctor] += 1
            except KeyError:
                diagnoses_per_doc[treating_doctor] = 1

print('-' * get_terminal_size().columns)
print('Bilan:')

frame = pd.DataFrame({'docteur': list(diagnoses_per_doc.keys()), 'nombre de patients': list(diagnoses_per_doc.values())})
frame = frame.sort_values(by='nombre de patients', ascending=False)
frame.index = np.arange(frame.shape[0]) + 1

print(frame.to_string())