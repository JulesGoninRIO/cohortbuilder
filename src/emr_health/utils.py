import pandas as pd
from io import StringIO

def extract_from_xml(string, start_tag, end_tag):
    string = str(string)
    try:
        DELIMITING_START = string.find(start_tag) + len(start_tag)
        DELIMITING_END = string.rfind(end_tag)
    except TypeError: # Did not find anything
        return string

    return string[DELIMITING_START:DELIMITING_END]

def convert_xml_str_to_pd(string):
    return pd.read_xml(StringIO(string),
                       dtype_backend='numpy_nullable',
                       parse_dates=['EntireDate']
                    ) # Requires pandas >= 1.3