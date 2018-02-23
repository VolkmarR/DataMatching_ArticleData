import pandas as pd
from unidecode import unidecode
import re

class Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.filename_1 = base_dir + 'file1.csv'
        self.filename_2 = base_dir + 'file2.csv'
        self.filename_perfect_match = base_dir + 'PerfectMapping.csv'

def pre_process_string(value):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """

    if type(value) in (float, int):
        value = ""

    value = unidecode(value)
    value = re.sub('\n', ' ', value)
    value = re.sub('-', '', value)
    value = re.sub('/', ' ', value)
    value = re.sub("'", '', value)
    value = re.sub(",", '', value)
    value = re.sub(":", ' ', value)
    value = re.sub('  +', ' ', value)
    value = value.strip().strip('"').strip("'").lower().strip()
    if not value:
        value = None
    return value


def load_perfect_match_as_index(filename):
    """
    Creates a MultiIndex based on the perfect mapping file
    Expects the record id of the first file in the first column and the record id of the secord file in the second column
    """
    # loading perfectMapping File
    pm = pd.read_csv(filename, encoding="iso-8859-1", engine='c', skipinitialspace=True)
    # create a list of tuples
    idx_tuples = []
    for index, row in pm.iterrows():
        idx_tuples.append((row[0], row[1]))
    # return as multiIndex
    return pd.MultiIndex.from_tuples(idx_tuples, names=["id1", "id2"])
