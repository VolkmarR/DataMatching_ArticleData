import pandas as pd
from unidecode import unidecode
import re
import os
import json
import sys
import random as rnd

class Config:
    """
    Main config class
    """
    def __init__(self, json_config, item_class):
        self.common = Config_Common(json_config["common"])
        self.items = []
        if not (item_class is None):
            for json_item in json_config["items"]:
                self.items.append(item_class(json_item))


class Config_Common:
    """
    Class for common config values
    """
    def __init__(self, json_common):
        self.base_dir = json_common["base_dir"]
        self.filename_1 = self.base_dir + json_common["filename_1"]
        self.filename_2 = self.base_dir + json_common["filename_2"]
        self.filename_perfect_match = self.base_dir + json_common["filename_perfect_match"]
        self.result_base_dir = json_common["result_base_dir"]
        self.fields = []

        for json_common_field in json_common["fields"]:
            self.fields.append(Config_Common_Field(json_common_field))

    def get_result_file_name(self, config_item_index, name):
        """
        returns an output directory for the config item index
        """
        filename = "{0}{1}\\{2}".format(self.result_base_dir, config_item_index + 1, name)
        ensure_directories(filename)
        return filename

class Config_Common_Field:
    """
    Class for a field config
    """
    def __init__(self, json_common_field):
        self.name = json_common_field["name"]
        if "type" in json_common_field:
            self.type = json_common_field["type"]


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


def load_file_as_df(filename, preprocessing_fieldnames):
    """
    Loads a Data File. It is expected, that the file contains the following columns:
    unique_id (the identifier column), title, description
    """
    data = pd.read_csv(filename, encoding="iso-8859-1", engine='c', skipinitialspace=True, index_col=[0])
    # call the preprocessing method on the 2 columns title and description

    if preprocessing_fieldnames:
        for fieldname in preprocessing_fieldnames:
            data[fieldname] = data[fieldname].apply(lambda x: pre_process_string(x))

    return data


def ensure_directories(filename):
    """
    creates the directories used in the filename, if they don't exist
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def load_json_config(filename, default_values):
    """
    loads configuration data from json
    """
    result_config = {}
    if os.path.isfile(filename):
        with open(filename, 'r') as config_file:
            json_Data = json.load(config_file)

    # copy used values to result (or set default value)
    for key, value in default_values.items():
        if key in json_Data:
            result_config[key] = json_Data[key]
        else:
            result_config[key] = default_values[key]

    return result_config

def get_config(config_item_class):
    """
    loads the config file passed as command line argument
    :return: a config instance
    """

    # checking if the config name is valid
    assert (len(sys.argv) == 2), "configuration file name missing"
    config_filename = sys.argv[1]
    assert (os.path.isfile(config_filename)), "configuration file {0} does not exist".format(config_filename)

    # load config
    with open(config_filename, 'r') as config_file:
        json_data = json.load(config_file)

    # move base parameters to class
    result = Config(json_data, config_item_class)

    return result

def init_random_with_seed():
    rnd.seed(34758139)