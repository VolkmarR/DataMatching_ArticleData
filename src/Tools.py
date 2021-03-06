import pandas as pd
from unidecode import unidecode
import re
import os
import json
import sys
import random as rnd
import math
from pathlib import Path
from collections import defaultdict
from recordlinkage.base import BaseIndexator

class Config:
    """
    Main config class
    """
    def __init__(self, json_config, item_class, config_name):
        self.common = Config_Common(json_config["common"], config_name)
        self.items = []
        if not (item_class is None):
            for json_item in json_config["items"]:
                self.items.append(item_class(json_item))


class Config_Common:
    """
    Class for common config values
    """
    def __init__(self, json_common, config_name):
        self.config_name = config_name
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
        filename = "{0}{1}\\{2}\\{3}".format(self.result_base_dir, self.config_name, config_item_index + 1, name)
        ensure_directories(filename)
        return filename


    def fields_to_string(self):
        """
        converts the fields to a string
        """
        return ",".join(map(lambda x: x.to_string(), self.fields))


class Config_Common_Field:
    """
    Class for a field config
    """
    def __init__(self, json_common_field):
        self.name = json_common_field["name"]
        if "type" in json_common_field:
            self.type = json_common_field["type"]

    def to_string(self):
        if self.type:
            return "{0} ({1})".format(self.name, self.type)
        else:
            return self.name


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
    Expects the record id of the first file in the first column
    and the record id of the second file in the second column
    """
    # loading perfectMapping File
    pm = pd.read_csv(filename, encoding="iso-8859-1", engine='c', skipinitialspace=True)
    # create a list of tuples
    idx_tuples = []
    for index, row in pm.iterrows():
        idx_tuples.append((str(row[0]), str(row[1])))
    # return as multiIndex
    return pd.MultiIndex.from_tuples(idx_tuples, names=["id1", "id2"])


def load_file_as_df(filename, preprocessing_fieldnames):
    """
    Loads a Data File. It is expected, that the file contains the following columns:
    unique_id (the identifier column), title, description
    """
    data = pd.read_csv(filename, encoding="iso-8859-1", engine='c', skipinitialspace=True, index_col=[0])
    # call the preprocessing method on the 2 columns title and description

    data.index = data.index.map(str)
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
    result = Config(json_data, config_item_class, Path(config_filename).stem)

    return result

def init_random_with_seed():
    """
    initializes the random generator with a fixed seed
    """
    rnd.seed(34758139)


def init_bin_top(max_value, bin_count):
    """
    creates a list for binning
    """
    max_value = math.ceil(max_value)
    bin_size = round(max_value / bin_count, 3)
    bins = []
    i = bin_size
    while i <= max_value:
        bins.append(i)
        i = round(i + bin_size, 3)

    return bins


def bin_values(bin_top, series):
    """
    puts the count of values into the bins
    """
    # init result
    bin_count = len(bin_top)
    result = [0] * bin_count

    # start binning
    i = 0
    for value in series.sort_values():
        # check if the index of the bin is valid
        while i < bin_count and bin_top[i] < round(value, 6):
            i += 1

        # increment the bin_count
        result[i] += 1

    # normalize values
    series_count = len(series)
    for i, val in enumerate(result):
        result[i] = round(val / series_count, 6)

    return result


def series_to_bins(series_match, series_distinct, bin_count):
    """
    splits the values of the series to bin_count bins.
    each bin contains the count of values contained in the bin
    """

    # create bin_top
    max_value = round(pd.Series([series_match.max(), series_distinct.max()]).max(), 6)
    bin_top = init_bin_top(max_value, bin_count)

    # create binned values
    bin_count_match = bin_values(bin_top, series_match)
    bin_count_distinct = bin_values(bin_top, series_distinct)

    # return the dataframe
    return pd.DataFrame({
        "bin_top": pd.Series(bin_top),
        "Match": pd.Series(bin_count_match),
        "Non-Match": pd.Series(bin_count_distinct)},
        columns=['bin_top', 'Match', 'Non-Match'])


def save_csv(df, filename, index=True):
    df.to_csv(filename, index=index, decimal=',', sep=';')


class CanopyClusterIndex(BaseIndexator):
    """Canopy clustering for indexing"""

    @staticmethod
    def buildbigram(str):
        result = set()
        str = str.lower()
        for i in range(len(str) - 1):
            result.add(str[i:i + 2])
        return result

    @staticmethod
    def sim_jacc(set1, set2):
        union = len(set1.union(set2))
        if union != 0:
            return len(set1.intersection(set2)) / union
        else:
            return 0

    def __init__(self,
                 left_on=None,
                 right_on=None,
                 threshold_add=0.3,
                 threshold_remove=0.8,
                 **kwargs):
        super(CanopyClusterIndex, self).__init__(**kwargs)

        if right_on is None:
            right_on = left_on

        # variables to block on
        self.left_on = left_on
        self.right_on = right_on
        self.threshold_add = threshold_add
        self.threshold_remove = threshold_remove

    def _link_index(self, df_a, df_b):
        """Make pairs ."""

        result = set()

        # create the inverted index and bigram dictionary for df_b
        data_dict = {}
        inverted_index = defaultdict(list)
        for index, row in df_b.iterrows():
            bigrams = CanopyClusterIndex.buildbigram(row[self.right_on])
            data_dict[index] = bigrams
            for bigram_b in bigrams:
                inverted_index[bigram_b].append(index)

        # create the canopies
        for idx_a, row in df_a.iterrows():
            bigrams_a = CanopyClusterIndex.buildbigram(row[self.left_on])

            # get all elements similar entries
            similar = set()
            for bigram_b in bigrams_a:
                for idx_b in inverted_index[bigram_b]:
                    similar.add(idx_b)

            # calc the distances
            for idx_b in similar:
                if idx_b in data_dict:
                    bigrams_b = data_dict[idx_b]
                    sim = CanopyClusterIndex.sim_jacc(bigrams_a, bigrams_b)
                    if sim > self.threshold_add:
                        result.add((idx_a, idx_b))
                        if sim > self.threshold_remove:
                            del data_dict[idx_b]

        return pd.MultiIndex.from_tuples(result, names=[df_a.index.name, df_b.index.name])
