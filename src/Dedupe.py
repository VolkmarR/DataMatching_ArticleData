#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use RecordLink with two comma separated
values (CSV) files. We have listings of products from two different
online stores. The task is to link products between the datasets.

The output will be a CSV with our linked results.

"""
import csv
import logging
import dedupe
import Tools as tools
import Evaluation as ev
import random as rnd


class Config_Item:
    """
    Config item class for Dedupe
    """
    def __init__(self, json_item):
        self.golden_pairs_count = json_item["golden_pairs_count"]

    def to_dict(self):
        return dict({"golden_pairs_count": self.golden_pairs_count})



def load_data(filename, preprocessing_fieldnames):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID.
    """

    data_d = {}
    if preprocessing_fieldnames:
        preprocessing_fieldnames_set = set(preprocessing_fieldnames)
    else:
        preprocessing_fieldnames_set = set()
    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            clean_row = dict([(k, v) for (k, v) in row.items()])
            for fieldname in preprocessing_fieldnames_set:
                clean_row[fieldname] = tools.pre_process_string(clean_row[fieldname])
            data_d[filename + str(i)] = dict(clean_row)

    return data_d


def build_corpus(fieldname):
    # returns a list of all not empty values of the field (used by the text comparer to build the list of rare words)
    corpus_set = []
    for dataset in (data_1, data_2):
        for row in dataset.values():
            if row[fieldname]:
                corpus_set.append(row[fieldname])
    return corpus_set


def count_matches(deduper):
    # Returns the number of match training pairs
    return len(deduper.training_pairs['match'])


def count_distinct(deduper):
    # Returns the number of distinct training pairs
    return len(deduper.training_pairs['distinct'])


def train_with_perfect_match(deduper, max_count, idx_perfect_match):
    pairs = []
    while True:
        # get next pair
        if not pairs:
            try:
                pairs = deduper.uncertainPairs()
            except IndexError:
                # exit the loop, if no pairs are available
                break

        record_pair = pairs.pop()

        # build key
        ids = []
        for pair in record_pair:
            ids.append(str(pair["id"]))
        key = (ids[0], ids[1])

        # check if key is match
        is_match = key in idx_perfect_match

        examples = {'distinct': [], 'match': []}
        if is_match and count_matches(deduper) < max_count:
            examples['match'].append(record_pair)
        elif not is_match and count_distinct(deduper) < max_count:
            examples['distinct'].append(record_pair)
        # add classified pair to list
        if len(examples['match']) + len(examples['distinct']) > 0:
            deduper.markPairs(examples)

        # stop, when enough matches and distincts are defined
        if count_matches(deduper) == max_count and \
                count_distinct(deduper) == max_count:
            break


def get_pairs_from_linker():
    result_pair_tuple_list = list()
    for item in linker._blockData(data_1, data_2):
        for item1 in item[0]:
            id1 = str(item1[1]["id"])
            for item2 in item[1]:
                id2 = str(item2[1]["id"])
                result_pair_tuple_list.append((id1, id2))

    return result_pair_tuple_list


# ---------------------- main ----------------------

# Setup

config = tools.get_config(Config_Item)

logging.getLogger().setLevel(logging.WARNING)

# Loading Data
print('importing data ...')
fieldnames = []
for cfg in config.common.fields:
    fieldnames.append(cfg.name)

data_1 = load_data(config.common.filename_1, fieldnames)
data_2 = load_data(config.common.filename_2, fieldnames)
index_perfect_match = tools.load_perfect_match_as_index(config.common.filename_perfect_match)

# Define the fields the linker will pay attention to
fields = []
# create the field-list based on the configuration
for index, cfg in enumerate(config.common.fields):
    field = {'field': cfg.name, 'type': cfg.type}
    if cfg.type.lower() == "text":
        # build the Corpus for the cosine similarity metric using the descriptions
        # These values are used to create a list of rare words
        field["corpus"] = build_corpus(cfg.name)
    fields.append(field)

# ## Test Loop
for config_index, config_item in enumerate(config.items):

    # init Random with a fixes seed (for reproducibility)
    tools.init_random_with_seed()

    # ## Training

    # Create a new linker object and pass our data model to it.
    linker = dedupe.RecordLink(fields)
    # To train the linker, we feed it a sample of records.
    linker.sample(data_1, data_2, 15000)


    # ## Active learning
    # Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as matches or not.
    print('starting active labeling...')

    # dedupe.consoleLabel(linker)
    train_with_perfect_match(linker, config_item.golden_pairs_count, index_perfect_match)

    linker.train()

    # ## Blocking

    # ## Clustering

    # Find the threshold that will maximize a weighted average of our
    # precision and recall.  When we set the recall weight to 2, we are
    # saying we care twice as much about recall as we do precision.
    #
    # If we had more data, we would not pass in all the blocked data into
    # this function but a representative sample.

    print('clustering...')
    linked_records = linker.match(data_1, data_2, 0)

    # ## Writing Results

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.

    record_pairs = []
    cluster_membership = {}
    cluster_id = None
    for cluster_id, (cluster, score) in enumerate(linked_records):
        match_data_1 = []
        match_data_2 = []
        for record_id in cluster:
            cluster_membership[record_id] = (cluster_id, score)
            # search the original ID in the Dataset
            if record_id in data_1:
                match_data_1.append(data_1[record_id]["id"])
            elif record_id in data_2:
                match_data_2.append(data_2[record_id]["id"])

        if len(match_data_1) > 0 and len(match_data_2) > 0:
            for match_1 in match_data_1:
                for match_2 in match_data_2:
                    record_pairs.append([match_1, match_2, "{:.6f}".format(score)])

    # Create Mapping File that can be compared to PerfectMapping
    filename_result = config.common.get_result_file_name(config_index, 'result.csv')
    with open(filename_result, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["idFile1", "idFile2", "Score"])
        for record in record_pairs:
            writer.writerow(record)

    # Evaluating
    filename_result = config.common.get_result_file_name(config_index, 'result.csv')
    add_data = dict({"classifier": "dedupe"}, **config_item.to_dict())
    add_data["classifier"] = "dedupe"
    add_data["config_name"] = config.common.config_name
    add_data["config_item_index"] = config_index
    add_data["fields"] = config.common.fields_to_string()

    result_eval = ev.evaluate_match_file(filename_result, index_perfect_match, get_pairs_from_linker(),
                                         additional_data=add_data)

    ev.print_evaluate_result(result_eval, "Evaluation")
    ev.save_results(config.common.result_base_dir + "log.csv", result_eval)
