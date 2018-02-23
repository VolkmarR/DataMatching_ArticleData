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
from Tools import pre_process_string, load_perfect_match_as_index, Config
from Evaluation import evaluate_match_file, print_evaluate_result


def load_data(filename):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID.
    """

    data_d = {}

    with open(filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for i, row in enumerate(reader):
            clean_row = dict([(k, pre_process_string(v)) for (k, v) in row.items()])
            if clean_row['price']:
                clean_row['price'] = float(clean_row['price'][1:])
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
            ids.append(int(pair["unique_id"]))
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


# ---------------------- main ----------------------

# Setup

config = Config('..\\Data\\AbtBuy\\')
filename_result = config.base_dir + 'dedupe\\result.csv'

logging.getLogger().setLevel(logging.WARNING)

# Loading Data
print('importing data ...')
data_1 = load_data(config.filename_1)
data_2 = load_data(config.filename_2)
index_perfect_match = load_perfect_match_as_index(config.filename_perfect_match)

# ## Training


# Define the fields the linker will pay attention to
#
# Notice how we are telling the linker to use a custom field comparator
# for the 'price' field.

# build the Corpus for the cosine similarity metric using the descriptions
# These values are  used to create a list of rare words
corpus = build_corpus("description")

"""
fields = [
    {'field': 'title', 'type': 'String'},
    {'field': 'title', 'type': 'Text', 'corpus': descriptions()},
    {'field': 'description', 'type': 'Text',
     'has missing': True, 'corpus': descriptions()},
    {'field': 'price', 'type': 'Price', 'has missing': True}]
"""
fields = [
    {'field': 'title', 'type': 'String'},
    {'field': 'title', 'type': 'Text', 'corpus': corpus},
    {'field': 'description', 'type': 'Text', 'has missing': True, 'corpus': corpus}]


# Create a new linker object and pass our data model to it.
linker = dedupe.RecordLink(fields)
# To train the linker, we feed it a sample of records.
linker.sample(data_1, data_2, 15000)


# ## Active learning
# Dedupe will find the next pair of records
# it is least certain about and ask you to label them as matches or not.
print('starting active labeling...')

# dedupe.consoleLabel(linker)
train_with_perfect_match(linker, 10, index_perfect_match)

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

print('# duplicate sets', len(linked_records))

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
            match_data_1.append(data_1[record_id]["unique_id"])
        elif record_id in data_2:
            match_data_2.append(data_2[record_id]["unique_id"])

    if len(match_data_1) > 0 and len(match_data_2) > 0:
        for match_1 in match_data_1:
            for match_2 in match_data_2:
                record_pairs.append([match_1, match_2, "{:.6f}".format(score)])


# Create Mapping File that can be compared to PerfectMapping
with open(filename_result, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idFile1", "idFile2", "Score"])
    for record in record_pairs:
        writer.writerow(record)

# Evaluating
print_evaluate_result(evaluate_match_file(filename_result, index_perfect_match), "Evaluation")
