#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This code demonstrates how to use RecordLink with two comma separated
values (CSV) files. We have listings of products from two different
online stores. The task is to link products between the datasets.

The output will be a CSV with our linkded results.

"""
from __future__ import print_function
from future.builtins import next

import os
import csv
import collections
import logging
import optparse
import numpy

import dedupe
from Tools import pre_process_string



# ## Logging

# dedupe uses Python logging to show or suppress verbose output. Added for convenience.
# To enable verbose logging, run `python examples/csv_example/csv_example.py -v`
optp = optparse.OptionParser()
optp.add_option('-v', '--verbose', dest='verbose', action='count',
                help='Increase verbosity (specify multiple times for more)'
                )
(opts, args) = optp.parse_args()
log_level = logging.WARNING
if opts.verbose:
    if opts.verbose == 1:
        log_level = logging.INFO
    elif opts.verbose >= 2:
        log_level = logging.DEBUG
logging.getLogger().setLevel(log_level)

# ## Setup

baseDir = '..\\data\\AbtBuySmall\\'
file1 = baseDir + 'file1.csv'
file2 = baseDir + 'file2.csv'
output_file = baseDir + 'dedupe\\data_matching_output.csv'
mapping_file = baseDir + 'dedupe\\mapping.csv'
settings_file = baseDir + 'dedupe\\data_matching_learned_settings'
training_file = baseDir + 'dedupe\\data_matching_training.json'

def readData(filename):
    """
    Read in our data from a CSV file and create a dictionary of records, 
    where the key is a unique record ID.
    """

    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = dict([(k, pre_process_string(v)) for (k, v) in row.items()])
            if clean_row['price']:
                clean_row['price'] = float(clean_row['price'][1:])
            data_d[filename + str(i)] = dict(clean_row)

    return data_d


print('importing data ...')
data_1 = readData(file1)
data_2 = readData(file2)

def descriptions():
    for dataset in (data_1, data_2):
        for record in dataset.values():
            yield record['description']


# ## Training


if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as sf:
        linker = dedupe.StaticRecordLink(sf)

else:
    # Define the fields the linker will pay attention to
    #
    # Notice how we are telling the linker to use a custom field comparator
    # for the 'price' field. 
    fields = [
        {'field': 'title', 'type': 'String'},
        {'field': 'title', 'type': 'Text', 'corpus': descriptions()},
        {'field': 'description', 'type': 'Text',
         'has missing': True, 'corpus': descriptions()},
        {'field': 'price', 'type': 'Price', 'has missing': True}]

    # Create a new linker object and pass our data model to it.
    linker = dedupe.RecordLink(fields)
    # To train the linker, we feed it a sample of records.
    linker.sample(data_1, data_2, 15000)

    # If we have training data saved from a previous run of linker,
    # look for it an load it in.
    # __Note:__ if you want to train from scratch, delete the training_file
    if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)
        with open(training_file) as tf:
            linker.readTraining(tf)

    # ## Active learning
    # Dedupe will find the next pair of records
    # it is least certain about and ask you to label them as matches
    # or not.
    # use 'y', 'n' and 'u' keys to flag duplicates
    # press 'f' when you are finished
    print('starting active labeling...')

    dedupe.consoleLabel(linker)

    linker.train()

    # When finished, save our training away to disk
    with open(training_file, 'w') as tf:
        linker.writeTraining(tf)

    # Save our weights and predicates to disk.  If the settings file
    # exists, we will skip all the training and learning next time we run
    # this file.
    with open(settings_file, 'wb') as sf:
        linker.writeSettings(sf)

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

if cluster_id:
    unique_id = cluster_id + 1
else:
    unique_id = 0

with open(output_file, 'w') as f:
    writer = csv.writer(f)

    header_unwritten = True

    for fileno, filename in enumerate((file1, file2)):
        with open(filename) as f_input:
            reader = csv.reader(f_input)

            if header_unwritten:
                heading_row = next(reader)
                heading_row.insert(0, 'source file')
                heading_row.insert(0, 'Link Score')
                heading_row.insert(0, 'Cluster ID')
                writer.writerow(heading_row)
                header_unwritten = False
            else:
                next(reader)

            for row_id, row in enumerate(reader):
                cluster_details = cluster_membership.get(filename + str(row_id))
                if cluster_details is None:
                    cluster_id = unique_id
                    unique_id += 1
                    score = None
                else:
                    cluster_id, score = cluster_details
                row.insert(0, fileno)
                row.insert(0, score)
                row.insert(0, cluster_id)
                writer.writerow(row)

# Create Mapping File that can be compared to PerfectMapping
with open(mapping_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idFile1", "idFile2", "Score"])
    for record in record_pairs:
        writer.writerow(record)
