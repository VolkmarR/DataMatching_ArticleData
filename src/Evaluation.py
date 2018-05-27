import pandas as pd
import os
import datetime

def evaluate_match_file(match_filename, perfect_match_index, additional_data=None):
    """
    evaluates the found matches using the perfect_match_index
    :param match_filename: match dataframe.
    :param perfect_match_index: multindex containing the perfect match id's
    :return: a dictionary containing "perfect_match_total", "match_correct", "match_incorrect"
    """
    if not os.path.isfile(match_filename):
        return {}

    # load matches and create a list of tuples
    match_tuples = []
    for index, row in pd.read_csv(match_filename, converters={0: str, 1: str}).iterrows():
        match_tuples.append((row[0], row[1]))
    match_index = pd.MultiIndex.from_tuples(match_tuples, names=["id1", "id2"])

    return evaluate_match_index(match_index, perfect_match_index, additional_data)


def evaluate_match_index(match_index, perfect_match_index, additional_data=None):
    """
    evaluates the found matches using the perfect_match_index
    :param match_index: multiindex containing the perfect match id's
    :param perfect_match_index: multiindex containing the perfect match id's
    :param additional_data additional data, that will be added to the result
    :return: a dictionary containing "perfect_match_total", "match_correct", "match_incorrect"
    """

    # pairs classified as matches that are true matches
    true_positives = (match_index & perfect_match_index).size
    # pairs classified as matches that are true non-matches
    false_positives = match_index.size - true_positives
    # pairs classified as non-matches that are true matches
    false_negatives = (perfect_match_index.difference(match_index)).size

    # calculate precision and recall
    if true_positives + false_positives != 0:
        precision = round(true_positives / (true_positives + false_positives), 3)
        recall = round(true_positives / (true_positives + false_negatives), 3)
    else:
        precision = 0.0
        recall = 0.0

    # calculate f_measure
    if precision + recall != 0:
        f_measure = round(2 * ((precision * recall) / (precision + recall)), 3)
    else:
        f_measure = 0.0

    result = {
        "Execute Date": datetime.date.today().strftime("%Y-%m-%d"),
        "Execute Time": datetime.datetime.now().time().strftime("%H:%M:%S"),
        "Perfect Match Count": perfect_match_index.size,
        "Match Count": match_index.size,
        "True Positives": true_positives,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F-Measure": f_measure,
    }

    # add the additional_data to the result
    if not (additional_data is None):
        for key, value in additional_data.items():
            result[key] = value

    return result


def print_evaluate_result(result, title=""):
    if result == {}:
        return

    if title != "":
        print(title)
        print("--------------------------")

    for key, value in result.items():
        print("{0}: {1}".format(key, value))
    print("")


def save_results(filename, result):
    # load file, if existing
    if os.path.isfile(filename):
        df = pd.read_csv(filename, sep=";", decimal=",")
    else:
        df = pd.DataFrame(columns=list(result.keys()))

    # add missing columns
    for key in result.keys():
        if key not in df.columns:
            df[key] = None

    # add the row
    df = df.append(result, ignore_index=True)

    # save
    df.to_csv(filename, index=False, sep=";", decimal=",")

