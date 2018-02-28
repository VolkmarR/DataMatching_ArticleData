import pandas as pd
import os
import datetime

def evaluate_match_file(match_filename, perfect_match_index):
    """
    evaluates the found matches using the perfect_match_index
    :param match_filename: match dataframe.
    :param perfect_match_index: multindex containing the perfect match id's
    :return: a dictionary containing "perfect_match_total", "match_correct", "match_incorrect"
    """
    if not os.path.isfile(match_filename):
        return {}

    match = pd.read_csv(match_filename, index_col=[0, 1])

    return evaluate_match_index(match.index, perfect_match_index)


def evaluate_match_index(match_index, perfect_match_index, pair_index=None, additional_data=None):
    """
    evaluates the found matches using the perfect_match_index
    :param match_index: multiindex containing the perfect match id's
    :param perfect_match_index: multiindex containing the perfect match id's
    :param pair_index: index of all the pairs
    :param additional_data additional data, that will be added to the result
    :return: a dictionary containing "perfect_match_total", "match_correct", "match_incorrect"
    """

    # create an empty pair index, if none is passed as parameter
    if pair_index is None:
        pair_index = pd.MultiIndex.from_tuples([], names=["id1", "id2"])

    non_match_index = pair_index.difference(match_index)

    # pairs classified as matches that are true matches
    true_positives = (match_index & perfect_match_index).size
    # pairs classified as matches that are true non-matches
    false_positives = match_index.size - true_positives
    # pairs classified as non-matches that are true matches
    false_negatives = (non_match_index & perfect_match_index).size
    # pairs classified as non-matches that are true non-matches
    true_negatives = non_match_index.size - false_negatives

    precision = round(true_positives / (true_positives + false_positives), 5)
    recall = round(true_positives / (true_positives + false_negatives), 5)
    f_measure = round(2 * ((precision * recall) / (precision + recall)), 5)

    result = {
        "execute_date": datetime.date.today().strftime("%Y-%m-%d"),
        "execute_time": datetime.datetime.now().time().strftime("%H:%M:%S"),
        "perfect_match_count": perfect_match_index.size,
        "match_count": match_index.size,
        "pair_count": pair_index.size,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
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
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=list(result.keys()))

    # add missing columns
    for key in result.keys():
        if key not in df.columns:
            df[key] = None

    # add the row
    df = df.append(result, ignore_index=True)

    # save
    df.to_csv(filename, index=False)

