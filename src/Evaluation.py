import csv
import pandas as pd
import os


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


def evaluate_match_index(match_index, perfect_match_index, pair_index = None):
    """
    evaluates the found matches using the perfect_match_index
    :param match_index: multiindex containing the perfect match id's
    :param perfect_match_index: multiindex containing the perfect match id's
    .:param pair_index: index of all the pairs
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

    result = {
        "perfect_match_count": perfect_match_index.size,
        "match_count": match_index.size,
        "pair_count": pair_index.size,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }

    return result


def print_evaluate_result(result, title=""):
    if result == {}:
        return

    if title != "":
        print(title)
        print("--------------------------")

    print("Perfect Matches Total {0}".format(result["perfect_match_count"]))
    print("Matches Total {0}".format(result["match_count"]))
    print("True Positives {0}".format(result["true_positives"]))
    print("False Positives {0}".format(result["false_positives"]))
    print("True Negatives {0}".format(result["true_negatives"]))
    print("False Negatives {0}".format(result["false_negatives"]))
    print("")



