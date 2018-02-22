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


def evaluate_match_index(match_index, perfect_match_index):
    """
    evaluates the found matches using the perfect_match_index
    :param match_index: multiindex containing the perfect match id's
    :param perfect_match_index: multiindex containing the perfect match id's
    :return: a dictionary containing "perfect_match_total", "match_correct", "match_incorrect"
    """

    correct = (match_index & perfect_match_index).size

    result = {
        "perfect_match_total": perfect_match_index.size,
        "match_correct": correct,
        "match_incorrect": match_index.size - correct,
        "missing_matches": perfect_match_index.size - correct,
    }

    return result


def print_evaluate_result(result, title=""):
    if result == {}:
        return

    if title != "":
        print(title)
        print("--------------------------")

    print("Correct Matches {0}".format(result["match_correct"]))
    print("Incorrect Matches {0}".format(result["match_incorrect"]))
    print("Perfect Matches Total {0}".format(result["perfect_match_total"]))
    print("Missing Matches {0}".format(result["missing_matches"]))
    print("")



