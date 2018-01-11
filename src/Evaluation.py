import csv
import pandas as pd
import os

baseDir = '..\\data\\AbtBuy\\'


def readFile(filename):
    return pd.read_csv(filename)

def readPerfectMatch(filename):
    result = {}
    data = pd.read_csv(filename)
    for index, row in data.iterrows():
        if row.idFile1 in result:
            result[row.idFile1].append(row.idFile2)
        else:
            result[row.idFile1] = [row.idFile2]
    return result


def evaluate(match_dir, perfect_match):

    matchFile = baseDir + match_dir + "\\Mapping.csv"

    if not os.path.isfile(matchFile):
        return

    print("Processing " + match_dir)
    match = readFile(baseDir + match_dir + "\\Mapping.csv")

    result = {
        "perfect_match_total": len(perfect_match.keys()),
        "match_correct": 0,
        "match_incorrect": 0
    }

    # alle matches durchgehen
    for index, row in match.iterrows():
        id_file_1 = row.idFile1

        # passenden eintrag in der perfect match tabelle suchen
        if id_file_1 in perfect_match:
            if row.idFile2 in perfect_match[id_file_1]:
                result["match_correct"] += 1
            else:
                result["match_incorrect"] += 1

    print("Correct Matches {0}".format(result["match_correct"]))
    print("Incorrect Matches {0}".format(result["match_incorrect"]))
    print("Perfect Matches Total {0}".format(result["perfect_match_total"]))
    print("Missing Matches {0}".format(result["perfect_match_total"] - result["match_correct"]))
    print("")


# -------------------------------- Main ------------------------

perfectMatch = readPerfectMatch(baseDir + "PerfectMapping.csv")
evaluate("dedupe", perfectMatch)
evaluate("prlt", perfectMatch)
