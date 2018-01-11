import recordlinkage
import pandas as pd
from unidecode import unidecode
import re
import csv


baseDir = '..\\Data\\AbtBuySmall\'
file1 = baseDir + 'file1.csv'
file2 = baseDir + 'file2.csv'
output_file = baseDir + 'prlt\\data_matching_output.csv'
mapping_file = baseDir + 'prlt\\mapping.csv'
features_file = baseDir + 'prlt\\features.csv'
# settings_file = baseDir + 'data_matching_learned_settings'
# training_file = baseDir + 'data_matching_training.json'

def preProcess(column):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """

    if type(column) in (float, int):
        column = ""

    column = unidecode(column)
    column = re.sub('\n', ' ', column)
    column = re.sub('-', '', column)
    column = re.sub('/', ' ', column)
    column = re.sub("'", '', column)
    column = re.sub(",", '', column)
    column = re.sub(":", ' ', column)
    column = re.sub('  +', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    if not column:
        column = None
    return column

def readFile(filename, dtypes):
    data = pd.read_csv(filename, encoding="iso-8859-1", engine='c', skipinitialspace=True, dtype=dtypes)

    # preprocession
    for index, row in data.iterrows():
        row["title"] = preProcess(row["title"])
        row["description"] = preProcess(row["description"])
    return data


# ------------------ Main ---------------
dfFile1 = readFile(file1, {"title": object, "description": object, "price": object})
dfFile2 = readFile(file2, {"title": object, "description": object, "manufacturer": object, "price": object})

# Indexing
#indexer = recordlinkage.SortedNeighbourhoodIndex(on="description")
indexer = recordlinkage.FullIndex()
pairs = indexer.index(dfFile1, dfFile2)

# comparing
compare_cl = recordlinkage.Compare()
compare_cl.string('title', 'title', label='title', missing_value="")
compare_cl.string('description', 'description', label='description', missing_value="")
features = compare_cl.compute(pairs, dfFile1, dfFile2)

# classification

features.to_csv(features_file)

matches = features[features.max(axis=1) > 0.65]
# kmeans = recordlinkage.KMeansClassifier()
# matches = kmeans.learn(features)

print("Anzahl Matches {0}".format(len(matches)))

# Create output
with open(mapping_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["idFile1", "idFile2", "Score"])
    for index, row in matches.iterrows():
        writer.writerow([dfFile1.iloc[row.name[0]]["unique_id"],
                         dfFile2.iloc[row.name[1]]["unique_id"]])
