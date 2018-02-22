import recordlinkage as rl
import pandas as pd
import random as rnd
from datetime import datetime
from Evaluation import evaluate_match_index, print_evaluate_result
from Tools import pre_process_string, load_perfect_match_as_index


def load_file_as_df(filename):
    """
    Loads a Data File. It is expected, that the file contains the following columns:
    unique_id (the identifier column), title, description
    """
    data = pd.read_csv(filename, encoding="iso-8859-1", engine='c', skipinitialspace=True, index_col=[0])
    # call the preprocessing method on the 2 columns title and description
    data["title"] = data["title"].apply(lambda x: pre_process_string(x))
    data["description"] = data["description"].apply(lambda x: pre_process_string(x))
    return data


def train_supervised_classifier(classifier):
    classifier.learn(golden_pairs, golden_matches_index)
    return classifier


def create_and_train_svm():
    """
    Creates and trains a SVM Classifier
    """
    return train_supervised_classifier(rl.SVMClassifier())


def create_and_train_naive_bayes():
    """
    Creates and trains a NaiveBayes Classifier
    """
    return train_supervised_classifier(rl.NaiveBayesClassifier())


def create_and_train_logistic_regression():
    """
    Creates and trains a KMeans Classifier
    """
    return train_supervised_classifier(rl.LogisticRegressionClassifier())


def create_and_train_kmeans():
    """
    Creates and trains a KMeans Classifier
    """
    classifier = rl.KMeansClassifier()
    classifier.learn(features)
    return classifier


def predict_and_save(classifier, filename_key):
    """
    Uses the trained classifier to classify the features and save them as file
    using the result_file_template by adding the filename_key
    """
    result = classifier.predict(features)
    pd.DataFrame(features, result).to_csv(filename_result_template.format(filename_key))

    # call the evaluation on the created matches
    print("Evaluating {0}".format(type(classifier).__name__))

    print_evaluate_result(evaluate_match_index(result, idxPM))


def create_golden_pairs(max_count):
    """
    Creates a sample of the features containing max_count matches and
    max_count distincts
    :return: golden_pair_df, golden_pair_matches_index
    """
    set_match = set()
    set_distinct = set()

    while True:
        key = rnd.choice(features.index)
        is_match = key in idxPM
        if is_match and len(set_match) < max_count:
            set_match.add(key)
        elif not is_match and len(set_distinct) < max_count:
            set_distinct.add(key)

        if len(set_match) == max_count and len(set_distinct) == max_count:
            break

    res_pairs = pd.DataFrame(features, pd.MultiIndex.from_tuples(list(set().union(set_match, set_distinct))))
    res_match = pd.MultiIndex.from_tuples(list(set_match))
    return res_pairs, res_match

# ------------------ Main ---------------

start_time = datetime.now()

baseDir = '..\\Data\\AbtBuy\\'
filename_1 = baseDir + 'file1.csv'
filename_2 = baseDir + 'file2.csv'
filename_perfect_match = baseDir + 'PerfectMapping.csv'
filename_result_template = baseDir + 'prlt\\result_{}.csv'

print("Load Files")
dfFile1 = load_file_as_df(filename_1)
dfFile2 = load_file_as_df(filename_2)
idxPM = load_perfect_match_as_index(filename_perfect_match)

print("Indexing")
indexer = rl.SortedNeighbourhoodIndex(on='title', window=9)
pairs = indexer.index(dfFile1, dfFile2)

print("Comparing")
compare_cl = rl.Compare()
compare_cl.string('title', 'title', label='title', method='damerau_levenshtein', missing_value=0)
compare_cl.string('description', 'description', label='description', method='cosine', missing_value=0)
features = compare_cl.compute(pairs, dfFile1, dfFile2)


print("Creating training data")
#golden_pairs = features[0:15000]
#golden_matches_index = golden_pairs.index & idxPM

golden_pairs, golden_matches_index = create_golden_pairs(25)
# classification

print("Classification")
print("")

predict_and_save(create_and_train_svm(), "svm")

predict_and_save(create_and_train_naive_bayes(), "nb")

predict_and_save(create_and_train_logistic_regression(), "lr")

predict_and_save(create_and_train_kmeans(), "km")

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
