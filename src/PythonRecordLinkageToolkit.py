import recordlinkage as rl
import pandas as pd
import random as rnd
from datetime import datetime
import Evaluation as ev
import Tools as tools


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
    result_index = classifier.predict(features)
    pd.DataFrame(features, result_index).to_csv(filename_result_template.format(filename_key))

    # call the evaluation on the created matches
    add_data = dict({"classifier": type(classifier).__name__}, **additional_config)
    result_eval = ev.evaluate_match_index(result_index, perfect_match_index, pairs_index, add_data)
    ev.print_evaluate_result(result_eval)

    ev.save_results(config.base_dir + "log.csv", result_eval)


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
        is_match = key in perfect_match_index
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

# init the configuration
config = tools.Config('..\\Data\\AbtBuy\\')
filename_result_template = config.base_dir + 'prlt\\result_{}.csv'

additional_config = tools.load_json_config(config.base_dir + "config.json", {"windows": 9, "golden_pairs_count": 25})

print("Load Files")
dfFile1 = tools.load_file_as_df(config.filename_1, ["title", "description"])
dfFile2 = tools.load_file_as_df(config.filename_2, ["title", "description"])
perfect_match_index = tools.load_perfect_match_as_index(config.filename_perfect_match)

print("Indexing")
indexer = rl.SortedNeighbourhoodIndex(on='title', window=additional_config["windows"])
# indexer = rl.FullIndex()
pairs_index = indexer.index(dfFile1, dfFile2)

print("Comparing {0} Pairs".format(pairs_index.size))
compare_cl = rl.Compare()
compare_cl.string('title', 'title', label='title', method='damerau_levenshtein', missing_value=0)
compare_cl.string('title', 'title', label='title_cos', method='cosine', missing_value=0)
compare_cl.string('description', 'description', label='description', method='cosine', missing_value=0)
features = compare_cl.compute(pairs_index, dfFile1, dfFile2)

print("Creating training data")
# golden_pairs = features[0:15000]
# golden_matches_index = golden_pairs.index & idxPM

golden_pairs, golden_matches_index = create_golden_pairs(additional_config["golden_pairs_count"])
# classification

print("Classification")
print("")

predict_and_save(create_and_train_svm(), "svm")

predict_and_save(create_and_train_naive_bayes(), "nb")

predict_and_save(create_and_train_logistic_regression(), "lr")

predict_and_save(create_and_train_kmeans(), "km")

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
