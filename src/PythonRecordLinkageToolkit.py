import recordlinkage as rl
import pandas as pd
import random as rnd
from datetime import datetime
import Evaluation as ev
import Tools as tools


class Config_Item:
    """
    Config item class for PRLT
    """
    def __init__(self, json_item):
        self.golden_pairs_count = json_item["golden_pairs_count"]
        self.sorted_neighborhood_window = json_item["sorted_neighborhood_window"]
        self.sorted_neighborhood_field_name = json_item["sorted_neighborhood_field_name"]

    def to_dict(self):
        return {"golden_pairs_count": self.golden_pairs_count,
                "sorted_neighborhood_window": self.sorted_neighborhood_window,
                "sorted_neighborhood_field_name": self.sorted_neighborhood_field_name}


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


def predict_and_save(classifier, filename_key, current_config_item, config_index):
    """
    Uses the trained classifier to classify the features and save them as file
    using the result_file_template by adding the filename_key
    """
    # predict the matches
    result_index = classifier.predict(features)

    # save the file
    result_filename = config.common.get_result_file_name(config_index, "result_{}.csv".format(filename_key))
    pd.DataFrame(features, result_index).to_csv(result_filename)

    # call the evaluation on the created matches
    add_data = current_config_item.to_dict()
    add_data["classifier"] = type(classifier).__name__
    add_data["config_item_index"] = config_index
    result_eval = ev.evaluate_match_index(result_index, perfect_match_index, pairs_index, add_data)
    ev.print_evaluate_result(result_eval)

    ev.save_results(config.common.result_base_dir + "log.csv", result_eval)


def create_golden_pairs(max_count):
    """
    Creates a sample of the features containing max_count matches and
    max_count distincts
    :return: golden_pair_df, golden_pair_matches_index
    """
    assert (max_count < perfect_match_index.size), "golden_pairs_count is greater then the count of golden pairs"

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
config = tools.get_config(Config_Item)

print("Load Files")
fieldnames = []
for cfg in config.common.fields:
    fieldnames.append(cfg.name)

dfFile1 = tools.load_file_as_df(config.common.filename_1, fieldnames)
dfFile2 = tools.load_file_as_df(config.common.filename_2, fieldnames)
perfect_match_index = tools.load_perfect_match_as_index(config.common.filename_perfect_match)

# for each config item
for index, config_item in enumerate(config.items):
    # init Random with a fixes seed (for reproducibility)
    tools.init_random_with_seed()

    print("Indexing")
    indexer = rl.SortedNeighbourhoodIndex(on=config_item.sorted_neighborhood_field_name, window=config_item.sorted_neighborhood_window)
    # indexer = rl.FullIndex()
    pairs_index = indexer.index(dfFile1, dfFile2)

    print("Comparing {0} Pairs".format(pairs_index.size))
    compare_cl = rl.Compare()
    for cfg in config.common.fields:
        compare_cl.string(s1=cfg.name, s2=cfg.name, method=cfg.type)
    features = compare_cl.compute(pairs_index, dfFile1, dfFile2)

    print("Creating training data")
    golden_pairs, golden_matches_index = create_golden_pairs(config_item.golden_pairs_count)
    # classification

    print("Classification")
    print("")

    predict_and_save(create_and_train_svm(), "svm", config_item, index)
    predict_and_save(create_and_train_naive_bayes(), "nb", config_item, index)
    predict_and_save(create_and_train_logistic_regression(), "lr", config_item, index)
    predict_and_save(create_and_train_kmeans(), "km", config_item, index)

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
