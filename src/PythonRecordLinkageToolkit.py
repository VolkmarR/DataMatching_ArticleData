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
        self.golden_pairs_count = json_item.get("golden_pairs_count", 50)
        self.sorted_neighborhood_window = json_item.get("sorted_neighborhood_window", 9)
        self.canopy_threshold_add = json_item.get("canopy_threshold_add", 0.5)
        self.canopy_threshold_remove = json_item.get("canopy_threshold_remove", 0.7)
        self.index_field_name = json_item.get("index_field_name", "")
        self.index_type = json_item.get("index_type", "sorted_neighbourhood")
        self.classifier_types = [x.lower() for x in json_item.get("classifier_types", ["svm"])]
        if isinstance(self.classifier_types, str):
            self.classifier_types = [self.classifier_types]

    def to_dict(self):
        return {"golden_pairs_count": self.golden_pairs_count,
                "index_type": self.index_type,
                "index_field_name": self.index_field_name,
                "canopy_threshold_add": self.canopy_threshold_add,
                "canopy_threshold_remove": self.canopy_threshold_remove,
                "sorted_neighborhood_window": self.sorted_neighborhood_window}


def classifier_abbreviation(classifier):
    if isinstance(classifier, rl.SVMClassifier):
        return "SVM"
    if isinstance(classifier, rl.KMeansClassifier):
        return "KM"
    if isinstance(classifier, rl.NaiveBayesClassifier):
        return "NB"
    if isinstance(classifier, rl.LogisticRegressionClassifier):
        return "LR"
    return "??"


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
    add_data["config_name"] = config.common.config_name
    add_data["config_item_index"] = config_index
    add_data["fields"] = config.common.fields_to_string()
    add_data["classifier"] = type(classifier).__name__
    add_data["classifier_abbreviation"] = classifier_abbreviation(classifier)
    add_data["Indexed_pairs"] = pairs_index.size
    add_data["Indexed_pairs_perfect_match"] = pairs_index.intersection(perfect_match_index).size

    result_eval = ev.evaluate_match_index(result_index, perfect_match_index, add_data)
    ev.print_evaluate_result(result_eval)

    ev.save_results(config.common.result_base_dir + "log.csv", result_eval)


def create_list_of_random_elements(index, max_count):
    """
    extract max_count items from the index and returns them as list
    """

    if isinstance(index, pd.core.index.MultiIndex):
        # if index is a MultiIndex, convert it to list
        elements = index.tolist()
    else:
        # otherwise create a copy of the list
        elements = list(index)
    result = list()

    while len(result) < max_count and elements:
        key = rnd.choice(elements)
        result.append(key)
        elements.remove(key)

    return result


def create_golden_pairs(max_count):
    """
    Creates a sample of the features containing max_count matches and
    max_count distincts
    :return: golden_pair_df, golden_pair_matches_index
    """
    assert (max_count < perfect_match_index.size), "golden_pairs_count is greater then the count of golden pairs"

    # create full match and distinct list
    full_index_match = features.index.intersection(perfect_match_index)
    full_index_distinct = features.index.difference(perfect_match_index)

    train_match = create_list_of_random_elements(full_index_match, max_count)
    train_distinct = create_list_of_random_elements(full_index_distinct, max_count)

    res_pairs = pd.DataFrame(features, pd.MultiIndex.from_tuples(list(set().union(train_match, train_distinct))))
    res_match = pd.MultiIndex.from_tuples(list(train_match))
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
    if config_item.index_type == "sorted_neighbourhood":
        indexer = rl.SortedNeighbourhoodIndex(config_item.index_field_name,
                                              window=config_item.sorted_neighborhood_window)
    elif config_item.index_type == "block":
        indexer = rl.BlockIndex(config_item.index_field_name)
    elif config_item.index_type == "canopy":
        indexer = tools.CanopyClusterIndex(config_item.index_field_name,
                                           threshold_add=config_item.canopy_threshold_add,
                                           threshold_remove=config_item.canopy_threshold_remove)
    elif config_item.index_type == "full":
        indexer = tools.FullIndex(config_item.index_field_name)
    else:
        raise ValueError("index_type {0} is invalid: must be sorted_neighbourhood, block, canopy or full".format(config_item.index_type))

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

    for classifier in config_item.classifier_types:
        if classifier == "svm":
            predict_and_save(create_and_train_svm(), "svm", config_item, index)
        elif classifier == "kmeans":
            predict_and_save(create_and_train_kmeans(), "km", config_item, index)
        elif classifier == "naive_bayes":
            predict_and_save(create_and_train_naive_bayes(), "nb", config_item, index)
        elif classifier == "logistic_regression":
            predict_and_save(create_and_train_logistic_regression(), "lr", config_item, index)
        else:
            raise ValueError("classifier_types {0} is invalid: must be kmeans, svm, naive_bayes or logistic_regression".format(
                config_item.classifier_types))

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
