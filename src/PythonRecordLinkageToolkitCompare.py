import pandas as pd
import recordlinkage as rl
import random as rnd
from Tools import load_file_as_df, load_perfect_match_as_index, ensure_directories, Config


def run_compare(fieldname, df1, df2, index):
    # list of the string comparers
    compare_methods = ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'q_gram', 'cosine',
                       'smith_waterman', 'longest_common_substring']

    # build compare-class
    compare_cl = rl.Compare()
    for method in compare_methods:
        compare_cl.string(fieldname, fieldname, label=method, method=method, missing_value=0)

    # calculate features
    features = compare_cl.compute(index, df1, df2)

    # add the original values
    features.insert(0, 'file1', '')
    features.insert(1, 'file2', '')
    for index, row in features.iterrows():
        # set values for the row
        features.loc[index, 'file1'] = df1.loc[index[0], fieldname]
        features.loc[index, 'file2'] = df2.loc[index[1], fieldname]

    # return the created dataframe
    return features


def sample_index(index, sample_count):
    return pd.MultiIndex.from_tuples(rnd.sample(list(index), sample_count))


# ------------------------- main ------------------

# setup
config = Config('..\\Data\\AbtBuy\\')
ensure_directories(config.base_dir + 'compare\\dummy')

# init Random with a fixes seed (for reproducibility)
rnd.seed(19740327)

# load files
df_1 = load_file_as_df(config.filename_1, ["title"])
df_2 = load_file_as_df(config.filename_2, ["title"])
idx_match = load_perfect_match_as_index(config.filename_perfect_match)

# build a full index without the matches
idx_full = rl.FullIndex().index(df_1, df_2)
idx_distinct = idx_full.difference(idx_match)


# run compare
run_compare('title', df_1, df_2, idx_match).\
    to_csv(config.base_dir + 'compare\\Compare_methods_matches.csv')

run_compare('title', df_1, df_2, sample_index(idx_distinct, 1000)).\
    to_csv(config.base_dir + 'compare\\Compare_methods_distinct_1000.csv')
