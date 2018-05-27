import pandas as pd
import recordlinkage as rl
import random as rnd
import Tools as tools
from datetime import datetime
from affinegap import normalizedAffineGapDistance
from simplecosine.cosine import CosineTextSimilarity

def dedupe_affine_gap(s1, s2):
    return pd.Series(list(zip(s1, s2))).apply(lambda x: normalizedAffineGapDistance(x[0], x[1]))

def dedupe_cosine(s1, s2):
    s1_2 = pd.Series(list(zip(s1, s2)));

    # build corpus
    corpus_set = []
    for index, value in s1_2.iteritems():
        corpus_set.append(value[0])
        corpus_set.append(value[1])

    # init cosine instance
    cosine = CosineTextSimilarity(corpus_set)

    # calc similarity
    return s1_2.apply(lambda x: cosine(x[0], x[1]))


def run_compare(fieldname, df1, df2, index):
    # list of the string comparers
    compare_methods = ['jaro', 'jarowinkler', 'levenshtein', 'damerau_levenshtein', 'q_gram', 'cosine',
                       'smith_waterman', 'longest_common_substring']

    # build compare-class
    compare_cl = rl.Compare()
    for method in compare_methods:
       compare_cl.string(fieldname, fieldname, label='prlt_' + method, method=method, missing_value=0)

    # dedupe classes
    compare_cl.compare_vectorized(dedupe_affine_gap, fieldname, fieldname, label='dedupe_affine_gap')
    compare_cl.compare_vectorized(dedupe_cosine, fieldname, fieldname, label='dedupe_cosine')

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

def save_binned_result(df_match, df_distinct, bin_count, filename_with_placeholder):
    for col_name in list(df_match)[2:]:
        df = tools.series_to_bins(df_match[col_name], df_distinct[col_name], bin_count, )
        tools.save_csv(df, config.common.result_base_dir + filename_with_placeholder.format(col_name), index=False)


def save_result(df, filename):
    # save the full result
    df.to_csv(config.common.result_base_dir + filename)


def sample_index(index, sample_count):
    # extracts a sample from the index
    return pd.MultiIndex.from_tuples(rnd.sample(list(index), sample_count))


# ------------------------- main ------------------

start_time = datetime.now()

# setup
config = tools.get_config(None)
assert (len(config.common.fields) == 1), "Only one Field is allowed for fields"
fieldname = config.common.fields[0].name

tools.ensure_directories(config.common.result_base_dir + "dummy")

# init Random with a fixes seed (for reproducibility)
tools.init_random_with_seed()

# load files
print("Loading files")
df_1 = tools.load_file_as_df(config.common.filename_1, [fieldname])
df_2 = tools.load_file_as_df(config.common.filename_2, [fieldname])
idx_match = tools.load_perfect_match_as_index(config.common.filename_perfect_match)

# build a full index without the matches
idx_full = rl.FullIndex().index(df_1, df_2)
idx_distinct = sample_index(idx_full.difference(idx_match), len(idx_match) * 10)

# run compare
print("Compare matches")
df_match = run_compare(fieldname, df_1, df_2, idx_match)

print("Compare distincts")
df_distinct = run_compare(fieldname, df_1, df_2, idx_distinct)

# save result
save_result(df_match, 'cm_matches.csv')
save_result(df_distinct, 'cm_distinct.csv')
save_binned_result(df_match, df_distinct, 25, 'cm_bin_{0}.csv')

print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
