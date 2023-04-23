import pandas as pd
import glob
from joblib import Parallel, delayed

EMOTION = "anger"


# names
df_names = pd.read_parquet("data/names.parquet",
                           columns=["index", "name", "is_full_name", "frequency_pile", "frequency_FL_corpus", 'n_tokens'])
df_names = df_names[df_names['is_full_name'] == "T"]
assert len(df_names) == df_names['name'].nunique()

# get embedding scores
fpaths = glob.glob(f"embeddings/embeddings_e={EMOTION}*.parquet")
scores = Parallel(n_jobs=-1, verbose=True)(delayed(pd.read_parquet)(fpath, columns=["name_index", "sentence_index", "mean_mean_pool_sim_score", "mean_cls_sim_score"]) for fpath in fpaths)
df_scores = pd.concat(scores, ignore_index=True)
# average score across all sentences
df_scores = df_scores.groupby(["name_index"])[["mean_mean_pool_sim_score", "mean_cls_sim_score"]].mean().reset_index()

# florida voters
df = pd.read_parquet("data/fl_voters.parquet")
df_first_name_map = df.drop_duplicates("full_name")[["full_name", "first_name", "last_name"]]
df = df.groupby(["full_name", "race", "sex"]).size().reset_index(name="count")
df = df.merge(df_first_name_map, on='full_name', how='left')  # merge to get first name and last name
df = df.merge(df_names, left_on="full_name", right_on="name", how="left")  # merge to get index and counts
df = df.merge(df_scores, left_on="index", right_on="name_index", how="left")  # merge to get scores


# check missing senteces
df[df['mean_mean_pool_sim_score'].isnull()]  # name index 415
df_names[df_names['index'] == 415]  # present in names

fpaths = glob.glob(f"data/sentences/sentences_e={EMOTION}*.parquet")
df_sents = pd.read_parquet(fpaths, columns=["name_index", "sentence_index", "name"])
df_sents[df_sents['name_index'] == 415]  # not present in sentences
df_sents[df_sents['name'] == "A Gillan"]  # not present in sentences

# mistokenized names
df['n_spaces'] = df['full_name'].apply(lambda x: len(x.split()))
df.sort_values("n_spaces", ascending=False).head(10)
df.sort_values("frequency_pile", ascending=False).head(10)

df['len_first_name'] = df['first_name'].str.len()
df['len_last_name'] = df['last_name'].str.len()


df.to_parquet("data/reg_data.parquet")