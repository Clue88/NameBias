import pandas as pd
from numba import njit, jit
import numpy as np
import dask.dataframe as da

EMOTION = "anger"
N_CHUNKS = 4


# emotion data
df_eec = pd.read_csv("data/Equity-Evaluation-Corpus.csv", header=0)

# names data
df_names = pd.read_parquet("data/names.parquet")
df_names = df_names[df_names["is_full_name"] == "T"][
    ["name", "frequency_male", "frequency_female"]
]
df_names = df_names.reset_index(drop=False).rename(columns={"index": "name_index"})

df_male_names = df_names[df_names["frequency_male"] > 0]
df_female_names = df_names[df_names["frequency_female"] > 0]


def get_template_sentences(emotion, gender):
    gender_name_map = {"male": "Alonzo", "female": "Katie"}
    person_name = gender_name_map[gender]
    sentences = df_eec[
        (df_eec["Person"] == person_name) & (df_eec["Emotion"] == emotion)
    ]["Sentence"].str.replace(person_name, "{}")
    return sentences


male_sentences = get_template_sentences(EMOTION, "male")
female_sentences = get_template_sentences(EMOTION, "female")


df_male_name_sents = df_male_names.merge(male_sentences, how="cross")
df_female_name_sents = df_female_names.merge(female_sentences, how="cross")


@jit
def get_name_sentence_pairs(df):
    names = df["name"].values
    sents = df["Sentence"].values
    n = len(names)
    name_sents = []
    for i in range(n):
        name_sents.append(sents[i].format(names[i]))
        if i % 1000000 == 0:
            print(i)
    return name_sents


df_male_name_sents["Sentence"] = get_name_sentence_pairs(df_male_name_sents)
df_female_name_sents["Sentence"] = get_name_sentence_pairs(df_female_name_sents)


df_sents = pd.concat([df_male_name_sents, df_female_name_sents], axis=0)
df_sents = df_sents.drop_duplicates("Sentence").reset_index(drop=True)
df_sents["sentence_index"] = df_sents.index


@jit
def get_sentence_length(sentences):
    sen_lengths = []
    for i, s in enumerate(sentences):
        sen_lengths.append(len(s.split()))
        if i % 1000000 == 0:
            print(i)
    return sen_lengths


# batching
# df_sents['n_tokens'] = get_sentence_length(df_sents['Sentence'].values)
# df_sents.sample(5)
# df_sents.sort_values("n_tokens", ascending=False, inplace=True)  # for batching


for p, df_part in enumerate(np.array_split(df_sents, N_CHUNKS)):
    print(p)
    df_part.to_parquet(f"data/sentences/sentences_e={EMOTION}_p={p}.parquet")


# df_names[(df_names['frequency_female'] > 0) & (df_names['frequency_male'] == 0)]
# df_names[(df_names['frequency_female'] == 0) & (df_names['frequency_male'] > 0)]
# df_names[(df_names['frequency_female'] > 0) & (df_names['frequency_male'] > 0)]
