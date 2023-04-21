import pandas as pd
import numpy as np

# create reference sentences
df = pd.read_csv("data/Equity-Evaluation-Corpus.csv", header=0)
df = df.drop_duplicates("Emotion word")[["Emotion", "Emotion word"]]
df = df[df['Emotion'].notnull()].reset_index()
df['Sentence'] = df['Emotion word'].apply(lambda x: f"I feel {x}.")


df['sentence_index'] = df.index
df['name_index'] = np.nan
df.to_parquet("data/sentences/sentences_e=reference_p=0.parquet")



