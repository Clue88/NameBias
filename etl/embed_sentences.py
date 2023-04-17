from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import numpy as np
import pandas as pd
import torch

EMBEDDING = BertEmbeddings("bert-large-uncased")


def embed_sentence(sent):
    sent = Sentence(sent)
    EMBEDDING.embed(sent)
    se = torch.stack([token.embedding for token in sent])
    return np.average(se.cpu().numpy(), axis=0)


if __name__ == "__main__":
    df = pd.read_parquet("embeddings_sadness.parquet")
    df = df.head(1000)
    df["embedding"] = df["sentence"].apply(lambda x: embed_sentence(x))
    # df.to_parquet("embeddings_sadness_new.parquet")
