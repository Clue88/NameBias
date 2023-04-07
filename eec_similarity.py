import os
import numpy as np
import pandas as pd

from eec import compare_single_name


def add_eec_similarity(pickle_path: str, dest: str):
    df = pd.read_pickle(pickle_path)
    df["ref_sim"] = df.apply(
        lambda x: compare_single_name(x["name"], x["sex"], "anger", "I feel angry"),
        axis=1,
    )
    df.to_pickle(dest)


if __name__ == "__main__":
    add_eec_similarity("pickles/full_cleaned.pickle", "pickles/new.pickle")
