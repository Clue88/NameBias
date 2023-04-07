import os
import numpy as np
import pandas as pd

from eec import compare_single_name

ARRAY_N_TASKS = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
ARRAY_TASK_NUM = int(os.environ["SLURM_ARRAY_TASK_ID"])

def add_eec_similarity(pickle_path: str, dest: str):
    df = pd.read_pickle(pickle_path)

    df = df.head(101)

    df = np.array_split(df, ARRAY_N_TASKS)[ARRAY_TASK_NUM]
    df["ref_sim"] = df.apply(
        lambda x: compare_single_name(x["name"], x["sex"], "anger", "I feel angry"),
        axis=1,
    )
    df.to_pickle(dest + str(ARRAY_TASK_NUM))


if __name__ == "__main__":
    add_eec_similarity("pickles/full_cleaned.pickle", "pickles/new.pickle")
