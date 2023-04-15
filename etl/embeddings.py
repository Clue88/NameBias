import pandas as pd

from eec import get_eec_sentences


def get_text_to_embed(pickle_path: str, dest: str, emotion: str):
    """
    Gets all of the embedded sentences for each name in the given dataframe.
    """
    df = pd.read_pickle(pickle_path)
    num_rows = df.shape[0]
    with open(dest, "a") as f:
        for index, row in df.iterrows():
            print(f"Processing {index + 1} of {num_rows}...")
            f.writelines(
                s + "\n" for s in get_eec_sentences(row["name"], row["sex"], emotion)
            )


def get_sentence_embeddings(csv_path: str, dest: str):
    df = pd.read_csv(csv_path, sep="*")
    print(df)


if __name__ == "__main__":
    get_sentence_embeddings("embeddings/embeddings_anger.csv", "out.pickle")
