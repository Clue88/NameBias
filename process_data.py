import math

import pandas as pd
import racebert


def fl_data_to_pickle(csv_path: str, dest: str):
    """
    Save the Florida voter registry data to a sorted Pickle file.
    """
    fl_df = pd.read_csv(csv_path)
    fl_df.sort_values("name", ascending=True, inplace=True)

    fl_df.to_pickle(dest)


def combine_pile_fl_data(pile_pickle: str, fl_pickle: str, dest: str):
    """
    Combines PILE and FL data into a single Pickle file.
    """
    pile_df = pd.read_pickle(pile_pickle)
    fl_df = pd.read_pickle(fl_pickle)

    combined_df = pd.merge_ordered(pile_df, fl_df, on="name")
    combined_df.fillna(value={"count": 0}, inplace=True)
    combined_df.to_pickle(dest)


def add_racebert_predictions(pickle_path: str, dest: str):
    """
    Adds RaceBERT predictions to combined PILE and FL data.
    """
    df = pd.read_pickle(pickle_path)
    df["first_letter"] = df.name.apply(lambda x: x[0])
    model = racebert.RaceBERT()
    df["pred_race"].fillna(
        value=df[df.first_letter == "A"][df.is_first_name == "T"].name.apply(
            lambda x: model.predict_race(x)[0]["label"]
        ),
        inplace=True,
    )
    df.to_pickle(dest)


def reformat_dataframe(pickle_path: str, dest: str):
    """
    Renames columns and cleans up the data.
    """
    df = pd.read_pickle(pickle_path)
    df.rename(columns={"count": "frequency_pile"}, inplace=True)
    df.drop(
        columns=[
            "first_letter",
            "frequency_text_corpus",
            "frequency_unknown",
            "frequency_not_reported",
            "frequency_aian",
            "frequency_api",
            "frequency_hispanic",
            "frequency_nh_black",
            "frequency_nh_white",
            "corpus_subset",
        ],
        inplace=True,
    )
    colors = {
        "nh_white": "red",
        "nh_black": "blue",
        "hispanic": "green",
        "api": "magenta",
        "aian": "black",
    }
    df["color"] = df["pred_race"].apply(lambda x: colors[x])

    df["frequency_pile_rounded"] = df["frequency_pile"].apply(
        lambda x: 0 if x == 0 else round(2 ** round(math.log(x, 2)))
    )
    df["frequency_FL_rounded"] = df["frequency_FL_corpus"].apply(
        lambda x: 0 if x == 0 else round(2 ** round(math.log(x, 2)))
    )
    df["sex"] = df.apply(
        lambda x: "M" if x["frequency_male"] >= x["frequency_female"] else "F", axis=1
    )

    df.to_pickle(dest)


if __name__ == "__main__":
    PICKLE_PATH = "pickles/new_combined_race.pickle"
    reformat_dataframe(PICKLE_PATH, "pickles/full_cleaned.pickle")
