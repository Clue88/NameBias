import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import racebert

mpl.rcParams["agg.path.chunksize"] = 10000


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
    model = racebert.RaceBERT()
    df["pred_race"] = df.name.apply(lambda x: model.predict_race(x)[0]["label"])
    df.to_pickle(dest)


def create_frequency_plot(pickle_path: str):
    """
    Displays a plot of the number of occurrences of each name in PILE (y-axis)
    vs. the frequency rank of the name in the FL voter registry (x-axis), using
    a log-log scale.
    """

    df = pd.read_pickle(pickle_path)

    # df[df.is_first_name == "T"].sort_values("frequency_FL_corpus", ascending=False).reset_index(drop=True).plot(y="count", loglog=True)
    # df[df.is_last_name == "T"].sort_values("frequency_FL_corpus", ascending=False).reset_index(drop=True).plot(y="count", loglog=True)
    # df[df.is_full_name == "T"].sort_values("frequency_FL_corpus", ascending=False).reset_index(drop=True).plot(y="count", loglog=True)
    df.sort_values("frequency_FL_corpus", ascending=False).reset_index(drop=True).plot(y="count", loglog=True)
    plt.show()


if __name__ == "__main__":
    PILE_PICKLE_PATH = "pickles/pile_data.pickle"
    FL_PICKLE_PATH = "pickles/fl_data.pickle"
    COMBINED_PICKLE_PATH = "pickles/combined.pickle"
    COMBINED_RACE_PICKLE_PATH = "pickles/combined_race.pickle"

    # combine_pile_fl_data(PILE_PICKLE_PATH, FL_PICKLE_PATH, "pickles/combined.pickle")
    add_racebert_predictions(COMBINED_PICKLE_PATH, COMBINED_RACE_PICKLE_PATH)
    # create_frequency_plot(COMBINED_PICKLE_PATH)
