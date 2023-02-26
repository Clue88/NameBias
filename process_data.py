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


if __name__ == "__main__":
    PILE_PICKLE_PATH = "pickles/pile_data.pickle"
    FL_PICKLE_PATH = "pickles/fl_data.pickle"
    COMBINED_PICKLE_PATH = "pickles/combined.pickle"
    COMBINED_RACE_PICKLE_PATH = "pickles/combined_race.pickle"
