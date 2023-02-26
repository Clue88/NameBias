import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

mpl.rcParams["agg.path.chunksize"] = 10000


def show_frequency_plot(pickle_path: str):
    """
    Displays a plot of the number of occurrences of each name in PILE (y-axis)
    vs. the frequency rank of the name in the FL voter registry (x-axis), using
    a log-log scale.
    """

    df = pd.read_pickle(pickle_path)
    df.sort_values("frequency_FL_corpus", ascending=False).reset_index(drop=True).plot(
        y="count", loglog=True
    )
    plt.show()


def show_race_subplots(pickle_path: str):
    """
    nh_white: red
    nh_black: blue
    hispanic: green
    api: magenta
    aian: black
    """

    df = pd.read_pickle(pickle_path)
    df[df.is_full_name == "F"].plot.scatter(x="frequency_pile", y="frequency_FL_corpus", s=0.25, c="color", alpha=0.25, loglog=True)
    df[df.is_full_name == "F"][df.pred_race == "nh_white"].plot.scatter(x="frequency_pile", y="frequency_FL_corpus", s=0.25, c="color", alpha=0.25, loglog=True)
    df[df.is_full_name == "F"][df.pred_race == "nh_black"].plot.scatter(x="frequency_pile", y="frequency_FL_corpus", s=0.25, c="color", alpha=0.25, loglog=True)
    df[df.is_full_name == "F"][df.pred_race == "hispanic"].plot.scatter(x="frequency_pile", y="frequency_FL_corpus", s=0.25, c="color", alpha=0.25, loglog=True)
    df[df.is_full_name == "F"][df.pred_race == "api"].plot.scatter(x="frequency_pile", y="frequency_FL_corpus", s=0.25, c="color", alpha=0.25, loglog=True)
    df[df.is_full_name == "F"][df.pred_race == "aian"].plot.scatter(x="frequency_pile", y="frequency_FL_corpus", s=0.25, c="color", alpha=1, loglog=True)
    plt.show()


def export_race_subplots(pickle_path: str):
    """
    """
    df = pd.read_pickle(pickle_path)
    fig = px.scatter(df[df.is_full_name == "F"], x="frequency_pile", y="frequency_FL_corpus", color="pred_race", opacity=0.25, log_x=True, log_y=True)
    fig.write_html("race.html")


def show_race_jointplot(pickle_path: str):
    """
    """
    df = pd.read_pickle(pickle_path)
    df = df[df.is_full_name == "F"]
    # df = df[(df["pred_race"] == "nh_white") | (df["pred_race"] == "nh_black")]
    g = sns.jointplot(df[df["pred_race"] == "nh_white"], x="frequency_pile", y="frequency_FL_corpus", hue="pred_race")
    g.ax_joint.set_xscale("log")
    g.ax_joint.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    PICKLE_PATH = "pickles/cleaned_data.pickle"
    # show_frequency_plot(PICKLE_PATH)
    # show_race_subplots(PICKLE_PATH)
    # export_race_subplots(PICKLE_PATH)
    show_race_jointplot(PICKLE_PATH)
