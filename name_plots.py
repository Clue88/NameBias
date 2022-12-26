import pickle
import matplotlib.pyplot as plt


def create_name_plots(pickle_path: str) -> None:
    df = pickle.load(open(pickle_path, "rb"))
    df["name_len"] = df.iloc[:, 0].apply(lambda x: len(x.split()))
    df.sort_values("count", ascending=False, inplace=True)

    df = df[df.name_len == 1]["count"].reset_index()
    ax = df.plot(xlabel="log(rank)", ylabel="log(count)", y="count", loglog=True)
    ax.get_legend().set_visible(False)

    plt.show()


if __name__ == "__main__":
    create_name_plots("pile_counts/combined.pickle")
