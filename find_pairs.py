import pandas as pd

from eec import compare_names


def generate_groups():
    """
    Splits the dataframe into groups of equal race, sex, counts, and tokens.
    """
    df = pd.read_pickle("pickles/full_cleaned.pickle")
    groups = {}
    races = ["nh_white", "nh_black", "hispanic", "api", "aian"]
    sexes = ["M", "F"]
    counts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for race in races:
        for sex in sexes:
            for count in counts:
                for token in tokens:
                    groups[race + "_" + sex + "_" + str(count) + "_" + str(token)] = df[
                        df["pred_race"] == race
                    ][df["sex"] == sex][df["frequency_pile"] == count][
                        df["num_tokens"] == token
                    ]

    for group_key in groups.keys():
        print("Processing " + group_key + "...")
        if len(groups[group_key]) > 0:
            groups[group_key].to_pickle("groups/" + group_key)


def sex_letter_to_word(letter):
    if letter == "M":
        return "male"
    else:
        return "female"


def compare_groups(group1, group2):
    """
    Compares each member of one group to each member of another and prints
    the average results.
    """
    df1 = pd.read_pickle(group1)
    df2 = pd.read_pickle(group2)
    df = pd.merge(df1, df2, how="cross")
    df.apply(
        lambda x: compare_names(
            x["name_x"],
            sex_letter_to_word(x["sex_x"]),
            x["name_y"],
            sex_letter_to_word(x["sex_y"]),
            "anger",
            "I feel angry",
        ),
        axis=1,
    )


if __name__ == "__main__":
    # generate_groups()
    compare_groups("groups/nh_white_M_512_4", "groups/nh_white_M_512_2")
