import json
import re
import warnings

import jsonlines
import pandas as pd
from transformers import AutoTokenizer
from sqlite3 import connect

warnings.filterwarnings("ignore")


def make_dataset_first_names():
    """
    Creates a file with each unique first name as well as demographic statistics about each name.
    Saves it as a csv file.
    """
    conn = connect(":memory:")
    df_names = pd.read_csv("../Process_Data/names.csv")
    df_names.to_sql("names", conn)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    df_first_names = pd.read_sql(
        """SELECT first_name AS name,
                                    COUNT(*) AS frequency_FL_corpus,
                                    SUM(CASE WHEN sex='M' THEN 1 ELSE 0 END) AS frequency_male,
                                    SUM(CASE WHEN sex='F' THEN 1 ELSE 0 END) AS frequency_female,
                                    SUM(CASE WHEN sex='U' THEN 1 ELSE 0 END) AS frequency_unknown,
                                    SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) AS frequency_not_reported,
                                    SUM(CASE WHEN race='aian' THEN 1 ELSE 0 END) AS frequency_aian,
                                    SUM(CASE WHEN race='api' THEN 1 ELSE 0 END) AS frequency_api,
                                    SUM(CASE WHEN race='hispanic' THEN 1 ELSE 0 END) AS frequency_hispanic,
                                    SUM(CASE WHEN race='nh_black' THEN 1 ELSE 0 END) AS frequency_nh_black,
                                    SUM(CASE WHEN race='nh_white' THEN 1 ELSE 0 END) AS frequency_nh_white
                                    FROM names
                                    GROUP BY name
                                    ORDER BY name""",
        conn,
    )

    df_first_names["num_tokens"] = 0
    for row_num in range(df_first_names.shape[0]):
        first_name = df_first_names["name"][row_num]
        df_first_names["num_tokens"][row_num] = (
            len(tokenizer(first_name)["input_ids"]) - 2
        )

    df_first_names["is_first_name"] = "T"
    df_first_names["is_last_name"] = "F"
    df_first_names["is_full_name"] = "F"
    df_first_names["frequency_text_corpus"] = 0
    df_first_names["corpus_subset"] = "PILE"
    df_first_names = df_first_names[
        [
            "name",
            "is_first_name",
            "is_last_name",
            "is_full_name",
            "frequency_FL_corpus",
            "frequency_text_corpus",
            "frequency_male",
            "frequency_female",
            "frequency_unknown",
            "frequency_not_reported",
            "frequency_aian",
            "frequency_api",
            "frequency_hispanic",
            "frequency_nh_black",
            "frequency_nh_white",
            "num_tokens",
            "corpus_subset",
        ]
    ]
    df_first_names = df_first_names.astype(
        {
            "name": str,
            "is_first_name": str,
            "is_last_name": str,
            "is_full_name": str,
            "frequency_FL_corpus": int,
            "frequency_text_corpus": int,
            "frequency_male": int,
            "frequency_female": int,
            "frequency_unknown": int,
            "frequency_not_reported": int,
            "frequency_aian": int,
            "frequency_api": int,
            "frequency_hispanic": int,
            "frequency_nh_black": int,
            "frequency_nh_white": int,
            "num_tokens": int,
            "corpus_subset": str,
        }
    )
    df_first_names.to_csv("first_names.csv", index=False)


def make_dataset_last_names():
    """
    Creates a file with each unique last name as well as demographic statistics about each name.
    Saves it as a csv file.
    """
    conn = connect(":memory:")
    df_names = pd.read_csv("../Process_Data/names.csv")
    df_names.to_sql("names", conn)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    df_last_names = pd.read_sql(
        """SELECT last_name AS name,
                                    COUNT(*) AS frequency_FL_corpus,
                                    SUM(CASE WHEN sex='M' THEN 1 ELSE 0 END) AS frequency_male,
                                    SUM(CASE WHEN sex='F' THEN 1 ELSE 0 END) AS frequency_female,
                                    SUM(CASE WHEN sex='U' THEN 1 ELSE 0 END) AS frequency_unknown,
                                    SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) AS frequency_not_reported,
                                    SUM(CASE WHEN race='aian' THEN 1 ELSE 0 END) AS frequency_aian,
                                    SUM(CASE WHEN race='api' THEN 1 ELSE 0 END) AS frequency_api,
                                    SUM(CASE WHEN race='hispanic' THEN 1 ELSE 0 END) AS frequency_hispanic,
                                    SUM(CASE WHEN race='nh_black' THEN 1 ELSE 0 END) AS frequency_nh_black,
                                    SUM(CASE WHEN race='nh_white' THEN 1 ELSE 0 END) AS frequency_nh_white
                                    FROM names
                                    GROUP BY name
                                    ORDER BY name""",
        conn,
    )

    df_last_names["num_tokens"] = 0
    for row_num in range(df_last_names.shape[0]):
        last_name = df_last_names["name"][row_num]
        df_last_names["num_tokens"][row_num] = (
            len(tokenizer(last_name)["input_ids"]) - 2
        )

    df_last_names["is_first_name"] = "F"
    df_last_names["is_last_name"] = "T"
    df_last_names["is_full_name"] = "F"
    df_last_names["frequency_text_corpus"] = 0
    df_last_names["corpus_subset"] = "PILE"
    df_last_names = df_last_names[
        [
            "name",
            "is_first_name",
            "is_last_name",
            "is_full_name",
            "frequency_FL_corpus",
            "frequency_text_corpus",
            "frequency_male",
            "frequency_female",
            "frequency_unknown",
            "frequency_not_reported",
            "frequency_aian",
            "frequency_api",
            "frequency_hispanic",
            "frequency_nh_black",
            "frequency_nh_white",
            "num_tokens",
            "corpus_subset",
        ]
    ]
    df_last_names = df_last_names.astype(
        {
            "name": str,
            "is_first_name": str,
            "is_last_name": str,
            "is_full_name": str,
            "frequency_FL_corpus": int,
            "frequency_text_corpus": int,
            "frequency_male": int,
            "frequency_female": int,
            "frequency_unknown": int,
            "frequency_not_reported": int,
            "frequency_aian": int,
            "frequency_api": int,
            "frequency_hispanic": int,
            "frequency_nh_black": int,
            "frequency_nh_white": int,
            "num_tokens": int,
            "corpus_subset": str,
        }
    )
    df_last_names.to_csv("last_names.csv", index=False)


def make_dataset_full_names():
    """
    Creates a file with each unique full (first + last) name as well as demographic statistics about each name.
    Saves it as a csv file.
    """
    conn = connect(":memory:")
    df_names = pd.read_csv("../Process_Data/names.csv")
    df_names["full_name"] = df_names["first_name"] + " " + df_names["last_name"]
    df_names.to_sql("names", conn)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    df_full_names = pd.read_sql(
        """SELECT full_name AS name,
                                    COUNT(*) AS frequency_FL_corpus,
                                    SUM(CASE WHEN sex='M' THEN 1 ELSE 0 END) AS frequency_male,
                                    SUM(CASE WHEN sex='F' THEN 1 ELSE 0 END) AS frequency_female,
                                    SUM(CASE WHEN sex='U' THEN 1 ELSE 0 END) AS frequency_unknown,
                                    SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) AS frequency_not_reported,
                                    SUM(CASE WHEN race='aian' THEN 1 ELSE 0 END) AS frequency_aian,
                                    SUM(CASE WHEN race='api' THEN 1 ELSE 0 END) AS frequency_api,
                                    SUM(CASE WHEN race='hispanic' THEN 1 ELSE 0 END) AS frequency_hispanic,
                                    SUM(CASE WHEN race='nh_black' THEN 1 ELSE 0 END) AS frequency_nh_black,
                                    SUM(CASE WHEN race='nh_white' THEN 1 ELSE 0 END) AS frequency_nh_white
                                    FROM names
                                    GROUP BY name
                                    ORDER BY name""",
        conn,
    )

    df_full_names["num_tokens"] = 0
    for row_num in range(df_full_names.shape[0]):
        full_name = df_full_names["name"][row_num]
        df_full_names["num_tokens"][row_num] = (
            len(tokenizer(full_name)["input_ids"]) - 2
        )

    df_full_names["is_first_name"] = "F"
    df_full_names["is_last_name"] = "F"
    df_full_names["is_full_name"] = "T"
    df_full_names["frequency_text_corpus"] = 0
    df_full_names["corpus_subset"] = "PILE"
    df_full_names = df_full_names[
        [
            "name",
            "is_first_name",
            "is_last_name",
            "is_full_name",
            "frequency_FL_corpus",
            "frequency_text_corpus",
            "frequency_male",
            "frequency_female",
            "frequency_unknown",
            "frequency_not_reported",
            "frequency_aian",
            "frequency_api",
            "frequency_hispanic",
            "frequency_nh_black",
            "frequency_nh_white",
            "num_tokens",
            "corpus_subset",
        ]
    ]
    df_full_names = df_full_names.astype(
        {
            "name": str,
            "is_first_name": str,
            "is_last_name": str,
            "is_full_name": str,
            "frequency_FL_corpus": int,
            "frequency_text_corpus": int,
            "frequency_male": int,
            "frequency_female": int,
            "frequency_unknown": int,
            "frequency_not_reported": int,
            "frequency_aian": int,
            "frequency_api": int,
            "frequency_hispanic": int,
            "frequency_nh_black": int,
            "frequency_nh_white": int,
            "num_tokens": int,
            "corpus_subset": str,
        }
    )
    df_full_names.to_csv("full_names.csv", index=False)


def make_dataset_all_names():
    """
    Concatenates all files (first_names, last_names, and full_names) into a single file.
    Saves it as a csv file.
    """
    df_first_names = pd.read_csv("first_names.csv")
    df_last_names = pd.read_csv("last_names.csv")
    df_full_names = pd.read_csv("full_names.csv")
    df_all_names = pd.concat(
        [df_first_names, df_last_names, df_full_names], ignore_index=True
    )
    df_all_names.to_csv("all_names.csv", index=False)


def check_file_for_names_set(
    file_nums=["00"], doc_start_num=0, doc_end_num=float("inf")
):
    """
    Creates a dictionary mapping each name to the number of times it appears in a given
    subsection of the PILE dataset. Saves dictionary as .txt file. Each file contains a
    list of documents
    Params:
            file_nums: list of files to include (will make separate dictionary for each file)
            doc_start_num: which number document to start with each file
            doc_end_num: which number document to end with each file
    """
    names_set = set()
    for file_name in ("first_names", "last_names", "full_names"):
        names_df = pd.read_csv(file_name + ".csv")
        for name in names_df["name"]:
            names_set.add(name)
    for file_num in file_nums:
        if len(str(file_num)) == 1:
            file_num = "0" + str(file_num)
        try:
            with jsonlines.open("pile/" + str(file_num) + ".jsonl", "r") as reader:
                names_2_counts = {}
                curr_doc_num = 0
                for doc in reader:
                    if curr_doc_num < doc_start_num:
                        curr_doc_num += 1
                    elif curr_doc_num >= doc_end_num:
                        break
                    else:
                        doc_list = re.split("[^a-zA-Z\-]", doc["text"])
                        for word in doc_list:
                            if word in names_set:
                                if word in names_2_counts:
                                    names_2_counts[word] = names_2_counts[word] + 1
                                else:
                                    names_2_counts[word] = 1
                        for idx in range(len(doc_list) - 1):
                            full_name = doc_list[idx] + " " + doc_list[idx + 1]
                            if full_name in names_set:
                                if full_name in names_2_counts:
                                    names_2_counts[full_name] = (
                                        names_2_counts[full_name] + 1
                                    )
                                else:
                                    names_2_counts[full_name] = 1
                        curr_doc_num += 1
            names_2_counts = {
                name: names_2_counts[name] for name in sorted(names_2_counts)
            }
            with open("pile_counts/" + str(file_num) + ".txt", "w") as f:
                f.write(json.dumps(names_2_counts))
        except FileNotFoundError:
            print("Couldn't find PILE subset", file_num)


def add_counts_to_names_file(file):
    """
    Adds the number of times each name appears in the PILE dataset to
    the name files. Also adds in which PILE subset each name appears.
    Param:
            file: which file to add (first_names, last_names, full_names)
    """
    df_names = pd.read_csv(file + ".csv")
    pile_dicts = []
    for pile_subset in range(30):
        if len(str(pile_subset)) == 1:
            pile_subset = "0" + str(pile_subset)
        try:
            with open("pile_counts/" + str(pile_subset) + ".txt", "r") as f:
                names_2_counts = json.load(f)
                pile_dicts.append(names_2_counts)
        except FileNotFoundError:
            print("Couldn't find dictionary for PILE subset", pile_subset)

    for row_num in range(df_names.shape[0]):
        name = df_names["name"][row_num]
        subsets = ""
        count = 0
        for subset_num, pile_dict in enumerate(pile_dicts):
            if name in pile_dict:
                if len(subsets) < 2:
                    if len(str(subset_num)) == 1:
                        subsets = "0" + str(subset_num)
                    else:
                        subsets = str(subset_num)
                else:
                    if len(str(subset_num)) == 1:
                        subsets = subsets + ", 0" + str(subset_num)
                    else:
                        subsets = subsets + ", " + str(subset_num)
                count += pile_dict[name]
        df_names["frequency_text_corpus"][row_num] = count
        if len(subsets) > 0:
            df_names["corpus_subset"][row_num] = "PILE (" + subsets + ")"
        else:
            df_names["corpus_subset"][row_num] = "N/A"
    df_names.to_csv(file + "_complete.csv", index=False)


def make_dataset_all_names_complete():
    """
    Concatenates all files (first_names_complete, last_names_complete, and
    full_names_complete) into a single file. Saves it as a csv file.
    """
    df_first_names = pd.read_csv("first_names_complete.csv")
    df_last_names = pd.read_csv("last_names_complete.csv")
    df_full_names = pd.read_csv("full_names_complete.csv")
    df_all_names = pd.concat(
        [df_first_names, df_last_names, df_full_names], ignore_index=True
    )
    df_all_names.to_csv("all_names_complete.csv", index=False)


if __name__ == "__main__":
    check_file_for_names_set(file_nums=[i for i in range(30)])
    # add_counts_to_names_file('first_names')
    # add_counts_to_names_file('last_names')
    # add_counts_to_names_file('full_names')
    # make_dataset_all_names_complete()
