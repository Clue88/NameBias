import json
import os

# The location of the pile_counts directory
PILE_COUNTS = "pile_counts"


def get_all_files(start: int, end: int, prefix: str, suffix: str) -> list:
    """
    Returns all of the files with numbers from start (incl.) to end (excl.)
    with the given suffix.
    """
    files = []
    for i in range(start, end):
        if i < 10:
            files.append(os.path.join(prefix, "0" + str(i) + suffix))
        else:
            files.append(os.path.join(prefix, str(i) + suffix))
    return files


def combine_name_counts(files: list, output: str) -> None:
    """
    Combines the provided name count files into a single text file.
    """
    total_name_counts = {}
    for file in files:
        print(f"Processing {file}...")
        try:
            with open(file, "r", encoding="utf-8") as f:
                name_counts = json.loads(f.read())
                for key in name_counts.keys():
                    if key in total_name_counts:
                        total_name_counts[key] += name_counts[key]
                    else:
                        total_name_counts[key] = name_counts[key]
        except FileNotFoundError:
            print(f"Couldn't find file {file}")
            
    with open(output, "w", encoding="utf-8") as f:
        f.write(json.dumps(total_name_counts))

    print("Done!")


if __name__ == "__main__":
    files = get_all_files(0, 30, PILE_COUNTS, ".txt")
    output = os.path.join(PILE_COUNTS, "combined.txt")
    combine_name_counts(files, output)
