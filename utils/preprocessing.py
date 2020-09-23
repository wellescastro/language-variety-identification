import re
import json

def filter_portuguese(text):
    portuguese_texts = re.findall(".+\tpt-[B P][R T]\n", text)
    return portuguese_texts

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="DSL-Task/data/DSLCC-v2.1", help="Directory which contains the 'train' and 'gold' folders")

    args = parser.parse_args()

    basepath = args.data_dir

    # 1. Read the dataset
    for partition in ["train/task1-train.txt", "train/task1-dev.txt", "gold/A.txt"]:
        with open(f"{basepath}/{partition}") as f:
            text_data = f.read()
            text_data = filter_portuguese(text_data)

        new_partition = partition.replace(".txt", "_pt.txt")
        with open(f"{basepath}/{new_partition}", "w") as f:
            f.writelines(text_data)