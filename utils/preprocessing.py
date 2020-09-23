import re
# Script to filter portuguese samples for simplifying the dataset
data_basepath = "DSL-Task/data/DSLCC-v2.1"
# 1. Read the dataset
for partition in ["train/task1-train.txt", "train/task1-dev.txt", "gold/A.txt"]:
    with open(f"{data_basepath}/{partition}") as f:
        text_data = f.read()
        portuguese_texts = re.findall(".+\tpt-[B P][R T]\n", text_data)
        print("Preprocessing {} portugues samples for partition {}".format(len(portuguese_texts), partition))
    
    new_partition = partition.replace(".txt", "_pt.txt")
    with open(f"{data_basepath}/{new_partition}", "w") as f:
        f.writelines(portuguese_texts)