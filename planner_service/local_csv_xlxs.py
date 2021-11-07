import pandas as pd
import os
import sys


def read_ets_dumps(folder_path: str):
    dumps = dict()
    for file in os.listdir(folder_path):
        filename = os.fsdecode(file)
        if filename.endswith(".xlsx"): 
            dumps[os.path.basename] = pd.read_excel(file)
        elif filename.endswith(".csv"):
            dumps[os.path.basename] = pd.read_csv(file)
    print(dumps)


if __name__ == "__main__":
    folder_path = "/home/i4ellendger/code/planner/data/old reports/2020/" #sys.argv[1]
    read_ets_dumps(folder_path)