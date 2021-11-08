import pandas as pd
import os
import sys
import datetime
import logging
import typing


logging.basicConfig(level=logging.DEBUG)


def read_ets_dumps(folder_path: str) -> typing.Dict[str, pd.DataFrame]:
    dumps = dict()
    total_start_time = datetime.datetime.now()
    for file in os.listdir(folder_path):
        filename = os.fsdecode(file)
        start_time = datetime.datetime.now()
        if filename.lower().endswith(".xlsx"):
            dumps[filename] = pd.read_excel(os.path.join(folder_path, filename), header=1)  # 2nd row
            logging.debug(f'Parsed Excel "{filename}" in {datetime.datetime.now() - start_time}.')
        elif filename.lower().endswith(".csv"):
            dumps[filename] = pd.read_csv(os.path.join(folder_path, filename), header=0)  # 1st row
            logging.debug(f'Parsed CSV "{filename}" in {datetime.datetime.now() - start_time}.')
    logging.info(f"Parsed {len(dumps)} files in {datetime.datetime.now() - total_start_time}.")
    return dumps


def read_ets_dumps_and_merge(folder_path: str) -> pd.DataFrame:
    dumps = read_ets_dumps(folder_path)
    data = pd.DataFrame()
    for k, v in dumps.items():
        new_data = v.dropna()
        data = data.append(new_data, ignore_index=True)
        logging.debug(f"Merged {len(new_data)} rows from '{k}' file into common dataframe with {len(data)} rows."
                      f" Columns {list(new_data.columns)} -> {list(data.columns)}.")
    data.sort_values(by='Date (MM/DD/YYYY)', inplace=True)
    logging.info(f"Parsed {list(data.columns)} columns and {len(data)} rows from '{folder_path}' files'.")
    logging.debug(data.describe)
    return data


if __name__ == "__main__":
    folder_path = "/home/i4ellendger/code/planner/data/old reports/2020/" #sys.argv[1]
    data = read_ets_dumps_and_merge(folder_path)
    data.to_csv("dump.csv", index=False)
