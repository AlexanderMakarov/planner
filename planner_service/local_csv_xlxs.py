import pandas as pd
import os
import sys
import datetime
import logging
import typing


DUMP_FILEPATH = "dump.csv"
RESULT_FILEPATH = "result.csv"
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
        elif filename.lower().endswith(".xls"):
            dumps[filename] = pd.read_excel(os.path.join(folder_path, filename), header=0)  # 1nd row
            logging.debug(f'Parsed Excel "{filename}" in {datetime.datetime.now() - start_time}.')
        elif filename.lower().endswith(".csv"):
            dumps[filename] = pd.read_csv(os.path.join(folder_path, filename), header=0)  # 1st row
            logging.debug(f'Parsed CSV "{filename}" in {datetime.datetime.now() - start_time}.')
    logging.info(f"Parsed {len(dumps)} files in {datetime.datetime.now() - total_start_time}.")
    return dumps


def find_date_column(columns: typing.List[str]) -> str:
    WORDS = ['date', 'дата', 'time', 'день', 'когда']
    # 1 step - find all columns possible.
    columns_with_word = dict()
    for col in columns:
        index = next((i for (i, x) in enumerate(WORDS) if x in col.lower()), None)
        if index >= 0:
            columns_with_word[col] = index
    logging.debug(f"To find 'date' column filtered {columns} to {columns_with_word}.")
    if len(columns_with_word) == 1:
        return list(columns_with_word.keys())[0]
    elif len(columns_with_word) == 0:
        return None
    # 2 step - set weight by index and number of characters.
    for k, v in columns_with_word.items():
        weight = (len(WORDS) - v) * 1000 + (999 - len(k))
        logging.debug(f"Set {weight} weight for '{k}'.")
        columns_with_word[k] = weight
    # 3 step - sort by weight and take first.
    return sorted(columns_with_word, key=columns_with_word.get, reverse=True)[0]


def extract_columns_to_timeseries_predict(cols: typing.List[str], cols_to_predict: typing.List[str]) ->\
                                          typing.Tuple[typing.Set[str], str]:
    result_columns  = set()
    # Take columns to predict and fill up remained columns to find "date" column from.
    contenders = []
    for col in cols:
        if col.lower() in cols_to_predict:
            result_columns.add(col)
        else:
            contenders.append(col)
    assert result_columns,\
           f"Can't find columns to predict: expected at least one from {cols_to_predict} to be in {cols}."
    # Find 'date' column.
    date_column = find_date_column(contenders)
    assert date_column, f"Can't find 'date' column in {contenders}."
    result_columns.add(date_column)
    return result_columns, date_column


def read_ets_dumps_and_merge(folder_path: str) -> pd.DataFrame:
    dumps = read_ets_dumps(folder_path)
    data = pd.DataFrame()
    for k, v in dumps.items():
        result_columns, date_column = extract_columns_to_timeseries_predict(
            list(v.columns), ['project-task', 'effort', 'description']
        )
        # Drop empty rows and not interesting columns.
        new_data: pd.DataFrame = v.dropna().drop(set(v.columns) - result_columns, axis=1)
        # Rename date column to 'ds'.
        new_data.rename(columns={date_column: 'ds'}, inplace=True)
        data = data.append(new_data, ignore_index=True)
        logging.debug(f"Merged {len(new_data)} rows from '{k}' file into common dataframe with {len(data)} rows now."
                      f" Columns {list(new_data.columns)} -> {list(data.columns)}.")
    data.sort_values(by='ds', inplace=True)
    logging.info(f"Parsed {list(data.columns)} columns and {len(data)} rows from '{folder_path}' files'.")
    logging.debug(data.describe)
    return data


def get_project_cn(data: pd.DataFrame) -> str:
    return [x for x in list(data.columns) if x.lower() == 'project-task']


def get_effort_cn(data: pd.DataFrame) -> str:
    return [x for x in list(data.columns) if x.lower() == 'effort']


def extract_units(data: pd.DataFrame) -> typing.List[str]:
    return data[get_project_cn(data)].unique().to_list()


def run_prophet(data: pd.DataFrame, date: str) -> float:
    pass


def predict_unit(df: pd.DataFrame, date: str, threshold: float) -> typing.Optional[pd.Index]:
    effort_cn = get_effort_cn(df)
    data = df.loc[:, ['ds']]
    data['y'] = 1.0
    if run_prophet(data, date) < threshold:
        return None
    data = df.loc[:, ['ds', effort_cn]]
    data.rename(axis=1, columns={effort_cn: 'y'})
    return run_prophet(data, date)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        #folder_path = "/home/i4ellendger/code/planner/data/old reports/2020/min"
        data = read_ets_dumps_and_merge(folder_path)
        data.to_csv(DUMP_FILEPATH, index=False)
    else:
        data = pd.read_csv(DUMP_FILEPATH, header=0)
    print(data.head)
    # Idea to predict ETS is follow:
    # 1) Divide rows on 'predict units' by project and description (by complexity stages)
    #       a) only project
    #       b) project + description
    #       c) project + common part of description (like "abc" and "abd" are same, but "cab" and "abc" are different)
    # 2) Predict probability of each 'predict unit' on new day. Remove units by some threshold.
    # 3) Predict effort of selected units on this day.
    dates_to_predict = ['2021-01-06', '2021-01-08', '2021-01-11', '2021-01-12', '2021-01-13']
    # a option
    units = extract_units(data)
    prediction = pd.DataFrame()
    project_cn = get_effort_cn(data)
    effort_cn = get_effort_cn(data)
    unit_histories = dict(((unit, data.loc[data[project_cn] == unit]) for unit in units))
    for date in dates_to_predict:
        day_prediction = []
        for unit, unit_data in unit_histories.items():
            y = predict_unit(unit_data, date, 0.8)
            if y is not None:
                day_prediction.append({
                    project_cn: unit,
                    effort_cn: y,
                    'Date': date,
                })
        logging.debug(f"{date}: predicted {len(day_prediction)} rows")
        prediction.append(day_prediction)
    logging.info(prediction.describe)
    prediction.to_csv(RESULT_FILEPATH, header=True)
