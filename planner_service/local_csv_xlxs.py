import pandas as pd
import os
import sys
import datetime
import logging
import typing
import difflib
import prophet



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


def extract_columns_to_timeseries_predict(cols: typing.List[str], cols_to_predict: typing.List[str])\
                                          -> typing.List[str]:
    """
    Finds out columns from dataframe. Returns them in order to rename into supported names.
    :param cols: list of datafram columns in order to work with.
    :param cols_to_predict: list of supported 'not date' lowercases columns in order.
    :returns: - list of supportd column names with last 'date' column.
    """
    result  = []
    # Take columns to predict and fill up remained columns to find "date" column from.
    contenders = []
    for i, col in enumerate(cols):
        if col.lower() in cols_to_predict:
            result.append(col)
            continue
        else:
            contenders.append(col)
    assert result,\
           f"Can't find columns to predict: expected at least one from {cols_to_predict} to be in {cols}."
    # Find 'date' column.
    date_column = find_date_column(contenders)
    assert date_column, f"Can't find 'date' column in {contenders}."
    result.append(date_column)
    return result


def read_ets_dumps_and_merge(folder_path: str) -> pd.DataFrame:
    dumps = read_ets_dumps(folder_path)
    data = pd.DataFrame()
    supported_columns = ['project-task', 'effort', 'description']
    for k, v in dumps.items():
        result_columns = extract_columns_to_timeseries_predict(
            list(v.columns), supported_columns
        )
        # Drop empty rows and not interesting columns.
        new_data: pd.DataFrame = v.dropna().drop(set(v.columns) - set(result_columns), axis=1)
        # Convert date column and rename it 'ds'.
        new_data['ds'] = pd.DatetimeIndex(new_data[result_columns[-1]])
        new_data.drop(columns=[result_columns[-1]], axis=1, inplace=True)
        # Rename columns into supported names.
        new_data.rename(
            columns={
                result_columns[0]: 'task',
                result_columns[1]: 'effort',
                result_columns[2]: 'description',
            },
            inplace=True
        )
        data = data.append(new_data, ignore_index=True)
        logging.debug(f"Merged {len(new_data)} rows from '{k}' file into common dataframe with {len(data)} rows now."
                      f" Columns {list(new_data.columns)} -> {list(data.columns)}.")
    data.sort_values(by='ds', inplace=True)
    logging.info(f"Parsed {list(data.columns)} columns and {len(data)} rows from '{folder_path}' files'.")
    logging.debug(data.describe)
    return data

class Tokenizer:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def get_unit_histories(self) -> typing.Dict[typing.Any, pd.DataFrame]:
        raise NotImplemented()

    def expand_unit_prediction(self, unit: typing.Any, y: float) -> dict:
        raise NotImplemented()

    def unit_to_str(self, unit: typing.Any) -> str:
        return str(unit)

    def dump_unit_histories(self, unit_histories: typing.Dict[typing.Any, pd.DataFrame]):
        for unit, df in unit_histories.items():
            df.to_csv(self.unit_to_str(unit) + ".csv", header=True)

    def _separate_by_task(self):
        tasks = self.df['task'].unique()
        return dict((x, self.df[self.df['task'] == x]) for x in tasks)


class SameDescriptionTokenizer(Tokenizer):
    """
    Simple tokens matcher with "same task and description".
    2020 Jan+Dec AM dataset - only 5 tokens with > 4 items.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def get_unit_histories(self) -> typing.Dict[str, pd.DataFrame]:
        result = dict()
        for task, df in self._separate_by_task().items():
            for desc in df['description'].unique():
                result[(task, desc)] = df[df['description'] == desc]
        return result

    def expand_unit_prediction(self, unit, y: float) -> dict:
        return {
            'task': unit[0],
            'effort': y,
            'description': unit[1]
        }

class SimilarDescriptionTokenizer(Tokenizer):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def find_similar_descriptions(df: pd.DataFrame) -> typing.List[str]:
        return None

    def get_unit_histories(self) -> typing.Dict[str, pd.DataFrame]:
        result = dict()
        for task, df in self._separate_by_task():
            # TODO
            # https://github.com/seatgeek/thefuzz
            # https://docs.python.org/3/library/difflib.html
            # df[df['var1'].str.contains('A|B')] - filter by "contains A or B"
            for desc in self.find_similar_descriptions(df):
                result[(task, desc), df[df['description'].str.startswith(desc)]]
        return result


def timeit(name:str, func, *args) -> typing.Any:
    start_time = datetime.datetime.now()
    result = func(*args)
    logging.debug(f"{name} took {datetime.datetime.now() - start_time}.")


def run_prophet(df: pd.DataFrame, date: str) -> float:
    m = prophet.Prophet().fit(df)
    pass  # See jupiter "ets"


def predict_unit(data: pd.DataFrame, date: str, y_col:str, threshold: float) -> typing.Optional[pd.Index]:
    # Don't predict for less than 2 data points.
    if len(data) < 2:
        return None
    # Predict probability of unit for this day and compare with threshold.
    df = data.loc[:, ['ds']]
    df['y'] = 1.0
    unit_probability = run_prophet(df, date)
    if unit_probability < threshold:
        return None
    # Predict real value for this day.
    df = data.loc[:, ['ds', y_col]]
    df.rename(axis=1, columns={y_col: 'y'})
    return run_prophet(df, date)


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
    #       a) only project => not enough, it predicts only amount of effort per day.
    #       b) only project and filter out old data ("only 15 events if recent" kinda rules) => optimization
    #       c) project + description + filtering
    #       d) project + common part of description (like "abc" and "abd" are same, but "cab" and "abc" are different)
    # 2) Predict probability of each 'predict unit' on new day. Remove units by some threshold.
    # 3) Predict effort of selected units on this day.
    dates_to_predict = ['2021-01-06', '2021-01-08', '2021-01-11', '2021-01-12', '2021-01-13']
    threshold_to_take_unit = 0.8
    tokenizer = SameDescriptionTokenizer(data)
    prediction = pd.DataFrame()
    unit_histories = timeit(f"Tokenize {len(data)} rows",  tokenizer.get_unit_histories)
    logging.info(f"From {len(data)} rows got {len(unit_histories)} units")
    for date in dates_to_predict:
        day_prediction = []
        for unit, unit_data in unit_histories.items():
            y = timeit(f"Predict {tokenizer.unit_to_str(unit)}", predict_unit,
                       unit_data, date, 'effort', threshold_to_take_unit)
            if y is not None:
                row = tokenizer.expand_unit_prediction(unit, y)
                row['date'] = date
                day_prediction.append(row)
        logging.debug(f"{date}: predicted {len(day_prediction)} rows")
        prediction.append(day_prediction)
    logging.info(prediction.describe)
    prediction.to_csv(RESULT_FILEPATH, header=True)
