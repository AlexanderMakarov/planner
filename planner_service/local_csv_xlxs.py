from numpy import datetime64
from numpy.lib.histograms import histogram
import pandas as pd
import os
import sys
import datetime
import logging
import typing
import enum
import difflib
from pandas.core.base import DataError
import prophet



DUMP_FILEPATH = "dump.csv"
RESULT_FILEPATH = "result.csv"
logging.basicConfig(level=logging.DEBUG)


# https://stackoverflow.com/a/56695622/1535127 - they only way shut up PyStan.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


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


class Period(enum.Enum):
    DAILY = 1
    WEEKLY = 7
    MONTHLY = 30
    YEARLY = 365


class Tokenizer:

    def __init__(self, df:pd.DataFrame):
        self.df = df

    def get_unit_histories(self, period: Period) -> typing.Dict[typing.Any, pd.DataFrame]:
        """
        Does main logic of class - finds all relevant units (rows to predict) in dataframe and separates dataframe
        per each unit.
        Unit here may be any hashable type (because may span few columns) and it is tokenizer-specific type.
        :param period: Period for which need to find relevant units. For example it is useless to provide few years
        history for "next day" prediciton.
        :return: Dictionary with dataframe (histories) per found unit.
        """
        raise NotImplemented()

    def expand_unit_prediction(self, unit: typing.Any, y: float) -> dict:
        raise NotImplemented()

    def unit_to_str(self, unit: typing.Any) -> str:
        return str(unit)

    def get_not_empty_history_points(self) -> typing.List[datetime64]:
        return self.df['ds'].unique()

    def get_max_effort(self, unit_histories: typing.Dict[typing.Any, pd.DataFrame]) -> float:
        return max((x['effort'].max() for x in unit_histories.values()))

    def dump_unit_histories(self, unit_histories: typing.Dict[typing.Any, pd.DataFrame]):
        for unit, df in unit_histories.items():
            df.to_csv(self.unit_to_str(unit) + ".csv", header=True)

    def _limit_history_by_period(self, df: pd.DataFrame, last_day_in_data: datetime64, period: Period) -> pd.DataFrame:
        # Measure relative to last day in the whole history data, not to "day to predict"!
        # 1. Last occurence shouldn't be older than 30 periods.
        # 2. Max history is 60 periods.
        # 3. At least one row in result.
        # 4. If one row it should be in last day.
        if df is None or df.empty:
            return None
        last_day = df['ds'].iloc[-1]
        if (last_day_in_data - last_day).days > 30 * period.value:
            return None
        first_day = last_day_in_data - pd.to_timedelta(60 * period.value, unit="d")
        df = df.drop(df[df['ds'] < first_day].index)
        if df.empty:
            return None
        if len(df) == 1 and df.iloc[0]['ds'] != last_day_in_data:
            return None
        return df

    def _separate_by_task(self):
        tasks = self.df['task'].unique()
        return dict((x, self.df[self.df['task'] == x]) for x in tasks)


class SameDescriptionTokenizer(Tokenizer):
    """
    Simple tokens matcher with "same task and description".
    Removes tokens not appearing long before.
    2020 Jan+Dec AM dataset - only 5 tokens with > 4 items.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def get_unit_histories(self, period: Period) -> typing.Dict[str, pd.DataFrame]:
        result = dict()
        last_day_in_data = self.df['ds'].iloc[-1]
        for task, df in self._separate_by_task().items():
            for desc in df['description'].unique():
                history = df[df['description'] == desc]
                history = self._limit_history_by_period(history, last_day_in_data, period)
                if history is not None:
                    result[(task, desc)] = history
        return result

    def expand_unit_prediction(self, unit: typing.Any, y: float) -> dict:
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
    return result


def run_prophet(df: pd.DataFrame, date: str) -> float:
    # FYI: changepoint_prior_scale 0.05->0.2 doesn't affect speed.
    model = prophet.Prophet().fit(df)
    future = pd.DataFrame((pd.to_datetime(date),), columns=['ds'])
    forecast = model.predict(future)
    return forecast.at[0, 'yhat']  # 0 index because only one row was asked.


def predict_unit(data: pd.DataFrame, date: str, not_empty_history_days: typing.List[datetime64],
                 y_col:str) -> typing.Optional[pd.Index]:
    # Add '0' to all days because we expect that if day exists in dataset then it contains all tokens.
    # If don't add those '0' then model will approximate graph into line between 2 not adjusted points.
    df = data.loc[:, ['ds', y_col]].rename(columns={y_col: 'y'})
    days_to_fill = set(not_empty_history_days) - set(data['ds'].to_list())
    df = df.append([{'ds': x, 'y': 0} for x in days_to_fill])
    with suppress_stdout_stderr():  # PyStan generates a lot of logs from cpp - so no control on it.
        return run_prophet(df, date)


def predict_day(unit_histories: typing.Dict[typing.Any, pd.DataFrame], date: str,
                not_empty_history_days: typing.List[datetime64]) -> typing.Dict[typing.Any, float]:
    day_prediction = {}
    for unit, unit_data in unit_histories.items():
        day_prediction[unit] = timeit(f"Predict {tokenizer.unit_to_str(unit)}", predict_unit,
                                      unit_data, date, not_empty_history_days, 'effort')
    return day_prediction


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        #folder_path = "/home/i4ellendger/code/planner/data/old reports/2020/min"
        data = read_ets_dumps_and_merge(folder_path)
        data.to_csv(DUMP_FILEPATH, index=False)
    else:
        data = pd.read_csv(DUMP_FILEPATH, header=0)
        data['ds'] = pd.DatetimeIndex(data['ds'])
    print(data.head)
    # Idea to predict ETS is follow:
    # 1) Divide rows on 'predict units' by project and description (by complexity stages)
    #       a) only project => not enough, it predicts only amount of effort per day.
    #       b) only project and filter out old data ("only 15 events if recent" kinda rules) => optimization
    #       c) project + description + filtering
    #       d) project + common part of description (like "abc" and "abd" are same, but "cab" and "abc" are different)
    # 2) Predict probability of each 'predict unit' on new day. Remove units by some threshold.
    # 3) Predict effort of selected units on this day.
    dates_to_predict = ['2021-01-06']  #TODO ['2021-01-06', '2021-01-08', '2021-01-11', '2021-01-12', '2021-01-13']
    period = Period.DAILY
    threshold_not_less = 0.25
    max_effort = 2.0
    tokenizer = SameDescriptionTokenizer(data)
    result = []
    unit_histories = timeit(f"Tokenize {len(data)} rows",  tokenizer.get_unit_histories, period)
    max_effort = tokenizer.get_max_effort(unit_histories)
    not_empty_history_days = tokenizer.get_not_empty_history_points()
    logging.info(f"From {len(data)} rows got {len(unit_histories)} units")
    for date in dates_to_predict:
        day_prediction = timeit(f"Predict {date}", predict_day, unit_histories, date, not_empty_history_days)
        logging.debug(f"{date}: predicted {len(day_prediction)} rows")
        for unit, y in day_prediction.items():
            if y is not None and y >= threshold_not_less:
                y_quantized = int(y * 4) * 0.25  # Quantize to 0.25.
                row = tokenizer.expand_unit_prediction(unit, min(y_quantized, max_effort))
                row['date'] = date
                result.append(row)
    logging.info("\n".join(str(x) for x in result))
    pd.DataFrame(result).to_csv(RESULT_FILEPATH, header=True)
