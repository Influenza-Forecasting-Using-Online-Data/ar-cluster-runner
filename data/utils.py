import os.path
from datetime import datetime

import pandas as pd


def get_datetime_df(index_col_name='year week', include_search_terms=True):
    parent_path = my_path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(parent_path, 'year_and_week_data_frame.csv'))
    df.drop(df.columns[0], axis=1, inplace=True)
    datetime_df = pd.DataFrame(
        {index_col_name: pd.to_datetime(df["year"].astype(str) + " " + df["week"].astype(str) + " 1",
                                        format="%G %V %w")})
    datetime_df = pd.concat([datetime_df, df], axis=1)
    if include_search_terms is False:
        datetime_df = datetime_df[[index_col_name, 'year', 'week', 'Disease Rate']]
    return datetime_df


def get_week_range_df(index_col_name='week range', include_search_terms=True):
    df = get_datetime_df(index_col_name, include_search_terms=include_search_terms)
    df.set_index(index_col_name, inplace=True)
    df.index = pd.DatetimeIndex(df.index, closed='left').to_period('W')
    return df


def to_week_range(year_num, week_num):
    # %G %V %w format to parse iso dates available in python >3.6
    # https://stackoverflow.com/questions/35128266/strptime-seems-to-create-wrong-date-from-week-number
    return datetime.strptime(str(year_num) + " " + str(week_num) + " 1", "%G %V %w")
