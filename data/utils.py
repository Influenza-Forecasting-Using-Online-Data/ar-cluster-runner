import os.path
from datetime import datetime

import pandas as pd


def get_datetime_df(index_col_name='year week', include_search_terms=True, search_query_threshold=50):
    search_query_thresholds = {50, 100, 150, 200, 250, 300, 350, 400}
    if search_query_threshold not in search_query_thresholds:
        raise Exception(
            "search_query_threshold can be one of 50, 100, 150, 200, 250, 300, 350, 400")

    parent_path = my_path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(parent_path, 'year_and_week_data_frame.csv'))
    df.drop(df.columns[0], axis=1, inplace=True)
    df = df[['year', 'week', 'Disease Rate']]

    if include_search_terms is True:
        search_query_df = pd.read_csv(os.path.join(parent_path, str(search_query_threshold) + '_zero_terms.csv'))
        search_query_df.drop(labels=['IsoWeeks', 'Disease Rate'], axis=1, inplace=True)
        df = pd.concat([df, search_query_df], axis=1)

    datetime_df = pd.DataFrame(
        {index_col_name: pd.to_datetime(df["year"].astype(str) + " " + df["week"].astype(str) + " 1",
                                        format="%G %V %w")})
    datetime_df = pd.concat([datetime_df, df], axis=1)
    # if include_search_terms is False:
    #     datetime_df = datetime_df[[index_col_name, 'year', 'week', 'Disease Rate']]
    return datetime_df


def get_week_range_df(index_col_name='week range', include_search_terms=True, search_query_threshold=50):
    df = get_datetime_df(index_col_name, include_search_terms=include_search_terms, search_query_threshold=search_query_threshold)
    df.set_index(index_col_name, inplace=True)
    df.index = pd.DatetimeIndex(df.index, closed='left').to_period('W')
    return df


def transform_index_to_week_range(df, index_col_name='week range'):
    df.drop(df.columns[0], axis=1, inplace=True)
    datetime_df = pd.DataFrame(
        {index_col_name: pd.to_datetime(df["year"].astype(str) + " " + df["week"].astype(str) + " 1",
                                        format="%G %V %w")})
    datetime_df = pd.concat([datetime_df, df], axis=1)
    datetime_df.set_index(index_col_name, inplace=True)
    datetime_df.index = pd.DatetimeIndex(datetime_df.index, closed='left').to_period('W')
    return datetime_df


def to_week_range(year_num, week_num):
    # %G %V %w format to parse iso dates available in python >3.6
    # https://stackoverflow.com/questions/35128266/strptime-seems-to-create-wrong-date-from-week-number
    return datetime.strptime(str(year_num) + " " + str(week_num) + " 1", "%G %V %w")
