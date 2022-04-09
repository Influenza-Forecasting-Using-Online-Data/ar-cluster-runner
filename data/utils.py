import os.path
from datetime import datetime

import pandas as pd


def get_datetime_df(index_col_name='year week', include_search_terms=True, search_query_threshold=50, exog_scaler=None):
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
        if exog_scaler is not None:
            scaled_search_queries = exog_scaler.fit_transform(search_query_df.values)
            search_query_df = pd.DataFrame(scaled_search_queries, index=search_query_df.index,
                                           columns=search_query_df.columns)
        df = pd.concat([df, search_query_df], axis=1)

    datetime_df = pd.DataFrame(
        {index_col_name: pd.to_datetime(df["year"].astype(str) + " " + df["week"].astype(str) + " 1",
                                        format="%G %V %w")})
    datetime_df = pd.concat([datetime_df, df], axis=1)
    # if include_search_terms is False:
    #     datetime_df = datetime_df[[index_col_name, 'year', 'week', 'Disease Rate']]
    return datetime_df


def get_week_range_df(index_col_name='week range', include_search_terms=True, search_query_threshold=50,
                      exog_scaler=None, outseason_start_week=None, outseason_end_week=None):
    """

    Parameters
    ----------
    index_col_name: Name to give to index column.
    include_search_terms: If True, will include query search terms specified using the search_query_threshold parameter. If set to False, disregards search_query_threshold value.
    search_query_threshold: Number of search query terms to include ordered descending by the numbre of 0-values they contain. Must be in {50, 100, 150, 200, 250, 300, 350, 400}.
    exog_scaler: Scikit learn scaler object (fit_transform interface) used to scale data. Ignored if include_search_terms set to False.
    outseason_start_week: Week marking the start of the out-of-season period (end of influenza season). This week is excluded from the returned dataframe. Must be in {1, ..., 52}.
    outseason_end_week:  Week marking the end of the out-of-season period (start of influenza season). This week is included in the returned dataframe. Must be in {1, ..., 52}.

    Returns
    -------
    DataFrame object.
    """
    df = get_datetime_df(index_col_name, include_search_terms=include_search_terms,
                         search_query_threshold=search_query_threshold, exog_scaler=exog_scaler)
    df.set_index(index_col_name, inplace=True)
    df.index = pd.DatetimeIndex(df.index, closed='left').to_period('W')

    if outseason_start_week is not None and outseason_end_week is not None:
        if (outseason_start_week <= 0 and outseason_start_week > 52) or (
                outseason_end_week <= 0 and outseason_end_week > 52):
            raise Exception("outseason_start_week and outseason_end_week must be None or in interval [1, 52].")
        # exclude weeks out of season
        df = df.loc[~df['week'].isin([i for i in range(outseason_start_week, outseason_end_week)])]
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


def get_train_and_test_intervals(num_years_train_interval, outseason_start_week, outseason_end_week):
    if num_years_train_interval == 2:
        return ([
                    (to_week_range(2004, outseason_end_week), to_week_range(2006, outseason_start_week)),
                    (to_week_range(2005, outseason_end_week), to_week_range(2007, outseason_start_week)),
                    (to_week_range(2006, outseason_end_week), to_week_range(2008, outseason_start_week)),
                    (to_week_range(2007, outseason_end_week), to_week_range(2009, outseason_start_week)),
                    (to_week_range(2008, outseason_end_week), to_week_range(2010, outseason_start_week)),
                    (to_week_range(2009, outseason_end_week), to_week_range(2011, outseason_start_week)),
                    (to_week_range(2010, outseason_end_week), to_week_range(2012, outseason_start_week)),
                    (to_week_range(2011, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2012, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2013, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2014, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2015, outseason_end_week), to_week_range(2017, outseason_start_week)),
                ],
                [
                    (to_week_range(2006, outseason_end_week), to_week_range(2007, outseason_start_week)),
                    (to_week_range(2007, outseason_end_week), to_week_range(2008, outseason_start_week)),
                    (to_week_range(2008, outseason_end_week), to_week_range(2009, outseason_start_week)),
                    (to_week_range(2009, outseason_end_week), to_week_range(2010, outseason_start_week)),
                    (to_week_range(2010, outseason_end_week), to_week_range(2011, outseason_start_week)),
                    (to_week_range(2011, outseason_end_week), to_week_range(2012, outseason_start_week)),
                    (to_week_range(2012, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2013, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2014, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2015, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2016, outseason_end_week), to_week_range(2017, outseason_start_week)),
                    (to_week_range(2017, outseason_end_week), to_week_range(2018, outseason_start_week)),
                ])
    elif num_years_train_interval == 4:
        return ([
                    (to_week_range(2004, outseason_end_week), to_week_range(2008, outseason_start_week)),
                    (to_week_range(2005, outseason_end_week), to_week_range(2009, outseason_start_week)),
                    (to_week_range(2006, outseason_end_week), to_week_range(2010, outseason_start_week)),
                    (to_week_range(2007, outseason_end_week), to_week_range(2011, outseason_start_week)),
                    (to_week_range(2008, outseason_end_week), to_week_range(2012, outseason_start_week)),
                    (to_week_range(2009, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2010, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2011, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2012, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2013, outseason_end_week), to_week_range(2017, outseason_start_week)),
                ],
                [
                    (to_week_range(2008, outseason_end_week), to_week_range(2009, outseason_start_week)),
                    (to_week_range(2009, outseason_end_week), to_week_range(2010, outseason_start_week)),
                    (to_week_range(2010, outseason_end_week), to_week_range(2011, outseason_start_week)),
                    (to_week_range(2011, outseason_end_week), to_week_range(2012, outseason_start_week)),
                    (to_week_range(2012, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2013, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2014, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2015, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2016, outseason_end_week), to_week_range(2017, outseason_start_week)),
                    (to_week_range(2017, outseason_end_week), to_week_range(2018, outseason_start_week)),
                ])
    elif num_years_train_interval == 5:
        return ([
                    (to_week_range(2004, outseason_end_week), to_week_range(2009, outseason_start_week)),
                    (to_week_range(2005, outseason_end_week), to_week_range(2010, outseason_start_week)),
                    (to_week_range(2006, outseason_end_week), to_week_range(2011, outseason_start_week)),
                    (to_week_range(2007, outseason_end_week), to_week_range(2012, outseason_start_week)),
                    (to_week_range(2008, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2009, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2010, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2011, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2012, outseason_end_week), to_week_range(2017, outseason_start_week)),
                ],
                [
                    (to_week_range(2009, outseason_end_week), to_week_range(2010, outseason_start_week)),
                    (to_week_range(2010, outseason_end_week), to_week_range(2011, outseason_start_week)),
                    (to_week_range(2011, outseason_end_week), to_week_range(2012, outseason_start_week)),
                    (to_week_range(2012, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2013, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2014, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2015, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2016, outseason_end_week), to_week_range(2017, outseason_start_week)),
                    (to_week_range(2017, outseason_end_week), to_week_range(2018, outseason_start_week)),
                ])
    elif num_years_train_interval == 10:
        return ([
                    (to_week_range(2004, outseason_end_week), to_week_range(2013, outseason_start_week)),
                    (to_week_range(2005, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2006, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2007, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2008, outseason_end_week), to_week_range(2017, outseason_start_week)),
                ],
                [
                    (to_week_range(2013, outseason_end_week), to_week_range(2014, outseason_start_week)),
                    (to_week_range(2014, outseason_end_week), to_week_range(2015, outseason_start_week)),
                    (to_week_range(2015, outseason_end_week), to_week_range(2016, outseason_start_week)),
                    (to_week_range(2016, outseason_end_week), to_week_range(2017, outseason_start_week)),
                    (to_week_range(2017, outseason_end_week), to_week_range(2018, outseason_start_week)),
                ])
    else:
        raise Exception("num_years_train_interval must be in {2, 4, 5, 10}.")
