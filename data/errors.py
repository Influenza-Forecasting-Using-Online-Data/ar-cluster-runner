import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd


class Errors:
    # upload weekly data and insert the column names for y_actual and y_predicted
    def __init__(self, data, actual_col_name, predicted_col_name):
        # TODO: add booleans to choose what errors you want calculated
        self.data_weekly = data
        self.actual_name = actual_col_name
        self.predicted_name = predicted_col_name
        self.year_df = None

    def _get_mae(self, actual, predicted):
        self.mae = mean_absolute_error(actual, predicted)
        return self.mae

    def _get_mape(self, actual, predicted):
        self.mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return self.mape

    def _get_mse(self, actual, predicted):
        self.mse = mean_squared_error(actual, predicted)
        return self.mse

    # TODO: add formulas
    def _get_negative_likelihood(self, actual, predicted):
        self.nll_like = []
        return self.nll_like

    def _get_rmse(self, actual, predicted):
        self.rmse = np.sqrt(self._get_mse(actual, predicted))
        return self.rmse

    def get_errors(self, show_all=True, error_types=None):
        # df cols:  week range | year | week | MAE | MAPE | MSE | NEG_LIKELIHOOD | RMSE
        if error_types is None:
            error_types = {}

        errors_df = self._get_yearly_index_df()

        # get stats per year
        (MAEs, MAPEs, MSEs, NLLs, RMSEs) = self._get_stats_per_year()

        error_types = set(map(lambda x: x.upper(), error_types))  # convert set entries to uppercase
        # append (temp solution: append only requested ones)
        if 'MAE' in error_types or show_all:
            errors_df['MAE'] = MAEs
        if 'MAPE' in error_types or show_all:
            errors_df['MAPE'] = MAPEs
        if 'MSE' in error_types or show_all:
            errors_df['MSE'] = MSEs
        if 'NLL' in error_types or show_all:
            errors_df['NLL'] = NLLs
        if 'RMSE' in error_types or show_all:
            errors_df['RMSE'] = RMSEs

        return errors_df

    ### HELPER FUNCTIONS

    def _extract_cols(self):
        # get columns of interest from data, to calculate values based on them
        self.actual = self.data_weekly[self.actual_name]
        self.predicted = self.data_weekly[self.predicted_name]

    def _get_yearly_index_df(self):
        yearly_df = self.data_weekly.iloc[:, 0:2]

        self.years = yearly_df['year'].unique()
        weeks = []
        for year in self.years:
            weeks.append(len(yearly_df[yearly_df['year'] == year]))

        temp_dict = {'years': self.years,
                     'no. of weeks': weeks}
        base_df = pd.DataFrame(temp_dict, index=None)

        return base_df

    def _get_stats_per_year(self):
        MAEs = []
        MAPEs = []
        MSEs = []
        NLLs = []
        RMSEs = []
        for year in self.years:
            self.year_df = self.data_weekly[self.data_weekly['year'] == year]
            actual = self.year_df[self.actual_name]
            predicted = self.year_df[self.predicted_name]

            mae_year = self._get_mae(actual, predicted)
            mape_year = self._get_mape(actual, predicted)
            mse_year = self._get_mse(actual, predicted)
            nll_year = self._get_negative_likelihood(actual, predicted)
            rmse_year = self._get_rmse(actual, predicted)

            MAEs.append(mae_year)
            MAPEs.append(mape_year)
            MSEs.append(mse_year)
            NLLs.append(nll_year)
            RMSEs.append(rmse_year)

        return (MAEs, MAPEs, MSEs, NLLs, RMSEs)


def get_models_error(data, error_type, actual_col_name, predicted_col_names):
    error_types = {"MAE", "MAPE", "MSE", "RMSE"}
    error_type = error_type.upper()
    if error_type not in error_types:
        raise Exception("error_type should be one of %s" % str(error_types))
    if len(predicted_col_names) == 0:
        raise Exception("predicted_col_names should a list of size > 0")

    error = Errors(data, actual_col_name, predicted_col_names[0])
    error_df = error.get_errors(show_all=False, error_types={error_type})  # grab year and no. of weeks columns
    error_df = error_df.loc[:, error_df.columns != error_type]  # exclude error column

    for i in range(0, len(predicted_col_names)):
        predicted_col_name = predicted_col_names[i]
        error = Errors(data, actual_col_name, predicted_col_name)
        error_column = error.get_errors(show_all=False, error_types={error_type})[error_type]
        error_df[predicted_col_name] = error_column

    error_df.style.set_caption(error_type)
    return error_df


def get_best_performing_models(error_df):
    # TODO: add error_type as param
    best_columns = []
    best_values = []
    copy_df = error_df.copy(deep=True)
    copy_df.drop('no. of weeks', axis=1, inplace=True)
    for index, row in copy_df.iterrows():
        best_column = row.idxmin()
        best_value = row.min()

        best_columns.append(best_column)
        best_values.append(best_value)

    return pd.DataFrame({'years': error_df['years'], 'best model': best_columns, 'error': best_values})
