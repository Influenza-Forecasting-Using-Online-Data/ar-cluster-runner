import logging
import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.arima.model
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAXResults, SARIMAXResultsWrapper
from termcolor import colored

from data.errors import get_models_error

log = logging.getLogger("models.ar_model")

WARNING_TERM = colored('WARNING', 'red', attrs=['reverse'])
FINISHED_TERM = colored('FINISHED', 'green', attrs=['reverse'])


class ARModelSpecification:
    def __init__(self, order, seasonal_order=(0, 0, 0, 0), model_class=statsmodels.tsa.arima.model.ARIMA):
        self.model_class = model_class
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_name = None

    def init_model(self, endog, exog=None):
        self.model_name = get_ar_model_name(self.order, self.seasonal_order)
        return self.model_class(endog=endog, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
                                freq='W')  # TODO: check if freq correct

    def __eq__(self, other):
        if self.model_class == other.model_class and self.order == other.order and self.seasonal_order == other.seasonal_order:
            return True
        return False

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.model_class, self.order, self.seasonal_order))

    def __str__(self):
        model_class_str = ""
        if self.model_class == statsmodels.tsa.arima.model.ARIMA:
            model_class_str = "statsmodels.tsa.arima.model.ARIMA"
        elif self.model_class == statsmodels.tsa.statespace.sarimax.SARIMAX:
            model_class_str = "statsmodels.tsa.statespace.sarimax.SARIMAX"
        else:
            model_class_str = "unknown class"
        return model_class_str + " " + str(self.model_name)

    def __repr__(self):
        model_class_str = ""
        if self.model_class == statsmodels.tsa.arima.model.ARIMA:
            model_class_str = "statsmodels.tsa.arima.model.ARIMA"
        elif self.model_class == statsmodels.tsa.statespace.sarimax.SARIMAX:
            model_class_str = "statsmodels.tsa.statespace.sarimax.SARIMAX"
        else:
            model_class_str = "unknown class"
        return model_class_str + " " + str(self.model_name)


class ARModelWrapper:
    def __init__(self, ar_model_spec):
        self.ar_model_spec = ar_model_spec
        self.test_and_train_data = None
        self.seasonal_order = None
        self.y_test_prediction = None
        self.test_result = None
        self.y_train_prediction = None
        self.train_result = None
        self.test_interval = None
        self.validation_interval = None
        self.train_interval = None
        self.test_data = None
        self.validate_data = None
        self.model = None
        self.model_name = None
        self.data = None
        self.endog_col_name = None
        self.train_data = None
        self.order = None
        self.method = None
        self.maxiter = 1000

    def set_model_name(self, model_name):
        self.model_name = model_name

    def set_data(self, data, endog_col_name):
        self.data = data
        self.endog_col_name = endog_col_name

    def set_model_specification(self, ar_model_spec):
        self.ar_model_spec = ar_model_spec

    def split_dataset_by_intervals(self, train_interval, test_interval, validation_interval=None, print_data=False):
        assert self.data is not None
        assert self.endog_col_name is not None
        assert self.ar_model_spec is not None

        self.train_data = self.data[self.endog_col_name][train_interval[0]:train_interval[1]]
        if validation_interval is not None:
            self.validate_data = self.data[self.endog_col_name][validation_interval[0]:validation_interval[1]]
        self.test_data = self.data[self.endog_col_name][test_interval[0]:test_interval[1]]
        self.test_and_train_data = self.data[self.endog_col_name]

        self.train_interval = train_interval
        self.validation_interval = validation_interval
        self.test_interval = test_interval

        self.model = self.ar_model_spec.init_model(endog=self.train_data,
                                                   exog=None)  # TODO: implement exogenous variables

        if print_data is True:
            print('Training Data 1\n-------------\n' + str(self.train_data))
            print('\n')
            print('Validate Data 2\n----------\n' + str(self.validate_data))
            print('\n')
            print('Test Data 2\n----------\n' + str(self.test_data))

    def set_optimization(self, method, maxiter):
        self.method = method
        self.maxiter = maxiter

    def select_best_hyperparams(self, maxlag, ic='bic', glob=False, trend='n', seasonal=False, period=None):
        """
        Selects a model with bet performing hyperparamters for given validation
        interval on data. If model was previously set using set_model, then this
        method will overwrite it. Will also overwrite model_name to be the name
        of the model class.

        Returns: ar_select_order result, NOT model.
        """
        assert self.model is not None

        if self.validation_interval is None:
            raise Exception('No validation_interval specified, use split_dataset_by_intervals to add it')
        select_result = ar_select_order(self.validate_data, maxlag=maxlag, seasonal=seasonal,
                                        period=period, glob=glob, ic=ic, old_names=False)
        return select_result

    def _fit(self, method, maxiter=None, cov_type=None):
        if self.ar_model_spec.model_class == ARIMA and maxiter is not None:
            warnings.warn(
                'Cannot use maxiter={x}: maxiter paramter not supported by statsmodels.tsa.arima.model.ARIMA'.format(
                    x=maxiter))
            if method is not None:
                return self.model.fit(method=method, cov_type=cov_type)
            else:
                return self.model.fit(cov_type=cov_type)

        if maxiter is not None:
            self.maxiter = maxiter
        if method is not None:
            return self.model.fit(method=method, maxiter=self.maxiter, cov_type=cov_type)
        else:
            return self.model.fit(maxiter=self.maxiter, cov_type=cov_type)

    def train_model(self, method=None, maxiter=None, cov_type=None):
        assert self.model is not None
        assert self.train_data is not None
        assert self.test_data is not None

        if method is not None:
            self.method = method
        if maxiter is not None:
            self.maxiter = maxiter
        self.train_result = self._fit(method=self.method, maxiter=self.maxiter, cov_type=cov_type)
        # TODO: remove train prediction for memory usage improvement
        # self.y_train_prediction = self.train_result.predict(self.train_interval[0], self.train_interval[1])
        return self.train_result

    def test_model(self, steps=1, keep_test_result_obj=False):
        if steps < 1:
            raise Exception('steps must be greater than or equal to 1')
        assert self.model is not None
        assert self.train_data is not None
        assert self.test_data is not None
        assert self.train_result is not None

        test_result = None
        if steps == 1:
            test_result = self.train_result.apply(endog=self.test_and_train_data, refit=False)
            self.y_test_prediction = test_result.predict(start=self.test_interval[0], end=self.test_interval[1],
                                                         dynamic=False)
        else:
            steps_ahead_forecasts = self.test_data.copy(deep=True).iloc[0:steps - 1]
            for i in range(0, len(self.test_data) - steps + 1):
                test_result = self.train_result.append(endog=[self.test_data.iloc[i]], refit=False)
                forecast_point_steps_ahead = pd.Series(test_result.forecast(steps=steps)[steps - 1],
                                                       index=[self.test_data.index[i + steps - 1]])
                # print(forecast_point_steps_ahead)
                steps_ahead_forecasts = pd.concat([steps_ahead_forecasts, forecast_point_steps_ahead], axis=0,
                                                  ignore_index=False)
            self.y_test_prediction = steps_ahead_forecasts
        if keep_test_result_obj is True:
            self.test_result = test_result

        if self.model is not None:
            self.model.endog = self.train_data
        # TODO does not work with seasonal
        return test_result

    def get_train_prediction_df(self, model_name=None, steps=1):
        assert self.train_result is not None
        assert steps > 0

        if model_name is None:
            model_name = self.model_name

        train_prediction_df = self.train_data.copy(deep=True)
        train_prediction_df[model_name] = self.y_train_prediction
        return train_prediction_df

    def get_test_prediction_df(self, model_name=None, steps=1):
        assert self.test_result is not None
        assert steps > 0

        if model_name is None:
            model_name = self.model_name

        test_prediction_df = self.test_data.copy(deep=True)
        test_prediction_df[model_name] = self.y_test_prediction
        return test_prediction_df


def get_ar_model_name(order, s_order=None):
    sorder_str = '' if s_order is None else str(s_order)
    return str(order) + sorder_str


def plot_models(data, model_cols, x_format='month', figsize=(30, 10), ticks_fontsize=16,
                legend_fontsize=20, title='', title_fontsize=20, label_fontsize=22,
                xlim=None, ground_truth_col='Disease Rate', include_ground_truth=True, save=False, show=True,
                folder_path=''):
    if isinstance(model_cols, list) is False:
        raise Exception("models_cols object must be passed as list")

    x_axis_locator = None
    x_axis_formatter = None
    if x_format == 'month':
        x_axis_locator = mdates.MonthLocator(bymonthday=1)
        x_axis_formatter = mdates.DateFormatter('%Y %b')
    elif x_format == 'week':
        x_axis_locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        x_axis_formatter = mdates.ConciseDateFormatter(x_axis_locator)
    elif x_format == 'year':
        x_axis_locator = mdates.YearLocator(interval=1)
        x_axis_formatter = mdates.DateFormatter('%Y')
        # x_axis_formatter = mdates.ConciseDateFormatter(x_axis_locator)
    else:
        raise Exception("x_format must be 'month' or 'week'")

    fig, ax = plt.subplots(figsize=figsize)
    for model_col in model_cols:
        data.plot(use_index=True, y=str(model_col), x_compat=True, ax=ax)
    if include_ground_truth:
        data.plot(use_index=True, y=ground_truth_col, x_compat=True, ax=ax)

    ax.xaxis.set_major_locator(x_axis_locator)
    ax.xaxis.set_major_formatter(x_axis_formatter)

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    # ax.set_xlabel(str(data_interval[0]) + ' - ' + str(data_interval[1]), fontsize=label_fontsize)
    ax.set_ylabel('ILI Rate', fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)

    if xlim is not None and len(xlim) == 2:
        plt.xlim(xlim[0], xlim[1])

    if save:
        file_name = os.path.join(folder_path, 'model_plot.png')
        if title is not None and title != '':
            file_name = os.path.join(folder_path, '%s.png' % str(title))
        plt.savefig(file_name, bbox_inches='tight')
        plt.clf()
        fig.clf()
        plt.close('all')
    if show:
        plt.show()


class ARModelsReport:
    def __init__(self):
        self.rmse_styled_df = None
        self.mse_styled_df = None
        self.mape_styled_df = None
        self.mae_styled_df = None
        self.model_cols = None
        self.test_df = None
        self.train_df = None
        self.mae_df = None
        self.mape_df = None
        self.mse_df = None
        self.rmse_df = None
        self.ground_truth_col = None
        self.trainResultsMap = {}

    def set_train_dataframe(self, train_df):
        self.train_df = train_df

    def set_test_dataframe(self, test_df):
        self.test_df = test_df

    def set_model_cols(self, model_cols):
        self.model_cols = model_cols

    def add_train_result(self, ar_model_spec, train_result):
        if isinstance(train_result, SARIMAXResultsWrapper) is False:
            raise Exception(
                'train_result should be an instance of statsmodels.tsa.statespace.sarimax.SARIMAXResults object')
        if isinstance(ar_model_spec, ARModelSpecification) is False:
            raise Exception(
                'ar_model_spec should be an instance of models.ar_model.ARModelSpecification object')
        self.trainResultsMap[ar_model_spec] = train_result

    def plot_models(self, multi_plot=False, include_ground_truth=True, xlim=None, save=False, show=True,
                    folder_path=''):
        plot_title = 'Out-of-sample Test Plot'
        plot_models(self.test_df, self.model_cols, x_format='week', title=plot_title,
                    ground_truth_col=self.ground_truth_col,
                    include_ground_truth=include_ground_truth, save=save, show=show,
                    folder_path=folder_path)  # TODO: add xlim here
        if multi_plot is True:
            for model_col in self.model_cols:
                plot_title = 'Out-of-sample test ' + str(model_col)
                plot_models(self.test_df, [model_col], x_format='week', title=plot_title,
                            ground_truth_col=self.ground_truth_col, include_ground_truth=include_ground_truth,
                            save=save, show=show, folder_path=folder_path)

    def highlight_min(self, df, col_subset, props=''):
        return

    def get_errors(self, highlight_min=True, highlight_max=False):

        self.mae_styled_df = self.mae_df.style.set_caption('MAE').highlight_min(axis=1, color='lightgreen',
                                                                                subset=self.mae_df.columns[2:])
        self.mape_styled_df = self.mape_df.style.set_caption('MAPE').highlight_min(axis=1, color='lightgreen',
                                                                                   subset=self.mape_df.columns[2:])
        self.mse_styled_df = self.mse_df.style.set_caption('MSE').highlight_min(axis=1, color='lightgreen',
                                                                                subset=self.mse_df.columns[2:])
        self.rmse_styled_df = self.rmse_df.style.set_caption('RMSE').highlight_min(axis=1, color='lightgreen',
                                                                                   subset=self.rmse_df.columns[2:])
        errors_map = {'mae': self.mae_styled_df, 'mape': self.mape_styled_df, 'mse': self.mse_styled_df,
                      'rmse': self.rmse_styled_df}
        return errors_map

    def show_errors(self, highlight_min=True, highlight_max=False):
        errors_map = self.get_errors(highlight_min=highlight_min, highlight_max=highlight_max)
        display(errors_map['mae'])
        display(errors_map['mape'])
        display(errors_map['mse'])
        display(errors_map['rmse'])
        # display(get_best_performing_models(mae_df))
        # display(get_best_performing_models(mape_df))
        # display(get_best_performing_models(mse_df))
        # display(get_best_performing_models(rmse_df))


def create_ar_models_report(data, ar_model_specs, train_interval, test_interval,
                            validation_interval=None, ground_truth_col='Disease Rate',
                            additional_model_cols=[], optimize_method=None,
                            train_maxiter=1000, cov_type=None, steps=1, add_train_result_obj=False):
    """
    Creates ARModelsReport object containing training and testing report of given models.

    Args
    ----------
    data : Dataframe
        Pandas dataframe containing input variables for the models. Should also include
        additional models whose column names can be specified through additional_model_cols parameter.

    ar_model_specs : list
        List of ARModelSpecification objects.

    train_interval: tuple
        Interval used to select the training data for models from the given dataframe.
        Left-most element should be start index of training data. Right-most element should
        be end index of training data.
        Both elements should be a datetime object in the same format as the index of the
        given data.

    test_interval: tuple
        Interval used to select the test data for models from the given dataframe.
        Same as train_interval.

    Returns
    -------
    ARModelsReport object.
    """
    if ar_model_specs is not None and isinstance(ar_model_specs, list) is False:
        raise Exception('ar_model_specs parameter must be a list of ARModelSpecification objects')
    if additional_model_cols is not None and isinstance(additional_model_cols, list) is False:
        raise Exception('additional models parameter must be a list of strings representing the column names of the '
                        'given data')

    ar_models_report = ARModelsReport()
    ar_models_report.ground_truth_col = ground_truth_col
    ar_model_wrappers = []
    model_name_list = []
    for i in range(0, len(ar_model_specs)):
        try:
            ar_model_spec = ar_model_specs[i]
            ar_wrapper = ARModelWrapper(ar_model_specs[i])

            ar_wrapper.set_data(data, ground_truth_col)
            ar_wrapper.split_dataset_by_intervals(train_interval=train_interval, test_interval=test_interval,
                                                  validation_interval=validation_interval, print_data=False)

            print("Training model {m} on interval {tr}...".format(m=str(ar_model_spec), tr=str(train_interval)))
            ar_wrapper.train_model(method=optimize_method, maxiter=train_maxiter, cov_type=cov_type)
            if add_train_result_obj is True:
                ar_models_report.add_train_result(ar_model_spec, ar_wrapper.train_result)

            print("Testing model {m} on interval {t}...".format(m=str(ar_model_spec), t=str(test_interval)))
            ar_wrapper.test_model(steps=steps)
            ar_model_wrappers.append(ar_wrapper)

            if ar_model_spec.model_name is not None:
                model_name_list.append(str(i) + ' ' + str(ar_model_spec.model_name))
            else:
                model_name_list.append(str(i))
        except:
            print(
                WARNING_TERM + 'create_ar_models_report: failed to generate report for model={m} on train_interval={''tri}, train_interval{ti} due to exception... resuming execution'.format(
                    m=str(ar_model_specs[i]), tri=str(train_interval), ti=str(test_interval)))

    i = 0
    # train_df = ar_model_wrappers[0].data.copy(deep=True)[train_interval[0]:train_interval[1]]
    test_df = ar_model_wrappers[0].data.copy(deep=True)[test_interval[0]:test_interval[1]]
    for ar_wrapper in ar_model_wrappers:
        # train_df[str(model_name_list[i])] = ar_wrapper.y_train_prediction
        test_df[str(model_name_list[i])] = ar_wrapper.y_test_prediction
        # display(test_df)
        i += 1

    for additional_model_col in additional_model_cols:
        # train_df[additional_model_col] = data[additional_model_col][train_interval[0]:train_interval[1]]
        test_df[additional_model_col] = data[additional_model_col][test_interval[0]:test_interval[1]]
        model_name_list.append(additional_model_col)

    # ar_models_report.set_train_dataframe(train_df)
    ar_models_report.set_test_dataframe(test_df)
    ar_models_report.set_model_cols(model_name_list)

    log.info("Creating error tables ...")
    print("Creating error tables ...")
    ar_models_report.mae_df = get_models_error(test_df, error_type='mae', actual_col_name='Disease Rate',
                                               predicted_col_names=model_name_list)
    ar_models_report.mape_df = get_models_error(test_df, error_type='mape', actual_col_name='Disease Rate',
                                                predicted_col_names=model_name_list)
    ar_models_report.mse_df = get_models_error(test_df, error_type='mse', actual_col_name='Disease Rate',
                                               predicted_col_names=model_name_list)
    ar_models_report.rmse_df = get_models_error(test_df, error_type='rmse', actual_col_name='Disease Rate',
                                                predicted_col_names=model_name_list)
    print("Finished creating error tables ...")
    print("ARModelsReport complete  ...")
    return ar_models_report


def create_all_ar_models_report(data, ar_model_specs, train_intervals, test_intervals,
                                validation_interval=None, ground_truth_col='Disease Rate',
                                additional_model_cols=[], optimize_method=None,
                                train_maxiter=1000, cov_type=None, steps=1):
    if len(train_intervals) != len(test_intervals):
        raise Exception(
            'length of train_intervals not equal to test_intervals. Make sure each train interval has an associated '
            'test interval.')
    # for test_interval in test_intervals:
    # if test_interval[0].year != test_interval[1].year:
    #     raise Exception(
    #         'all test interval must span single, disjoint years form each other (current implementation).')
    # TODO: check if all intervals are tuples, as they must be hashable

    test_intervals_to_reports_map = {'mae': pd.DataFrame(), 'mape': pd.DataFrame(), 'mse': pd.DataFrame(),
                                     'rmse': pd.DataFrame()}
    for i in range(0, len(train_intervals)):
        try:
            train_interval = train_intervals[i]
            test_interval = test_intervals[i]
            ar_models_report = create_ar_models_report(data, ar_model_specs, train_interval, test_interval,
                                                       validation_interval=validation_interval,
                                                       ground_truth_col=ground_truth_col,
                                                       additional_model_cols=additional_model_cols,
                                                       optimize_method=optimize_method,
                                                       train_maxiter=train_maxiter, cov_type=cov_type, steps=steps)
            # TODO: allow multiple years (hash by test_interval tuple). Currently only accepts test_intervals that span
            #  disjoint years
            test_intervals_to_reports_map[test_intervals[i]] = ar_models_report
            ar_models_report.get_errors()
            test_intervals_to_reports_map['mae'] = test_intervals_to_reports_map['mae'].append(ar_models_report.mae_df,
                                                                                               ignore_index=True)
            test_intervals_to_reports_map['mape'] = test_intervals_to_reports_map['mape'].append(
                ar_models_report.mape_df,
                ignore_index=True)
            test_intervals_to_reports_map['mse'] = test_intervals_to_reports_map['mse'].append(ar_models_report.mse_df,
                                                                                               ignore_index=True)
            test_intervals_to_reports_map['rmse'] = test_intervals_to_reports_map['rmse'].append(
                ar_models_report.rmse_df,
                ignore_index=True)
        except:
            print(
                WARNING_TERM + 'create_all_ar_models_report: failure on generating report for train_interval={tri}... '
                               'resuming execution'.format(
                    tri=str(train_intervals[i])))

    test_intervals_to_reports_map['mae'] = test_intervals_to_reports_map['mae'].style.set_caption('MAE').highlight_min(
        axis=1, color='lightgreen',
        subset=test_intervals_to_reports_map[
                   'mae'].columns[2:])
    test_intervals_to_reports_map['mape'] = test_intervals_to_reports_map['mape'].style.set_caption(
        'MAPE').highlight_min(axis=1, color='lightgreen',
                              subset=test_intervals_to_reports_map[
                                         'mape'].columns[2:])
    test_intervals_to_reports_map['mse'] = test_intervals_to_reports_map['mse'].style.set_caption('MSE').highlight_min(
        axis=1, color='lightgreen',
        subset=test_intervals_to_reports_map[
                   'mse'].columns[2:])
    test_intervals_to_reports_map['rmse'] = test_intervals_to_reports_map['rmse'].style.set_caption(
        'RMSE').highlight_min(axis=1, color='lightgreen',
                              subset=test_intervals_to_reports_map[
                                         'rmse'].columns[2:])
    print(FINISHED_TERM + ' create_all_ar_models_report: finished generating all report')
    return test_intervals_to_reports_map
