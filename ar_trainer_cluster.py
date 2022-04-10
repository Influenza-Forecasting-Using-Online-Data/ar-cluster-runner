import logging
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import preprocessing

from data.utils import get_week_range_df, to_week_range, get_train_and_test_intervals
from models.ar_model import ARModelSpecification


def read_num_steps():
    try:
        return int(sys.argv[1])
    except:
        print("Cmd line argument STEPS must be integer value greater than 0")
        raise Exception("Cmd line argument STEPS must be integer value greater than 0")

    if STEPS < 1:
        print("Cmd line argument STEPS must be integer value greater than 0")
        raise Exception("Cmd line argument STEPS must be integer value greater than 0")


GROUND_TRUTH_COLUMN = 'Disease Rate'

INCLUDE_EXOG = True  # If False, ignores SEARCH_QUERY_THRESHOLD value and does not add any search query terms to dataframe
SEARCH_QUERY_THRESHOLD = 50  # options include 50, 100, 150, 200, 250, 300, 350, 400

OUTSEASON_START_WEEK = 23
OUTSEASON_END_WEEK = 38

ENFORCE_STATIONARITY = True
ENFORCE_INVERTIBILITY = True

SCALER = preprocessing.StandardScaler()  # None or sklearn.preprocessing scaler object (fit_transform interface)

DF = get_week_range_df('week range', include_search_terms=INCLUDE_EXOG, search_query_threshold=SEARCH_QUERY_THRESHOLD,
                       outseason_start_week=OUTSEASON_START_WEEK, outseason_end_week=OUTSEASON_END_WEEK,
                       exog_scaler=SCALER)

EXOG = None
if INCLUDE_EXOG is True:
    EXOG = DF.drop(labels=['Disease Rate', 'year', 'week'], axis=1, inplace=False)

TRAIN_INTERVAL_NUM_YEARS = 5  # positive integer e.g., 2, 5, 10

TRAIN_INTERVALS, TEST_INTERVALS = get_train_and_test_intervals(TRAIN_INTERVAL_NUM_YEARS, OUTSEASON_START_WEEK,
                                                               OUTSEASON_END_WEEK)

SEASON_PERIOD = 52 - OUTSEASON_END_WEEK + 1 + OUTSEASON_START_WEEK

MODEL_SPECS = [
    # best BIC (without differencing)
    ARModelSpecification(order=(1, 0, 1), model_class=SARIMAX),
    # best AIC (without differencing)
    ARModelSpecification(order=(1, 0, 1), seasonal_order=(2, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
    ARModelSpecification(order=(1, 1, 1), model_class=SARIMAX),
    ARModelSpecification(order=(1, 1, 1), seasonal_order=(2, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
    # best AIC (with differencing)
    ARModelSpecification(order=(2, 1, 1), seasonal_order=(0, 0, 2, SEASON_PERIOD), model_class=SARIMAX),
    # best BIC (with differencing)
    ARModelSpecification(order=(0, 1, 2), model_class=SARIMAX),
    # contender - short-lag
    ARModelSpecification(order=(3, 0, 2), model_class=SARIMAX),
    ARModelSpecification(order=(3, 0, 2), seasonal_order=(1, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
    # contender - medium-lag
    ARModelSpecification(order=(5, 0, 3), model_class=SARIMAX),
    ARModelSpecification(order=(5, 0, 3), seasonal_order=(1, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
    ARModelSpecification(order=(5, 0, 3), seasonal_order=(1, 0, 1, SEASON_PERIOD), model_class=SARIMAX),
    ARModelSpecification(order=(5, 0, 3), seasonal_order=(2, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
    ARModelSpecification(order=(5, 0, 3), seasonal_order=(2, 0, 1, SEASON_PERIOD), model_class=SARIMAX),
    # contender - long-lag
    ARModelSpecification(order=(10, 0, 3), seasonal_order=(1, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
    ARModelSpecification(order=(10, 0, 3), seasonal_order=(1, 0, 1, SEASON_PERIOD), model_class=SARIMAX),
    ARModelSpecification(order=(10, 0, 3), seasonal_order=(2, 0, 0, SEASON_PERIOD), model_class=SARIMAX),
]

STEPS = read_num_steps()
OPTIMIZE_METHOD = 'powell'
MAXITER = 1800
COV_TYPE = None

PICKLE_TEST_RESULT = False  # boolean

# Plan to create scripts on UCL nodes:
# - Calculate AIC, BIC on the entire dataset => output results to files (use pmdarima.auto_arima)
# - Run 3-year training window with 1-year testing window
# - Run 5-year training window with 1-year testing window
# - Run 10-year training window with 1-year testing window
# - Run 15-year training window with 1-year testing window
# Note: should keep training windows small as AR tends to have converging patterns which affect performance (is not a global model, see link below)
#
# https://stats.stackexchange.com/questions/453386/working-with-time-series-data-splitting-the-dataset-and-putting-the-model-into (partea cu AR, fara LSTM)


OUTPUT_ROOT_DIR = "ar_trainer_cluster_out"


def train_model(model, method='powell', maxiter=500, cov_type=None):
    train_result = None
    if isinstance(model, ARIMA):
        train_result = model.fit(method=method, cov_type=cov_type)
    else:
        train_result = model.fit(method=method, maxiter=maxiter, cov_type=cov_type)
    return train_result


def test_model(endog_all, exog_all, train_result, start, end, steps=1):
    y_test_prediction = None
    test_result = train_result.apply(endog=endog_all, exog=exog_all, refit=False)
    endog_test = endog_all[start:end]
    if steps == 1:
        y_test_prediction = test_result.predict(start=endog_test.index[0], end=endog_test.index[-1], dynamic=False)
    else:
        steps_ahead_forecasts = endog_test.copy(deep=True).iloc[0:steps - 1]

        for i in range(0, len(endog_test) - steps + 1):
            index_at_i = endog_test.index[i]
            index_at_steps_ahead = endog_test.index[i + steps - 1]

            forecast_point_steps_ahead = pd.Series(
                test_result.predict(start=index_at_i, end=index_at_steps_ahead, dynamic=True)[steps - 1],
                index=[index_at_steps_ahead])

            steps_ahead_forecasts = pd.concat([steps_ahead_forecasts, forecast_point_steps_ahead], axis=0,
                                              ignore_index=False)
        y_test_prediction = steps_ahead_forecasts
    return y_test_prediction, test_result


def write_summary(relative_output_path):
    padding = "  "
    with open(os.path.join(relative_output_path, "report.txt"), 'w', encoding='utf-8') as f:
        f.write("SUMMARY\n-------\n\n")
        f.write("MODEL SPECIFICATIONS\n")
        for model_spec in MODEL_SPECS:
            f.write(padding + "{m}\n".format(m=str(model_spec)))
        f.write("\n")
        f.write("STEPS = %i\n\n" % STEPS)
        f.write("OPTIMIZATION METHOD = {o} \n\n".format(o=OPTIMIZE_METHOD))
        f.write("EXOGENOUS VARIABLES = {b}\n".format(b=INCLUDE_EXOG))
        if INCLUDE_EXOG:
            f.write(padding + "EXOGENOUS THRESHOLD = {t}\n".format(t=SEARCH_QUERY_THRESHOLD))
            f.write(padding + "EXOGENOUS SCALER = {s}\n\n".format(
                s=str(SCALER.__repr__() if SCALER is not None else 'None')))
        f.write("TRAINING/TESTING INTERVALS\n")
        for i in range(0, len(TRAIN_INTERVALS)):
            f.write(padding + "# " + str(i + 1) + ": \n" + padding + padding + "training=" + prettify_interval(
                TRAIN_INTERVALS[i]) + "\n" + padding + padding + "testing=" + prettify_interval(
                TEST_INTERVALS[i]) + "\n")


def prettify_interval(time_interval):
    return "(" + time_interval[0].strftime("%Y-%m-%d") + ", " + time_interval[1].strftime("%Y-%m-%d") + ")"


def run():
    start_time = time.time()
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.mkdir(OUTPUT_ROOT_DIR)
    folder_timestamp = str(datetime.now()).replace(":", "_").replace(".", "_").replace(" ", "_")
    relative_output_path = os.path.join(OUTPUT_ROOT_DIR, folder_timestamp)
    os.mkdir(relative_output_path)
    print('Created report folder at %s ...\n' % str(relative_output_path))

    log_format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename=os.path.join(relative_output_path, "logfile.log"),
                        filemode="w",
                        format=log_format,
                        level=logging.NOTSET)
    LOG = logging.getLogger("logger")
    LOG.setLevel(logging.INFO)
    LOG.info("STARTED")

    write_summary(relative_output_path)

    all_test_predictions_df = pd.DataFrame()
    for i in range(0, len(TRAIN_INTERVALS)):
        train_interval = TRAIN_INTERVALS[i]
        test_interval = TEST_INTERVALS[i]
        predictions_df = DF.loc[test_interval[0]:test_interval[1]].copy(deep=True)

        if PICKLE_TEST_RESULT is True:
            test_result_folder_path = os.path.join(relative_output_path, prettify_interval(test_interval))
            if not os.path.exists(test_result_folder_path):
                os.mkdir(test_result_folder_path)

        for model_spec in MODEL_SPECS:
            try:
                model = None
                if EXOG is not None:
                    model = model_spec.init_model(endog=DF[GROUND_TRUTH_COLUMN][train_interval[0]:train_interval[1]],
                                                  exog=EXOG[train_interval[0]:train_interval[1]],
                                                  enforce_stationarity=ENFORCE_STATIONARITY,
                                                  enforce_invertibility=ENFORCE_INVERTIBILITY)
                else:
                    model = model_spec.init_model(endog=DF[GROUND_TRUTH_COLUMN][train_interval[0]:train_interval[1]],
                                                  enforce_stationarity=ENFORCE_STATIONARITY,
                                                  enforce_invertibility=ENFORCE_INVERTIBILITY)

                LOG.info(
                    "TRAIN model_spec={m} on train_interval={tri}".format(m=str(model_spec), tri=prettify_interval(
                        train_interval)))
                train_result = train_model(model, method=OPTIMIZE_METHOD, maxiter=MAXITER, cov_type=COV_TYPE)

                LOG.info("TEST model_spec={m} on test_interval={ti}".format(m=str(model_spec),
                                                                            ti=prettify_interval(test_interval)))
                y_test_prediction, test_result = test_model(endog_all=DF[GROUND_TRUTH_COLUMN],
                                                            exog_all=EXOG,
                                                            train_result=train_result,
                                                            start=test_interval[0],
                                                            end=test_interval[1],
                                                            steps=STEPS)
                predictions_df[model_spec.model_name] = y_test_prediction

                if PICKLE_TEST_RESULT:
                    test_result_file_name = model_spec.model_name + "_test_result_obj"
                    test_result_file_path = os.path.join(test_result_folder_path, test_result_file_name)
                    LOG.info("PICKLE test_result in file={f}".format(f=test_result_file_name))
                    with open(test_result_file_path, "w") as f:
                        test_result.save(test_result_file_path)
            except Exception:
                LOG.error("Failed running model_spec={m} on train_interval={tri}, test_interval={ti}".format(
                    m=str(model_spec), tri=str(train_interval), ti=str(test_interval)))
                LOG.error(traceback.format_exc())
        all_test_predictions_df = all_test_predictions_df.append(predictions_df, ignore_index=False)
    all_test_predictions_df.to_csv(os.path.join(relative_output_path, "predictions_df.csv"))
    LOG.info("FINISHED")
    LOG.info("ELAPSED TIME = %s s" % (time.time() - start_time))


if __name__ == '__main__':
    run()
