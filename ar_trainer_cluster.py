import logging
import os
import time
from datetime import datetime

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data.utils import get_week_range_df, to_week_range
from models.ar_model import ARModelSpecification

GROUND_TRUTH_COLUMN = 'Disease Rate'
PERSISTENCE_COL_NAME = 'Persistence'
BASELINE_SHIFT = 1

INCLUDE_EXOG = False

DF = get_week_range_df('week range', include_search_terms=INCLUDE_EXOG)
# DF = create_persistence(DF, BASELINE_SHIFT, persistance_col_name=PERSISTENCE_COL_NAME)

TRAIN_INTERVALS = [
    (to_week_range(2004, 2), to_week_range(2008, 52)),
    (to_week_range(2005, 1), to_week_range(2009, 52)),
    (to_week_range(2006, 2), to_week_range(2010, 52)),
    (to_week_range(2007, 2), to_week_range(2011, 52)),
    (to_week_range(2008, 2), to_week_range(2012, 52)),
    (to_week_range(2009, 2), to_week_range(2013, 52)),
    (to_week_range(2010, 2), to_week_range(2014, 52)),
    (to_week_range(2011, 2), to_week_range(2015, 52)),
    (to_week_range(2012, 2), to_week_range(2016, 52)),
    (to_week_range(2013, 2), to_week_range(2017, 52)),
]

TEST_INTERVALS = [
    (to_week_range(2009, 1), to_week_range(2009, 52)),
    (to_week_range(2010, 1), to_week_range(2010, 52)),
    (to_week_range(2011, 2), to_week_range(2011, 52)),
    (to_week_range(2012, 2), to_week_range(2012, 52)),
    (to_week_range(2013, 2), to_week_range(2013, 52)),
    (to_week_range(2014, 2), to_week_range(2014, 52)),
    (to_week_range(2015, 2), to_week_range(2015, 52)),
    (to_week_range(2016, 2), to_week_range(2016, 52)),
    (to_week_range(2017, 2), to_week_range(2017, 52)),
    (to_week_range(2018, 2), to_week_range(2018, 52)),
]

MODEL_SPECS = [
    ARModelSpecification(order=(0, 1, 2), model_class=SARIMAX),  # best BIC
    ARModelSpecification(order=(0, 1, 2), seasonal_order=(1, 0, 0, 52), model_class=SARIMAX),
    ARModelSpecification(order=(0, 1, 2), seasonal_order=(0, 0, 1, 52), model_class=SARIMAX),
    ARModelSpecification(order=(0, 1, 3), model_class=SARIMAX),
    ARModelSpecification(order=(1, 1, 1), model_class=SARIMAX),
    ARModelSpecification(order=(2, 1, 1), seasonal_order=(1, 0, 1, 52), model_class=SARIMAX),
    ARModelSpecification(order=(2, 1, 1), seasonal_order=(1, 0, 2, 52), model_class=SARIMAX),  # best AIC
    ARModelSpecification(order=(2, 1, 1), seasonal_order=(0, 0, 2, 52), model_class=SARIMAX),
    ARModelSpecification(order=(2, 1, 1), seasonal_order=(1, 0, 3, 52), model_class=SARIMAX),
    ARModelSpecification(order=(3, 1, 1), seasonal_order=(0, 0, 2, 52), model_class=SARIMAX),
]

STEPS = 1
OPTIMIZE_METHOD = 'powell'
MAXITER = 500
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


def test_model(endog_all, train_result, start, end, steps=1):
    test_result = None
    y_test_prediction = None
    if steps == 1:
        test_result = train_result.apply(endog=endog_all, refit=False)
        y_test_prediction = test_result.predict(start=start, end=end, dynamic=False)
    else:
        endog_test = endog_all[start:end]
        steps_ahead_forecasts = endog_test.copy(deep=True).iloc[0:steps - 1]
        for i in range(0, len(endog_test) - steps + 1):
            test_result = train_result.append(endog=[endog_test.iloc[i]], refit=False)
            forecast_point_steps_ahead = pd.Series(test_result.forecast(steps=steps)[steps - 1],
                                                   index=[endog_test.index[i + steps - 1]])
            # print(forecast_point_steps_ahead)
            steps_ahead_forecasts = pd.concat([steps_ahead_forecasts, forecast_point_steps_ahead], axis=0,
                                              ignore_index=False)
        y_test_prediction = steps_ahead_forecasts
    # if self.model is not None:
    #     self.model.endog = self.train_data
    return y_test_prediction, test_result


def write_summary(relative_output_path):
    padding = "  "
    with open(os.path.join(relative_output_path, "report.txt"), 'w', encoding='utf-8') as f:
        f.write("SUMMARY\n-------\n\n")
        f.write("MODEL SPECIFICATIONS\n")
        for model_spec in MODEL_SPECS:
            f.write(padding + "{m}\n".format(m=str(model_spec)))
        f.write("\n")
        f.write("OPTIMIZATION METHOD = {o} \n\n".format(o=OPTIMIZE_METHOD))
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
    folder_timestamp = str(datetime.now()).replace(":", "_").replace(".", "_")
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

        test_result_folder_path = os.path.join(relative_output_path, prettify_interval(test_interval))
        if not os.path.exists(test_result_folder_path):
            os.mkdir(test_result_folder_path)

        for model_spec in MODEL_SPECS:
            try:
                model = model_spec.init_model(endog=DF[GROUND_TRUTH_COLUMN][train_interval[0]:train_interval[1]])

                LOG.info(
                    "TRAIN model_spec={m} on train_interval={tri}".format(m=str(model_spec), tri=prettify_interval(
                        train_interval)))
                train_result = train_model(model, method=OPTIMIZE_METHOD, maxiter=MAXITER, cov_type=COV_TYPE)

                LOG.info("TEST model_spec={m} on test_interval={ti}".format(m=str(model_spec),
                                                                            ti=prettify_interval(test_interval)))
                y_test_prediction, test_result = test_model(DF[GROUND_TRUTH_COLUMN], train_result,
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
            except:
                LOG.error("Failed running model_spec={m} on train_interval={tri}, test_interval={ti}".format(
                    m=str(model_spec), tri=str(train_interval), ti=str(test_interval)))
        all_test_predictions_df = all_test_predictions_df.append(predictions_df, ignore_index=False)
    all_test_predictions_df.to_csv(os.path.join(relative_output_path, "predictions_df.csv"))
    LOG.info("FINISHED")
    LOG.info("ELAPSED TIME = %s s" % (time.time() - start_time))


if __name__ == '__main__':
    run()
