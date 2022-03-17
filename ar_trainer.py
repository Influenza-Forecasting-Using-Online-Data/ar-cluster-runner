import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

from data.utils import get_week_range_df, to_week_range
from models.ar_model import ARModelSpecification, create_all_ar_models_report
from models.persistence_model import create_persistence

log = logging.getLogger("models.ar_model")

# Global set-up
pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")

# Global set-up
pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")

GROUND_TRUTH_COLUMN = 'Disease Rate'
BASELINE_SHIFT = 1
PERSISTENCE_COL_NAME = 'Persistence'
LAGS = 20

DF = get_week_range_df('week range', include_search_terms=False)
DF = create_persistence(DF, BASELINE_SHIFT, persistance_col_name=PERSISTENCE_COL_NAME)

TR = [
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

T = [
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

TRAIN_INTERVALS = [(to_week_range(2004, 2), to_week_range(2010, 52))]
TEST_INTERVALS = [(to_week_range(2011, 1), to_week_range(2018, 52))]

MODEL_SPECS = [
    ARModelSpecification(order=(3, 0, 0), seasonal_order=(3, 0, 0, 52), model_class=SARIMAX),
    ARModelSpecification(order=(3, 1, 0), seasonal_order=(3, 1, 0, 52), model_class=SARIMAX),
    ARModelSpecification(order=(3, 1, 1), seasonal_order=(3, 1, 1, 52), model_class=SARIMAX),
    ARModelSpecification(order=(5, 0, 0), seasonal_order=(5, 0, 0, 52), model_class=SARIMAX),
    ARModelSpecification(order=(5, 1, 0), seasonal_order=(5, 1, 0, 52), model_class=SARIMAX),
    # ARModelSpecification(order=(20, 0, 10), model_class=SARIMAX),
]

OPTIMIZE_METHOD = 'powell'

OUTPUT_ROOT_DIR = "ar_runs"

if __name__ == "__main__":
    assert len(TRAIN_INTERVALS) == len(TEST_INTERVALS)
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.mkdir(OUTPUT_ROOT_DIR)
    folder_timestamp = str(datetime.now()).replace(":", "_").replace(".", "_")
    relative_output_path = os.path.join(OUTPUT_ROOT_DIR, folder_timestamp)
    print('Created report folder at %s ...\n' % str(relative_output_path))

    os.mkdir(relative_output_path)
    print('Started creating AR models report...\n')
    test_intervals_to_reports_map = create_all_ar_models_report(data=DF, ar_model_specs=MODEL_SPECS,
                                                                train_intervals=TRAIN_INTERVALS,
                                                                test_intervals=TEST_INTERVALS,
                                                                additional_model_cols=[PERSISTENCE_COL_NAME],
                                                                train_maxiter=1000, optimize_method=OPTIMIZE_METHOD,
                                                                cov_type=None)
    print('Finished creating AR models report...\n')
    print('Writing errors to xlsx ...\n')
    if OPTIMIZE_METHOD is None:
        OPTIMIZE_METHOD = ''
    with pd.ExcelWriter(
            os.path.join(relative_output_path, 'errors ' + OPTIMIZE_METHOD + ' ' + folder_timestamp + '.xlsx'),
            engine='xlsxwriter') as writer:
        test_intervals_to_reports_map['mae'].to_excel(writer, sheet_name='MAE')
        test_intervals_to_reports_map['mape'].to_excel(writer, sheet_name='MAPE')
        test_intervals_to_reports_map['mse'].to_excel(writer, sheet_name='MSE')
        test_intervals_to_reports_map['rmse'].to_excel(writer, sheet_name='RMSE')
    print('Finished writing errors to xlsx ...\n')
    print('Saving AR models plots...\n')
    for test_interval in TEST_INTERVALS:
        plots_folder_name = test_interval[0].strftime("%Y-%m-%d") + "_" + test_interval[1].strftime("%Y-%m-%d")
        plots_path = os.path.join(relative_output_path, plots_folder_name)
        os.mkdir(plots_path)
        test_intervals_to_reports_map[test_interval].plot_models(include_ground_truth=True, multi_plot=True, save=True,
                                                                 folder_path=plots_path, show=False)
    padding = "   "
    with open(os.path.join(relative_output_path, "report.txt"), 'w', encoding='utf-8') as f:
        f.write("SUMMARY\n-------\n\n")
        f.write("MODEL SPECIFICATIONS\n")
        for model_spec in MODEL_SPECS:
            f.write(padding + "{m}\n".format(m=str(model_spec)))
        f.write("\n")
        f.write("OPTIMIZATION METHOD = {o} \n\n".format(o=OPTIMIZE_METHOD))
        f.write("TRAINING/TESTING INTERVALS\n")
        for i in range(0, len(TRAIN_INTERVALS)):
            f.write(padding + "# " + str(i + 1) + ": \n" + padding + padding + "training=" + str(
                TRAIN_INTERVALS[i]) + "\n" + padding + padding + "testing=" + str(TEST_INTERVALS[i]) + "\n")
    print('Done...\n')
