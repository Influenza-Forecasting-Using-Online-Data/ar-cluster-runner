#!/usr/bin/env python

import numpy as np
from pmdarima import auto_arima

from data.utils import get_week_range_df, to_week_range


import logging
import contextlib

log = logging.getLogger("ar_trainer_cluster")

GROUND_TRUTH_COLUMN = 'Disease Rate'
PERSISTENCE_COL_NAME = 'Persistence'
BASELINE_SHIFT = 1

INCLUDE_EXOG = False

DF = get_week_range_df('week range', include_search_terms=INCLUDE_EXOG)

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


if __name__ == "__main__":
    #test_types = ['aicc', 'hqic', 'oob', 'aic', 'bic']
    test_types = ['aic', 'bic']

    print('Start TESTS: \n\n\n')

    for type in test_types:
        auto_arima(y=DF[GROUND_TRUTH_COLUMN], start_p=0, start_q=0,
                   test='adf',
                   max_p=1, max_q=1,
                   seasonal=True,
                   m=52,
                   max_d=1,
                   start_P=0,
                   max_P=1,
                   start_Q=0,
                   max_Q=1,
                   max_D=1,
                   information_criterion=type,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=False,
                   n_fits=50,
                   n_jobs=1,
                   maxiter=500)

        # auto_arima(y=DF[GROUND_TRUTH_COLUMN], start_p=0, start_q=0,
        #            test='adf',
        #            max_p=10, max_q=10,
        #            seasonal=True,
        #            m=52,
        #            max_d=2,
        #            start_P=0,
        #            max_P=10,
        #            start_Q=0,
        #            max_Q=10,
        #            max_D=1,
        #            information_criterion=type,
        #            trace=True,
        #            error_action='ignore',
        #            suppress_warnings=True,
        #            stepwise=False,
        #            n_fits=50,
        #            n_jobs=32,
        #            maxiter=500)
        print("Test type: {type}".format(type=type))
        print('\n\n')

    print('End INTERVALS.')
