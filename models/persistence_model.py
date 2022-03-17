import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def create_persistence(data, shift, persistance_col_name='Persistance'):
    """
    Parameters:
    data (DataFrame): original data
    shift (int): number of weeks to shift forward, must be greater than 0.
    
    Returns:
    DataFrame containing baseline prediction column with label 'Prediction Rate'
    """
    if shift <= 0:
        raise ValueError('shift must be greater than 0')
    data_baseline = data.copy()
    data_baseline[persistance_col_name] = data_baseline['Disease Rate'].shift(shift)
    data_baseline.drop(data_baseline.head(shift).index, inplace=True)  # drop first 'shift' num of rows
    return data_baseline

# def plot_yearly_data()  
#   fig, ax = plt.subplots(figsize=(30, 10))
#
#   df_with_1week_baseline.plot(x='year week', y='Disease Rate', x_compat=True, ax=ax)
#   df_with_1week_baseline.plot(x='year week', y='Predicted Rate', x_compat=True, ax=ax)
#
#   ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
#   ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#
#   plt.legend(labels=["Baseline ILI Prediction","Reported ILI Rates"], loc="upper right")
#   plt.show()
