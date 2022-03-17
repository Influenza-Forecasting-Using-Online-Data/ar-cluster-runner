import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_models_single(data, ground_truth_col_name, model_col_names, interval=None, figsize=(30, 10), ticks_fontsize=16,
                       legend_fontsize=20, title='', title_fontsize=20, label_fontsize=22,
                       xlim=None, x_format='month'):
    for model_col_name in model_col_names:
        plot_models_multi(data, ground_truth_col_name, [model_col_name], interval=interval, figsize=figsize, ticks_fontsize=ticks_fontsize,
                          legend_fontsize=legend_fontsize, title=title, title_fontsize=title_fontsize,
                          label_fontsize=label_fontsize, xlim=xlim, x_format=x_format)


def plot_models_multi(data, ground_truth_col_name, model_col_names, interval=None, figsize=(30, 10), ticks_fontsize=16,
                      legend_fontsize=20, title='', title_fontsize=20, label_fontsize=22,
                      xlim=None, x_format='month'):
    if model_col_names is not None and isinstance(model_col_names, list) is False:
        raise Exception('additional models parameter must be a list of strings representing the column names of the '
                        'given data')
    if interval is None:
        interval = [data.index[0], data.index[-1]]

    if x_format == 'month':
        x_axis_locator = mdates.MonthLocator(bymonthday=1)
        x_axis_formatter = mdates.DateFormatter('%Y %b')
    elif x_format == 'week':
        x_axis_locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        x_axis_formatter = mdates.ConciseDateFormatter(x_axis_locator)
    else:
        raise Exception("x_format must be 'month' or 'week'")

    fig, ax = plt.subplots(figsize=figsize)
    data.plot(use_index=True, y=ground_truth_col_name, x_compat=True, ax=ax)
    for model_col_name in model_col_names:
        data.plot(use_index=True, y=str(model_col_name), x_compat=True, ax=ax)

    ax.xaxis.set_major_locator(x_axis_locator)
    ax.xaxis.set_major_formatter(x_axis_formatter)

    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    ax.set_xlabel(str(interval[0]) + ' - ' + str(interval[1]), fontsize=label_fontsize)
    ax.set_ylabel('ILI Rate', fontsize=label_fontsize)

    plt.legend(fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)

    if xlim is not None and len(xlim) == 2:
        plt.xlim(xlim[0], xlim[1])

    plt.show()
