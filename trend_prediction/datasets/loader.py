import pandas as pd


def load_data(filename, is_pickle=False, main_time_series=None, na_values=None, header=0, main_only=True, separator=',', fill_na_method='pad'):
    if is_pickle:
        data = pd.read_pickle(filename)
    else:
        if main_only:
            data = pd.read_csv(filename, sep=separator, usecols=[main_time_series],
                             na_values=na_values, header=header)
            data = data.squeeze()
        else:
            data = pd.read_csv(filename, sep=separator, na_values=na_values, header=header)
    if fill_na_method is not None:
        data = data.fillna(method=fill_na_method)
    return data


