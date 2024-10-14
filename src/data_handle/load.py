import pandas as pd


# Load input csv into a dataframe
#   - path: csv's location
#   - nrows: (default all) rows to be loaded
#   - skiprows: (default none) rows to be skipped at the beginning
def load_csv(path, nrows=None, skiprows=None):
    return pd.read_csv(path, nrows=nrows, skiprows=skiprows)
