import pandas as pd

from src.data_handle import(
    load_csv
)


def describe_missing(rows_path:str, column_path:str) -> None:
    # Print column missing stats
    df_column_missing = load_csv(column_path)
    column_stats = df_column_missing.describe()
    print("Column stats:")
    print(column_stats)

    # Print row missing stats
    df_row_missing = load_csv(rows_path)
    row_stats = df_row_missing.describe()
    print("Row stats:")
    print(row_stats)

# Save missing columns and rows
def update_missing(dataframe: pd.DataFrame, rows_path:str, column_path:str) -> None:
    dataframe.isnull().sum().to_csv('../data/processed/column_missing_data.csv')
    dataframe.isnull().sum(axis=1).to_csv('../data/processed/row_missing_data.csv')