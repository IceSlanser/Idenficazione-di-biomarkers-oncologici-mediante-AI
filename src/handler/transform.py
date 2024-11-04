import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Prepare the dataframe to be processed
def adapt_dataframe(dataframe:pd.DataFrame, type_dataframe:pd.DataFrame) -> pd.DataFrame:
    # Remove unnecessary rows from the dataframe
    df_filtered = dataframe[~dataframe['name1'].isin(['sample_type', 'sample_site'])]

    # Merge the two columns referring at the same gene
    df_filtered = df_filtered.copy()
    df_filtered['gene'] = df_filtered['name1'] + '_' + df_filtered['name2']

    # Pivot the df to have unique 'sample_id' as rows and 'gene' as columns
    # reset_index() transforms 'sample_id' in a column
    df_pivot = df_filtered.pivot_table(index='sample_id', columns='gene', values='value')
    df_pivot.reset_index(inplace=True)

    # Merge the pivoted dataframe with the type_df
    df_adapted = df_pivot.merge(type_dataframe[['sample_id', 'sample_status']], on='sample_id', how='left')

    return df_adapted

# Rows and columns that do not reach the given thresholds will be removed
def optimize_missing_data(dataframe: pd.DataFrame, row_threshold: float, column_threshold: float) -> pd.DataFrame:
    # Check if the thresholds are given as percentages (greater than 1)
    if row_threshold > 1:
        row_threshold = row_threshold / 100
    if column_threshold > 1:
        column_threshold = column_threshold / 100

    # Convert to int because thresh expects an integer
    dataframe = dataframe.dropna(thresh=int(len(dataframe.columns) * row_threshold), axis=0)
    dataframe = dataframe.dropna(thresh=int(len(dataframe) * column_threshold), axis=1)

    return dataframe

# Reduce the given dataframe with VarianceThreshold
def reduceVT_dataframe(dataframe:pd.DataFrame, thresh:float) -> pd.DataFrame:
    # Create a selector
    selector = VarianceThreshold(threshold=thresh)

    # Fit the selector the input dataframe
    df_reduced = selector.fit_transform(dataframe)

    # The selector returns a numpy.ndarray so we are converting it back as a pd.DataFrame
    return pd.DataFrame(df_reduced, columns=dataframe.columns[selector.get_support(indices=True)])