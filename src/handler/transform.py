import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


# Prepare the dataframe to be processed
def adapt_dataframe(dataframe:pd.DataFrame, type_dataframe:pd.DataFrame) -> pd.DataFrame:
    print("Adapting dataframe...")
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
    df_adapted = df_adapted[df_adapted['sample_status'].notna()]

    return df_adapted

# Rows and columns that do not reach the given thresholds will be removed
def optimize_missing_data(dataframe: pd.DataFrame, row_threshold: float, column_threshold: float) -> pd.DataFrame:
    print("Optimizing missing data...")
    # Convert to int because thresh expects an integer
    dataframe = dataframe.dropna(thresh=int(len(dataframe.columns) * row_threshold), axis=0)
    dataframe = dataframe.dropna(thresh=int(len(dataframe) * column_threshold), axis=1)

    return dataframe

# Reduce the given dataframe with VarianceThreshold
def reduceVT_dataframe(dataframe:pd.DataFrame, thresh:float) -> pd.DataFrame:
    print("Reducing VT data...")
    # Create a selector
    selector = VarianceThreshold(threshold=thresh)

    # Fit the selector the input dataframe
    df_reduced = selector.fit_transform(dataframe)

    # The selector returns a numpy.ndarray so we are converting it back as a pd.DataFrame
    return pd.DataFrame(df_reduced, columns=dataframe.columns[selector.get_support(indices=True)])

# Classify not defined status into Healthy or Ill
def classify_not_defined_with_kmeans(dataframe: pd.DataFrame) -> pd.DataFrame:
    print("Classifying not defined with k-means...")

    # Divide datas
    classified_data = dataframe[dataframe['sample_status'] != -1].copy()
    not_defined_data = dataframe[dataframe['sample_status'] == -1].copy()

    # Prepare training data
    genes = [col for col in dataframe.columns if col not in {'sample_id','sample_status'}]
    X_classified = classified_data[genes]
    X_not_defined = not_defined_data[genes]

    # Normalize gene values
    scaler = StandardScaler()
    X_classified_scaled = scaler.fit_transform(X_classified)
    X_not_defined_scaled = scaler.transform(X_not_defined)

    # Principal Component Analysis, reduce redundancy and speed up the process, using the 10 most important components
    pca = PCA(n_components=10, random_state=42)
    X_classified_pca = pca.fit_transform(X_classified_scaled)
    print("X_classified_pca cumulative variance", np.cumsum(pca.explained_variance_ratio_))
    X_not_defined_pca = pca.transform(X_not_defined_scaled)

    # Apply k-means with 2 clusters (0 and 1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_classified_pca)

    # Mapping cluster values
    classified_data['cluster'] = kmeans.labels_
    cluster_map = {
        classified_data[classified_data['sample_status'] == 0]['cluster'].mode().iloc[0]: 0,
        classified_data[classified_data['sample_status'] == 1]['cluster'].mode().iloc[0]: 1
    }

    # Predict not_defined values with kmeans and cluster map and update the dataframe
    not_defined_clusters = kmeans.predict(X_not_defined_pca)
    not_defined_data['sample_status'] = [cluster_map[cluster] for cluster in not_defined_clusters]
    dataframe.loc[dataframe['sample_status'] == -1, 'sample_status'] = not_defined_data['sample_status'].values

    return dataframe