import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import seaborn as sns
from src.handler import(
    load_csv
)


def describe_missing(rows_path:str, column_path:str) -> None:
    # Print column missing stats
    df_column_missing = load_csv(column_path)
    column_stats = df_column_missing.describe()
    column_stats.to_csv('../data/processed/column_missing_stats.csv')
    print("Row stats:")
    print(column_stats)

    # Print row missing stats
    df_row_missing = load_csv(rows_path)
    row_stats = df_row_missing.describe()
    row_stats.to_csv('../data/processed/row_missing_stats.csv')
    print("Column stats:")
    print(row_stats)

# Save missing columns and rows
def update_missing(dataframe: pd.DataFrame, rows_path:str, column_path:str) -> None:
    dataframe.isnull().sum().to_csv(column_path, index=False)
    dataframe.isnull().sum(axis=1).to_csv(rows_path, index=False)


def visualize_feature(X:pd.DataFrame, y:pd.DataFrame) -> None:
    model = XGBClassifier(random_state=42)
    model.fit(X, y)

    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])

    plt.figure(figsize=(10, 8))
    sns.heatmap(X.isnull(), cbar=False)
    plt.title("Mappa dei Dati Mancanti")
    plt.show()