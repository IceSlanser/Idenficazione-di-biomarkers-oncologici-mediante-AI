import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import seaborn as sns

# Update missing rows and columns
def update_missing(dataframe: pd.DataFrame) -> None:
    print("Updating missing data...")

    # Update row missing stats
    row_missing = dataframe.isnull().sum(axis=1)
    row_stats = row_missing.describe()
    row_stats.to_csv('../data/processed/missing/row_missing_stats.csv')
    print("Column stats:")
    print(row_stats)

    # Update column missing stats
    column_missing = dataframe.isnull().sum()
    column_stats = column_missing.describe()
    column_stats.to_csv('../data/processed/missing/column_missing_stats.csv')
    print("Row stats:")
    print(column_stats)

def visualize_feature(X:pd.DataFrame, y:pd.Series) -> None:
    print("Plot important features...")
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