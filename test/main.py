import pandas as pd

from src.handler import (load_csv, optimize_missing_data, reduceVT_dataframe)
from src.utils import (describe_missing, update_missing, visualize_feature)
from src.machine_learning import (split_data)
from src.machine_learning.xgboost import build_XGB_model

## To display all the columns and not just the preview
pd.set_option('display.max_columns', None)

# df_adapted= load_csv("../data/processed/adapted.csv")
# print(df_adapted.shape)
# print((1583, 57713))

df_optimized = load_csv("../data/processed/optimized.csv")
print(df_optimized.shape)

# df_reduced = reduceVT_dataframe(df_adapted, 0.01)
# print(df_reduced.shape)

# df_optimized = optimize_missing_data(df_adapted, 0.47, 0.99)
# df_optimized= load_csv("/home/kevin/PycharmProjects/Idenficazione di biomarkers oncologici mediante AI/data/processed/adapted.csv")

## Check missing values for columns and rows
# update_missing(df_optimized, '../data/processed/column_missing_data.csv', '../data/processed/row_missing_data.csv')
## Print missing stats
# describe_missing('../data/processed/row_missing_data.csv', '../data/processed/column_missing_data.csv')

## Dividing X:characteristic_set and y:target_set
# X = df_optimized.drop(columns=['sample_status']).copy()
# y = df_optimized['sample_status'].copy()
# y.to_csv("../data/processed/target_set.csv", index=False)
# X.to_csv("../data/processed/characteristic_set.csv", index=False)
# visualize_feature(X, y)

## Split data
X_train, X_test, y_train, y_test = split_data(df_optimized, 'sample_status')
print(X_train.shape)
build_XGB_model(X_train, X_test, y_train, y_test)

