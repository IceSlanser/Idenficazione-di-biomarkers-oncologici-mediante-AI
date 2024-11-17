import pandas as pd

from src.handler import (load_csv, optimize_missing_data, reduceVT_dataframe)
from src.utils import (update_missing, visualize_feature)
from src.machine_learning import (split_data)
from src.machine_learning.xgboost import build_XGB_model

## To display all the columns and not just the preview
pd.set_option('display.max_columns', None)

## Load adapted dataframe
# df_adapted= load_csv("../data/processed/adapted.csv")
# print(df_adapted.shape)

## Load reduced dataframe
# df_reduced = reduceVT_dataframe(df_adapted, 0.01)
# print(df_reduced.shape)

## Load optimized dataframe
df_optimized= load_csv("../data/processed/optimized.csv")
# print(df_optimized.shape)

## Check missing values for columns and rows
# update_missing(df_optimized)

## Split data
X_train, X_test, y_train, y_test = split_data(df_optimized, 'sample_status')
print(X_test.shape)

## Visualize feature
# visualize_feature(X_train, y_train)

## Data training and tree building
build_XGB_model(X_train, X_test, y_train, y_test)

