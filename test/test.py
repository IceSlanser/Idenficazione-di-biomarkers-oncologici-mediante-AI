import pandas as pd  # Load and manipulate cvs, also for One-Hot Encod.

from src.data_handle.extract import extract_type, extract_site
from src.data_handle.load import load_csv
from src.data_handle.transform import adapt_dataframe

# To display all the columns and not just the preview
pd.set_option('display.max_columns', None)

# Load the input data and extract the characteristics
df = load_csv("../data/raw/breast_raw.csv")
sample_type_df = extract_type(df)
sample_site_df = extract_site(df)

# Adapt the dataframe as wished: unique sample_id as rows and gene as columns merged with the type df
df_adapted = adapt_dataframe(df, sample_type_df)
df_adapted.to_csv('../data/processed/adapted.csv', index=False)