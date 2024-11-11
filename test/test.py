import pandas as pd  # Load and manipulate cvs, also for One-Hot Encod.

from src.handler import(
    load_csv, extract_type, extract_site, adapt_dataframe, optimize_missing_data, reduceVT_dataframe
)
# To display all the columns and not just the preview
pd.set_option('display.max_columns', None)

# Load the input data and extract the characteristics
df = load_csv("../data/raw/cancer.csv")
print(df.shape)
sample_type_df = extract_type(df)
sample_site_df = extract_site(df)

# Adapt the dataframe as wished: unique sample_id as rows and gene as columns merged with the type df
df_adapted = adapt_dataframe(df, sample_type_df)
df_adapted.to_csv("../data/processed/adapted.csv", index=False)

# Optimize the dataframe removing rows and columns with a lot of missing datas
df_optimized = optimize_missing_data(df_adapted, 0.0, 0.1)
df_optimized.to_csv("../data/processed/optimized.csv", index=False)

