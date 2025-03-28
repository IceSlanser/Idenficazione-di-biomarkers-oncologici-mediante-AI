import pandas as pd

# Extract sample types from the dataframe
#   - dataframe: input dataframe
#   Extract 'sample_id' and  'value' when 'name1' is equal to 'sample_type'
#   Map type's values: 1 references to ill samples, 0 to healthy ones, -1 not defined
def extract_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    print("Extracting type...")
    type_df = dataframe[dataframe['name1'] == 'sample_type'][['sample_id', 'value']]

    # Lambda function: map 'value' into 1, 0, -1
    type_df['sample_status'] = type_df['value'].apply(lambda x: 0 if x in [11] else (1 if x in [1, 85, 86, 2] else -1))
    type_df = type_df[type_df['sample_status'] != -1]

    return type_df

# Extract sample site from the dataframe
#   - dataframe: input dataframe
#   Extract 'sample_id' and  'value' when 'name1' is equal to 'sample_site'
def extract_site(dataframe: pd.DataFrame) -> pd.DataFrame:
    print("Extracting site...")
    return dataframe[dataframe['name1'] == 'sample_site'][['sample_id', 'value']]