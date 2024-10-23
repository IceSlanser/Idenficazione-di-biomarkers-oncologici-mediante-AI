import pandas as pd

# Extract sample types from the dataframe
#   - dataframe: input dataframe
#   Extract 'sample_id' and  'value' when 'name1' is equal to 'sample_type'
#   Map type's values: 1 references to ill samples, 9 to healthy ones
def extract_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    type_df = dataframe[dataframe['name1'] == 'sample_type'][['sample_id', 'value']]

    # Lambda function: map 'value' into 1 or 9
    type_df['sample_status'] = type_df['value'].apply(lambda x: 1 if x in [1, 6, 85, 86] else 9)

    return type_df

# Extract sample site from the dataframe
#   - dataframe: input dataframe
#   Extract 'sample_id' and  'value' when 'name1' is equal to 'sample_site'
def extract_site(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe[dataframe['name1'] == 'sample_site'][['sample_id', 'value']]