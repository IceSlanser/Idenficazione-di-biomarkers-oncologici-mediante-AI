import pandas as pd

# Extract sample types from the dataframe
#   - dataframe: input dataframe
#   Extract 'sample_id' and  'value' when 'name1' is equal to 'sample_type'
#   Map type's values: 1 references to ill samples, 9 to healthy ones
def extract_type(dataframe: pd.DataFrame) -> pd.DataFrame:
    print("Extracting type...")
    type_df = dataframe[dataframe['name1'] == 'sample_type'][['sample_id', 'value']]

    # Lambda function: map 'value' into 1 or 9
    type_df['sample_status'] = type_df['value'].apply(lambda x: 0 if x in [11, 6] else 1)

    return type_df

# Extract sample site from the dataframe
#   - dataframe: input dataframe
#   Extract 'sample_id' and  'value' when 'name1' is equal to 'sample_site'
def extract_site(dataframe: pd.DataFrame) -> pd.DataFrame:
    print("Extracting site...")
    return dataframe[dataframe['name1'] == 'sample_site'][['sample_id', 'value']]