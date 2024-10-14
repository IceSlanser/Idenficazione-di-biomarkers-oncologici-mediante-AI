# Prepare the dataframe to be processed
def adapt_dataframe(dataframe, type_dataframe):
    # Remove unnecessary rows from the dataframe
    df_filtered = dataframe[~dataframe['name1'].isin(['sample_type', 'sample_site'])]

    # Merge the two columns referring at the same gene
    df_filtered.loc[:, 'gene'] = df_filtered['name1'] + '_' + ['name2']

    # Pivot the df to have unique 'sample_id' as rows and 'gene' as columns
    # reset_index() transforms 'sample_id' in a column
    df_pivot = df_filtered.pivot_table(index='sample_id', columns='gene', values='value')
    df_pivot.reset_index(inplace=True)

    # Merge the pivoted dataframe with the type_df
    df_adapted = df_pivot.merge(type_dataframe[['sample_id', 'sample_status']], on='sample_id', how='left')

    return df_adapted
