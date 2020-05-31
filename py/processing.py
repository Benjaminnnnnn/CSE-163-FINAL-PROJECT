'''
This program cleans and processes data needed for
world happiness analysis
'''
import re
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


def read_data(base_url, files):
    '''
    read the data from the given url and files,
    return a list of dataframes
    '''
    return [pd.read_csv(file) for file in files]


def add_year(dataframes):
    '''
    add year columns to each dataframe with the given
    list of dataframes
    '''
    year = 2015
    for df in dataframes:
        df['Year'] = year
        year += 1
    return dataframes


def column_filter(dataframes, columns):
    '''
    takes the columns to be extracted from the original
    list of dataframes, return a list of dataframes with
    filtered columns
    '''
    column_filter = re.compile('|'.join([col_name[0] for col_name in columns]))

    for i, df in enumerate(dataframes):
        all_matches = list(filter(column_filter.match, df.columns))
        dataframes[i] = df[all_matches]
    return dataframes


def rename_cols(dataframes, sub_patterns):
    '''
    Rename the columns of the given dataframes to
    the sub_pattern defined.

    Parameters:
        - dataframes        - a list of dataframes to be renamed
        - sub_patterns      - a list of tuples consists matching patterns
                              on the first entry, and the replacements on
                              the second entry

    Returns:
        - renamed_dfs       - renamed dataframes
    '''
    # col  - column
    # pat  - pattern
    # repl - replacement
    for i, df in enumerate(dataframes):
        new_cols = {}
        for col in df.columns:
            for pat, repl in sub_patterns:
                if re.match(pat, col):
                    new_cols[col] = re.sub(pat, repl, col)
        dataframes[i] = df.rename(columns=new_cols)
    return dataframes


def concatenate_dataframes(dataframes):
    '''
    takes a list of dataframes, concatenate
    the dataframes into one big dataframe,
    return the new concatenated datafram
    '''
    return pd.concat(dataframes, ignore_index=True)
