'''
Claire Luo, Samantha Shimogawa, Benjamin Zhuang
CSE 163 Section AD

This program tests the correctness of the world
happiness analysis processes.
'''
from pandas.testing import assert_frame_equal
from cse163_utils import assert_equals
import processing


def test_add_year(dataframes):
    # check that length of columns has been increased by 1
    dataframes = processing.add_year(dataframes)
    assert_equals([13, 14, 13, 10, 10], [len(df.columns) for df in dataframes])

    # check that the year for each dataframe is correct
    for i in range(5):
        assert_equals(2015+i, dataframes[i]['Year'][0])
    
    # return modified list of dataframes so testing can continue
    return dataframes


def test_column_filter(dataframes, columns):
    dataframes = processing.column_filter(dataframes, columns)

    # confirm dataframes have the correct number of columns after filtering
    assert_equals([10, 10, 10, 10, 10], [len(df.columns) for df in dataframes])

    # return modified list of dataframes so testing can continue
    return dataframes


def test_rename_col(dataframes, columns):
    dataframes = processing.rename_cols(dataframes, columns)
    # confirm dataframes have the correct number of columns
    assert_equals([10, 10, 10, 10, 10], [len(df.columns) for df in dataframes])

    # confirm dataframes have correctly named columns
    for i in range(2):
        assert_equals(['country',
            'happiness rank',
            'happiness score',
            'GDP per capita',
            'family',
            'life expectancy',
            'freedom',
            'government corruption',
            'generosity',
            'year'], dataframes[i].columns.tolist())
    
    for i in range(2, 5):
        assert_equals(['country',
            'happiness rank',
            'happiness score',
            'GDP per capita',
            'family',
            'life expectancy',
            'freedom',
            'generosity',
            'government corruption',
            'year'], dataframes[i].columns.tolist())

    # return modified list of dataframes so testing can continue
    return dataframes


def test_concatenate_dataframes(dataframes):
    df = processing.concatenate_dataframes(dataframes)

    # check that the shape is correct
    assert_equals((782, 10), df.shape)

    # confirm that columns of new dataframe are correct
    assert_equals(['country',
            'happiness rank',
            'happiness score',
            'GDP per capita',
            'family',
            'life expectancy',
            'freedom',
            'government corruption',
            'generosity',
            'year'], df.columns.tolist())

    # return modified dataframe so testing can continue
    return df


def test_add_region(df):
    df = processing.add_region(df)

    # check that the dataframe has the correct number of columns
    assert_equals(11, len(df.columns))

    # check that column names are correct
    assert_equals(['country',
        'happiness rank',
        'happiness score',
        'GDP per capita',
        'family',
        'life expectancy',
        'freedom',
        'government corruption',
        'generosity',
        'year',
        'region'], df.columns.tolist())

    # make sure first entry is a region, not empty (and correct)
    assert_equals('Western Europe', df['region'][0])


def main():
    # read data from my github repository
    base_url = 'https://raw.githubusercontent.com/'\
            + 'Benjaminnnnnn/CSE-163-FINAL-PROJECT/master/data'
    # data to be read
    files = [base_url + '/' + str(i) + '.csv' for i in range(2015, 2020)]

    # read data
    dataframes = processing.read_data(base_url, files)

    dataframes = test_add_year(dataframes)
    # create the columns to be extracted for data analysis
    # store the regex pattern of column and its full name in
    # a pair of tuple, [0] for pattern, [1] for full name
    columns = list(map(lambda x: ('(?i).*' + x[0] + '(?i).*', x[1]),
                   [('rank', 'happiness rank'),
                    ('country', 'country'),
                    ('score', 'happiness score'),
                    ('GDP', 'GDP per capita'),
                    ('family', 'family'),
                    ('social', 'family'),
                    ('freedom', 'freedom'),
                    ('health', 'life expectancy'),
                    ('corruption', 'government corruption'),
                    ('trust', 'government corruption'),  # same as "corruption"
                    ('generosity', 'generosity'),
                    ('year', 'year')
                    ])
                   )
    dataframes = test_column_filter(dataframes, columns)
    dataframes = test_rename_col(dataframes, columns)
    df = test_concatenate_dataframes(dataframes)
    test_add_region(df)


if __name__ == '__main__':
    main()
