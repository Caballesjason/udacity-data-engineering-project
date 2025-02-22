import sys
import pandas as pd
import re


def load_data(messages_filepath, categories_filepath):
    """load_data loads the messages and categories data, executes an inner join by their id attributes and removes the original column

    args:
        messages_filepath (str):  The disaster_messages.csv file path
        categories_filepath (str):  The disaster_categories.csv file path

    returns:
        the disaster_messages.csv and disaster_categories.csv joined by id
    """
    # import and define data types for categories
    categories_dtypes = {'id':  'int', 'categories': 'string'}
    categories = pd.read_csv("data/disaster_categories.csv", dtype=categories_dtypes)

    # import and define data types for messages and remove original field
    messages_dtypes = {'id':  'int', 'message': 'string', 'genre': 'string'}
    messages_usecols = ['id', 'message', 'genre']
    messages = pd.read_csv('data/disaster_messages.csv', dtype=messages_dtypes, usecols=messages_usecols)

    # inner join the dataframes
    df = categories.merge(messages, how='inner', on='id')

    # remove duplicates
    df = df.drop_duplicates()

    # reset index
    df.reset_index()
    
    return df


def clean_data(df):
    """clean_data cleans the joined csv data to create columns for the categories
    
        args:
            df (Pandas DataFrame):  The dataframe to be cleaned

        returns: The cleaned dataframe
    """
    # get categories column as series
    categories = df['categories']

    # drop categories column from original dataframe
    df = df.drop('categories', axis=1)

    # split categories by to get column categories
    split_categories = categories.str.split(";", expand=True)

    # get example string from categories series
    example_category_string = categories[0]

    # remove - and digits from categories string
    clean_titles = re.sub(r"(-\d)|", "", example_category_string)

    # get list of categories
    clean_titles = clean_titles.split(";")

    # set the category column titles
    split_categories = split_categories.set_axis(clean_titles, axis=1)

    # Convert all values to their integers
    get_ints = lambda val: int(val[-1])
    split_categories = split_categories.map(get_ints)

    # merge the original dataframe with the categories dataframe by index
    df = df.merge(split_categories, left_index=True, right_index=True)

    return df


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()