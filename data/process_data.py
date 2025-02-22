import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """load_data loads the messages and categories data, executes an inner join by their id attributes and removes the original column

    args:
        messages_filepath (str):  The disaster_messages.csv file path
        categories_filepath (str):  The disaster_categories.csv file path

    returns:
        the disaster_messages.csv and disaster_categories.csv joined by id
    """
    # import and define data types for categories
    categories_dtypes = {'id':  'Int64', 'categories': 'string'}
    categories = pd.read_csv("data/disaster_categories.csv", dtype=categories_dtypes)

    # import and define data types for messages and remove original field
    messages_dtypes = {'id':  'Int64', 'message': 'string', 'genre': 'string'}
    messages_usecols = ['id', 'message', 'genre']
    messages = pd.read_csv('data/disaster_messages.csv', dtype=messages_dtypes, usecols=messages_usecols)

    # inner join the dataframes
    df = categories.merge(messages, how='inner', on='id')

    # remove duplicates
    df.drop_duplicates(inplace=True)

    # reset index
    df.reset_index()
    
    return df




def clean_data(df):
    pass


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