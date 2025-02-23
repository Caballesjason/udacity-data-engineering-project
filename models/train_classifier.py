import sys
import data.process_data as processer
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd

def load_data(database_filepath):
    """load_data returns the features, targets, and a list of all categories

    args:
        database_file_path (str):  The file path to a database

    returns:
        X (Pandas DataFrame):  The feature set
        Y (Pandas Series):  The target series
        category_names (list):  The list of unique category names
    """    

    path = 'sqlite+pysqlite:///' + database_filepath # add dbapi and dialect declaration
    print(path)
    engine = create_engine(url=path, echo=False) # database engine
    conn = engine.connect() # database connection
    query_string = "SELECT * FROM messages;" # database connection
    df = pd.read_sql(sql=query_string, con=conn) # returned data from database
    X = df[['id', 'message', 'genre']] # create feature set
    Y = df.drop(labels=['id', 'message', 'genre', 'index'], axis=1, inplace=False) # create target set
    category_names = list(Y.columns) # get all category names

    return X, Y, category_names

def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()