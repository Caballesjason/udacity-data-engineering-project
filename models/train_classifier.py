import sys
# import data.process_data as processer
from sqlalchemy.sql import text
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle


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
    X = df['message'] # create feature set
    Y = df.drop(labels=['id', 'message', 'genre', 'index'], axis=1, inplace=False) # create target set
    category_names = list(Y.columns) # get all category names

    return X, Y, category_names

def tokenize(text):
    """tokenize will take a string and do the following transformations
    1. lowercase all characters
    2. remove punctuation, digits and uneeded whitespace
    3. tokenize the data
    4. remove stop words
    5. apply lemmatization using NLTK
    6. convert back to string
    
    args:
    text (string): The text to tokenize

    returns:
    the clean string ready to be tokenized in TFIDFVectorizer
    """
    clean_text = text.lower() # make text lowercase
    clean_text = re.sub(r"[^a-zA-Z]", " ", clean_text) # remove punctuation, digits and whitespace
    tokenized = word_tokenize(clean_text) # tokenize words
    # remove stop words
    stop_words = set(stopwords.words('english'))
    tokenized = [word for word in tokenized if word not in stop_words]
    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokenized = [lemmatizer.lemmatize(word) for word in tokenized]
    # return cleaned text for TFIDF tokenization
    clean_string = " ".join(tokenized)
    return tokenized



def build_model():
    # Define transformer for tokenizer
    def tokenizer(X):
        return X.apply(tokenize)
        
    TokenizerTransformer = FunctionTransformer(func=tokenizer)
    pipe = [
        # ('tokenizer', TokenizerTransformer),
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ]
    pipeline = Pipeline(pipe)

    params = {
        "clf__estimator__n_estimators": [50, 100, 150],
        "clf__estimator__min_samples_split": [2, 3, 4],
        # "clf__estimator__max_depth": [2]

    }

    # params = {
    #     "tfidf__max_df": [0.9],
    #     "tfidf__min_df": [0.05],
    #     "clf__estimator__n_estimators": [20],

    # }

    cv = GridSearchCV(pipeline, param_grid=params, error_score='raise')
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    predictions = model.predict(X_test)
    for index, category in enumerate(category_names):
        # print(type(predictions), type(Y_test))
        
        precision = precision_score(predictions[:, index], Y_test.iloc[: ,index], average='macro')
        precision = round(precision, 4)
        recall = recall_score(predictions[:, index], Y_test.iloc[: ,index], average='macro')
        recall = round(recall, 4)
        f_one = f1_score(predictions[:, index], Y_test.iloc[: ,index], average='macro')
        f_one = round(f_one, 4)
        print("----  " + category  + " ----")
        print("Precision: {}\tRecall: {}\tF1: {}\n".format(precision, recall, f_one))
    return None



def save_model(model, model_filepath):
    with open("classifier.pkl", "wb") as file:
        pickle.dump(model, file)
    return None

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