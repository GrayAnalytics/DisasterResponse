"""
Queries a SQLite database, and then builds a model to classify if a message is related to a disaster, 
and under which categoryies if should be assigned
"""

import sys
# import libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

import re

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle 

def load_data(database_filepath):
    '''
    Queries a SQLite server database for the disaster dataset,
    Separates into independent and dependent variables
    Also provides list of column names

    Parameters
    ----------
    database_filepath : string
        path and name of the database.

    Returns
    -------
    X : vector
        List of messages that are used to make the prediction
    Y : dataframe
        Dataframe of binary variables, these are the dependent variables that are predicted.
    column_list : List
        List of column names for the Y dataframe

    '''
    #X, Y, category_names = load_data(database_filepath)
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM Message", engine)
    X = df.message.values
    column_list = df.columns 
    column_list = column_list.drop(['id', 'message', 'original', 'genre'])
    Y = df[column_list]
    return X, Y, column_list

    


def tokenize(text):
    '''
    Processes the message. 
    Tokenizes it, removes stop words, converts to stems

    Parameters
    ----------
    text : string
        message that may be related to a disaster.

    Returns
    -------
    tokens : list
        converted list of words making up the input message. Processed for NLP.

    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize
    tokens = word_tokenize(text)

    #remove stop words
    tokens = [x for x in tokens if x not in stopwords.words("english")]

    #change words to their stems

    tokens = [PorterStemmer().stem(x) for x in tokens]

    return tokens


def build_model():
    '''
    Creates a pipeline for processing a message and producing a predictive model

    Returns
    -------
    cv : pipeline model
        cross validated model.

    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    parameters = {
        
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [ 4, 8]
        }
    #parameters = {
    #    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
    #    'clf__estimator__n_estimators': [5, 10, 20],
    #    'clf__estimator__min_samples_split': [2, 3, 4]
    #    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

    
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Takes in a model and runs against test set 
    Prints out the evaluation results for each possible category

    Parameters
    ----------
    model : pipeline model
        cross validated model.
    X_test : vector
        List of strings, of messages.
    Y_test : dataframe
        Dataframe of binary variables, these are the dependent variables that are predicted.
    category_names : List
        List of column names for the Y dataframe

    Returns
    -------
    None.

    '''
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    for column_name in y_pred_df.columns:
        print(column_name)
        print(classification_report(Y_test[column_name], y_pred_df[column_name]))

def save_model(model, model_filepath):
    '''
    Takes a model and saves it as a pickle file

    Parameters
    ----------
    model : pipeline model
        cross validated model.
    model_filepath : string
        path and name of the pickle file we'll write the model to.

    Returns
    -------
    None.

    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file=file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)
        
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