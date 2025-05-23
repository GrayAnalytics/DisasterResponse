"""
This program reads in message and category csv files, joins and cleans the data, the outputs to a sqlite database
"""

import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reads in data from csvs

    Parameters
    ----------
    messages_filepath : string
        Path to messages csv file.
    categories_filepath : string
        path to categories csv file.

    Returns
    -------
    df : dataframe
        merged dataframe of the messages and categories files.

    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how = 'left', on = 'id') 
    #print(df.head())
    return df


def clean_data(df):
    '''
    cleans up the column names
    removes the column name from individual cell's data values
    removes undesired columns

    Parameters
    ----------
    df : dataframe
        dataframe that is returned by load_data().

    Returns
    -------
    df : dataframe
        modified version of the input dataframe.

    '''
    categories = df['categories'].str.split(';', expand=True) 
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    category_colnames = [iteration[0:iteration.rfind('-')] for iteration in row]
    categories.columns = category_colnames
    for column in categories:
        #print(column)
        # set each value to be the last character of the string
        categories[column] = [x[x.find('-')+1:] for x in categories[column]]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 
    # drop the original categories column from `df`
    df = df[['id', 'message', 'original', 'genre']]
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df



def save_data(df, database_filename):
    '''
    Saves the resulting dataframe to a sqlite database

    Parameters
    ----------
    df : dataframe
        dataframe that has been passed through the program.
    database_filename : string
        path and name of the database we will write this to.

    Returns
    -------
    None.

    '''
    engine = create_engine('sqlite:///'+ database_filename)  
    df.to_sql('message', engine, index=False)


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        #print(df.head())
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