# DisasterResponse
Seattle Air BnB analysis for Udacity  

## Project Overview  
This project is part of the Udacity DataScientist NanoDegree.  
In this project we receive message data from disasters, along with how they were categorized.
 * We will read in the data from csv files, clean it and write to a SQLite database
 * We will build a model to predict a messages categories and save this model to a pickle file
 * Using the code provided by Udacity, we create a dash dashboard to graph out the data and lets users input new messages to classify.

## Data Overview  
The data was sourced from [Appen](https://www.figure-eight.com/)  
The data is two csv files:
 * messages
 * categories 


## Libraries Used 
For this project, we used the libraries:  
 * sqlite3
 * pandas  
 * sqlalchemy  
 * re  
 * nltk  
 * sklearn
 * pickle
 * json
 * plotly
 * flask




## Process  
1) Begin by running process_data.py followed by the path to the message csv and the path to the classifier csv and the path to the sqlite database
bash ex: python process_data.py messages.cav categories.csv disaster.db
2) Next run train_classifier.py followed by the path to the sqlite database and the path to the classifier model pickle
bash ex: python train_classifier.py DisasterResponse.db classifier.pkl
3) Finally, to run the dash dashboard, run the run.py file
bash ex: python run.py

&nbsp;&nbsp;&nbsp;&nbsp;


## Acknowledgements

Some of the python code was reused from the exercises found in Udacity's Data Science Nanodegree.
For example, the naming schema for the training data 

> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3)


