# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

Description

This project is about analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

This project main sections are:
1) building ETL pipeline that loads the data, cleans it and save it to SQLite database
2) building Machine Learning pipeline that trains a classifier to accept message as input and output classification results on the 36 categories in the dataset
3) running a a web app where an emergency worker can input a new message in real time and get classification results in several categories 

Dependencies
This repository is written in Python and the following Python packages are required:
NumPy, SciPy, Pandas, Sciki-Learn, NLTK, SQLalchemy, Pickle, Flask, Plotly

Executing Program:

You can run the following commands in the project's directory to set up the database, train model and save the model.
To run ETL pipeline to clean data and store the processed data in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/

File Descriptions:

ETL Pipeline Preparation.ipynb: notebook containing codes to extract dataset, cleans it and save to SQLite database
ML Pipeline Preparation.ipynb: notebook containing codes to extract dataset from SQLite database, trains a classifier , performs grid search and print out classification report                                for predictions on test set 
process_data.py: a python script that loads the messages and categories dataset, merge it, cleans it and stores in a SQLite database
train_classifier.py: a python script that loads data from SQLite database, split it to train and test set, trains the classifier and print out classification report for                              predictions made on test set and save the trained model to a pickle file for later use
data: contains messages and categories datasets in csv format.
app: contains the run.py which runs the web application.


Screenshots:

Running process_data.py
![cleans data and stores in sqlite database (1)](https://user-images.githubusercontent.com/73007150/110245969-99659b80-7fa0-11eb-9cea-f9a4c891e009.PNG)

Running train_classifier.py
![trains the classifier and saves](https://user-images.githubusercontent.com/73007150/110245993-b7330080-7fa0-11eb-9582-7936bc96e7ab.PNG)

Classification report for predictions on test dataset
![classification report for predictions on test dataset](https://user-images.githubusercontent.com/73007150/110246061-fb260580-7fa0-11eb-8107-42a450198468.PNG)

Running web app
![running the web app](https://user-images.githubusercontent.com/73007150/110246430-86ec6180-7fa2-11eb-8890-d4f2e4a85662.PNG)

Web app main page
![disaster response need medical assistance](https://user-images.githubusercontent.com/73007150/110246573-17c33d00-7fa3-11eb-90a3-915b5f10a11f.PNG)
![disaster response distribution of cat](https://user-images.githubusercontent.com/73007150/110248336-768cb480-7fab-11eb-8f3b-7e16e76876fc.PNG)

Output when inputting message of 'need medical assistance'
![disaster response need medical assistance](https://user-images.githubusercontent.com/73007150/110248353-92905600-7fab-11eb-9925-cf5c57d61a89.PNG)



