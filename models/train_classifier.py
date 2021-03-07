import sys
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    #input: sqlite database filepath containing clean dataset
    #Output: X:messages y: array of 36 individual categories columns, categories_names: Names for the 36 categories
    
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///' + database_filepath)
    #read in the clean dataset from sqlite database
    df = pd.read_sql_table('disaster_table', engine)
    #extract the messages
    X = df.message.values
    #extract the array of 36 individual categories columns
    y = np.array(df.iloc[:,4:])
    #extract the names for the 36 categories
    category_names = df.columns
    return X, y, category_names


def tokenize(text):
    #input: messages text
    #Output: tokenized and lemmatized text
    
    #store stop words
    stop_words = stopwords.words("english")
    #Lemmatizer function
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenize the text
    tokens = word_tokenize(text)
    #lemmatize the word tokens if they are not stop words
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word not in stop_words]
    return tokens


def build_model():
    from sklearn.multioutput import MultiOutputClassifier
    #Build a machine learning pipeline
    
    pipeline = Pipeline([
        #Convert a collection of text documents to a matrix of lemmatized token counts
        ('vect', CountVectorizer(tokenizer=tokenize)),
        #Transform a count matrix to a normalized tf or tf-idf representation
        ('tfidf', TfidfTransformer()),
        #train a Random Forest Classifier
        ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=4, min_samples_split=2, n_estimators=40)))
    ])
    
    # these lines are to perform grid search , however we already found the best hyperparameters combination in the notebook, so this is commented out
    """parameters = {
  
        'clf__estimator__n_estimators': [20, 40],
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_depth': [4,6]
    }"""

    #model = GridSearchCV(pipeline, param_grid=parameters)
    #model.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    #input: trained model, test dataset, test labels, names of the 36 categories
    #Output: Print the f1 score, precision and recall for each output category of the dataset
    
    from sklearn.metrics import classification_report
    length_categories = len(category_names)
    #use the trained model to predict on test dataset
    y_pred = model.predict(X_test)
    
    #iterating through the 36 categories columns
    for i in range(length_categories-4):
        #extract the predictions for each column
        y_predict1 = y_pred[:,i].reshape(-1,1)
        #extract the correct labels for each column
        y_test1 = y_test[:,i].reshape(-1,1)
        print(category_names[i+4])
        #Report the f1 score, precision and recall for each output category of the dataset
        print(classification_report(y_test1, y_predict1, target_names=['not 1', '1']))


def save_model(model, model_filepath):
    #input: trained model, model filepath to save the model to
    #Output: model saved to the specified filepath
    
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        #load the clean dataset from SQLite database and extract the messages, array of 36 individual categories columns and name for the 36 categories
        X, Y, category_names = load_data(database_filepath)
        #split the dataset into training and test dataset with test_size set at 0.2
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        #build the ML model
        print('Building model...')
        model = build_model()
        #fit the model on training dataset
        print('Training model...')
        model.fit(X_train, Y_train)
        #evaluate the model by reporting the f1 score, precision and recall for each output category of the dataset
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        #save the trained model to the specified filepath
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