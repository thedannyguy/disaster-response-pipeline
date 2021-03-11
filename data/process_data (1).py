import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

def load_data(messages_filepath, categories_filepath):
    # Input: messages and categories dataset
    # Output: merged messages and categories dataset
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge messages and categories dataset
    df = messages.merge(categories, on=('id'))
    return df
    


def clean_data(df):
    #Input: merged messages and categories dataset
    #Output: Dataframe of messages and 36 individual category columns
    
    #Split the values in the categories column on the ; character so that each value becomes a separate column
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df['categories']).str.split(pat=';', n=-1, expand=True)
    # select the first row of the categories dataframe
    row = np.array(categories[0:1])
    #extract a list of new column names for categories.
    category_colnames = []
    for i in range(36):
        colnames = row[0,i][:-2]
        category_colnames.append(colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        #convert the Series to be of type string
        categories[column] = pd.Series(categories[column]).astype(str)
        #keep only the last character of each string (the 1 or 0)
        categories[column] = pd.Series(categories[column]).str[-1]
        #Convert the string (the '1' or '0') to a numeric value
        categories[column] = pd.to_numeric(categories[column]) 
    
    # drop the original categories column from `df`
    df.drop(['categories'], inplace=True, axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # check number of duplicates before dropping
    print('Number of duplicates: {}'.format(sum(df.duplicated())))
    # drop duplicate messages
    df.drop_duplicates(subset="message", keep='first', inplace=True)
    # check number of duplicates after dropping
    print('Number of duplicates: {}'.format(sum(df.duplicated())))    
    #check that variables only contain binary values
    np.count_nonzero(df==2, axis=0)
    #drop rows containing value 2 in related column
    df = df[df['related'] != 2]
    return df
    
def save_data(df, database_filepath):
    #input:  Dataframe of messages and 36 individual category columns
    #Output:  Save the clean dataset into an sqlite database.
    from sqlalchemy import create_engine
    
    engine = create_engine('sqlite:///' + database_filepath)
    #name the clean dataset, disaster_table
    table_name = 'disaster_table'
    #save clean dataset to sqlite database
    df.to_sql(table_name, engine, if_exists = 'replace', index=False) 


def main():
    if len(sys.argv) == 4:
        
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        #load messages and categories dataset and merge them
        df = load_data(messages_filepath, categories_filepath)
        #Create a dataframe of messages and 36 individual category columns
        print('Cleaning data...')
        df = clean_data(df)
        #Save the clean dataset into an sqlite database.
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


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    main()