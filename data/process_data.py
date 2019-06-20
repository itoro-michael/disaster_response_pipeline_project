import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ This function loads the csv files provided by the function
        arguments and returns them as dataframes.

    Args:
        messages_filepath (str): The filepath to csv file containing messages.
        categories_filepath (str): The filepath to csv file containing message labels.

    Returns:
        DataFrame: containing messages.
        DataFrame: containing labels.

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    return messages, categories


def clean_data(messages, categories):
    """ This function merges the received dataframes and cleans the resulting dataset to be 
        training ready. It returns the cleaned dataset.

    Args:
        messages (DataFrame): The dataframe containing messages.
        categories (DataFrame): The dataframe containing message labels.

    Returns:
        DataFrame: The cleaned dataset.

    """
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = list(category_colnames)
    
    # upgrade column value
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from df
    df.drop(['categories'], inplace=True, axis=1)
    
    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop select rows with nulls
    col_list = list(df.columns)
    col_list.remove('original')
    df.dropna(axis=0, subset=col_list, inplace=True)
    
    # convert the df.related column to binary
    df.related[df.related == 2.] = 1.0
    
    return df


def save_data(df, database_filename):
    clean_data(messages, categories):
    """ This function saves the cleaned dataset.

    Args:
        df (DataFrame): The cleaned dataframe containing messages and labels.
        database_filename (str): The database filename to save dataset.

    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('tbl_message', engine, index=False, if_exists='replace')


def main():
    """ The main program for module.

    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('database filepath:', database_filepath)

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_msg, df_cat = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df_msg, df_cat)
        
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