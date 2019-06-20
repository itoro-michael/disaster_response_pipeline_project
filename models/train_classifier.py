import sys
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from numpy import unravel_index
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier


def load_data(database_filepath):
    """ This function loads the database file given as function argument and returns
        the independent variable X, the dependent variable y, and the list of categories
         as col_list.

    Args:
        database_filepath (str): The filepath to database file containing messages and labels.

    Returns:
        numpy.array: The messages.
        numpy.array: The labels.
        List: The category names

    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM tbl_message", engine)
    
    # Define X and y
    X = df['message'].values
    y = df.iloc[:, 4:].values
    
    # Get category names
    col_list = list(df.columns)[4:]
    
    return X, y, col_list

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    """ The tokenize function takes a string message and returns a list of tokens.

    Args:
        text (str): The string message.

    Returns:
        List: of string tokens

    """
     # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "<url>")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initialise lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        lemmatizer.lemmatize(tok)
        clean_tok = tok.lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ This function defines the pipeline used for training and testing.

    Returns:
        Pipeline:

    """
    # build pipeline
    pipeline = Pipeline\
                ([
                    ('vect', CountVectorizer(tokenizer=tokenize, max_features=5000)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=0), n_jobs=1))
                ])
    
    parameters =    {
                        'clf__estimator__n_estimators': [50, 60]
                    }
    # Make an fbeta_score scoring object
    scorer = make_scorer(fbeta_score, beta=1.0, average='macro')

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ This function evaluates the trained classifier.

    Args:
        model (Pipeline): Pipeline object for evaluation.
        X_test (numpy.array): The cleaned messages.
        Y_test (numpy.array): The labels.
        category_names (List): The label names.

    """
    # predict on test data
    y_test_pred = model.predict(X_test)
    
    # Obtain accuracy
    accuracy_mask = (y_test_pred == Y_test)
    accuracy = accuracy_mask.mean()
    print("Accuracy:", round(accuracy,2))
    
    # print f1 score, precision and recall
    for i in range(Y_test.shape[1]):
        report = classification_report(Y_test[:,i], y_test_pred[:,i])
        print(str(i+1)+'.', category_names[i]+" label\n", report)


def save_model(model, model_filepath):
    """ This function saves the trained model.

    Args:
        model (Pipeline): The classifier pipeline object
        model_filepath (str): The filename for saving classifier.

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ The main module code.

    """
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