import sys

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    """load data from database

    Args:
        database_filepath (str): path to the file of database

    Returns:
        X: features
        Y: labels
        category_name: a list of names of category
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("labeled_disaster_message", engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = df.columns.drop(['id', 'message', 'original', 'genre']).tolist()
    return X, Y, category_names


def tokenize(text):
    """normalize (lower the case) and tokenize the text, then remove space(s)

    Args:
        text (str): input string

    Returns:
        [str]: a list of tokens
    """
    tokens = word_tokenize(text)
    
    lemmantizer = WordNetLemmatizer()
    return [lemmantizer.lemmatize(token.lower().strip()) for token in tokens]

def build_model():
    """build ML pipeline, which includes CoutVectorizer and TfidfTransformer for feature extraction, then use MultiOutputClassifier for classification

    Returns:
        [type]: [description]
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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