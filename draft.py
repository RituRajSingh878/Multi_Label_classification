"""
Starting point for your journey

This code is written in Python 3
(don't attempt to use Python 2 or you'll waste your time with unicode issues)
"""

import zipfile

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


if __name__ == '__main__':
    # Read in the dataset
    with zipfile.ZipFile('data.zip', 'r').open('data.csv') as fp:
        data = pd.read_csv(fp).fillna('')

    print("The columns of this dataset are: %s" % ', '.join(data.columns))
    print("Here is how a few entries look like")
    print(data.sample(n=3))


    # Load the tag information
    with zipfile.ZipFile('data.zip', 'r').open('data.csv') as fp:
        data = pd.read_csv(fp).fillna('')

    tag_data = pd.read_csv('label_info.csv')
    # Create the set of level 2 tags
    l2_tags = set(tag_data[tag_data['level'] == 2]['name'])


    # Get raw text from products
    txt = data[['name', 'description', 'brand']].apply(lambda x: ' '.join(x), axis=1)

    # Get level 2 tags for each product
    y = data['labels'].apply(lambda x: x.split(','))  # split the tags in each row
    y = y.apply(lambda x: set(x) & l2_tags)  # only select the tags at level 2

    # Split the data between training and test set
    txt_train, txt_test, y_train, y_test = train_test_split(txt, y, test_size=0.2,
                                                            random_state=112358)

    # Create a word vectorizer, and train it on the text of the training set
    # Get the vectorized training set text as output
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(txt_train)
    print("\nNumber of text features: %s" % X_train.shape[1])

    # Vectorize the test set text
    X_test = vectorizer.transform(txt_test)

    # One-hot encoder for the product tags
    label_binarizer = MultiLabelBinarizer().fit(y)
    # Convert the training and test set tags
    y_train_bin = label_binarizer.transform(y_train)
    y_test_bin = label_binarizer.transform(y_test)
    print("Number of level 2 tags: %s" % y_train_bin.shape[1])

    # Create a one-versus-rest classifier for out multi-output problem
    model = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)

    print('\nTraining multi-output model...')
    model.fit(X_train.tocoo(), y_train_bin)

    # Measure the accuracy of the model
    accuracy = metrics.accuracy_score(
        y_test_bin,
        model.predict(X_test)
    )
    print("\nThe accuracy of my model is %.1f%%" % (100 * accuracy))

