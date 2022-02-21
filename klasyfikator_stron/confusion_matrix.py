import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import os
from sklearn.tree import ExtraTreeClassifier

# def confusion():
p = os.listdir('Kategorie_cleaned/')
NUM_MODELS = 3
# p = os.listdir("models/")

for i in range(1, 1 + NUM_MODELS):
    df = pd.read_csv('models/model'+str(i)+'.csv')
    col = ['Category', 'Description']
    df = df[col]
    df = df[pd.notnull(df['Description'])]
    df.columns = ['Category', 'Description']
    df = df.sample(frac=1).reset_index(drop=True)
    labels = df['Category']
    text = df['Description']

    X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.25)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_transformed = tf_transformer.transform(X_train_counts)

    X_test_counts = count_vect.transform(X_test)
    X_test_transformed = tf_transformer.transform(X_test_counts)

    labels = LabelEncoder()
    y_train_labels_fit = labels.fit(y_train)
    y_train_lables_trf = labels.transform(y_train)

    y_test_labels_fit = labels.fit(y_test)
    y_test_lables_trf = labels.transform(y_test)
    # print(y_test_lables_trf)

    classifier = LinearSVC(random_state=42).fit(X_train_transformed, y_train_lables_trf)
    predicted = classifier.predict(X_test_transformed)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [
        # ("Confusion matrix, without normalization", None),
            ("Normalized confusion matrix", 'true')
    ]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test_transformed, y_test_lables_trf,
                                     display_labels=p,
                                     cmap=plt.cm.Blues,
                                     xticks_rotation='vertical',
                                     normalize=normalize
                                     )
        disp.ax_.set_title(title)

        # print(title)
        print(disp.confusion_matrix)

    # plt.savefig(str(i)+'.png')