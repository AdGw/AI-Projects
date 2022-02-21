import json
import re
import numpy as np
import pickle
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from ttictoc import toc, tic
import os
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter


def evaluate_models():
    NUM_MODELS = 3
    for i in range(1, 1 + NUM_MODELS):
        df = pd.read_csv("models/model_" + str(i) + ".csv")
        col = ['Category', 'Description']
        df = df[col]
        df = df[pd.notnull(df['Description'])]
        df.columns = ['Category', 'Description']
        df = df.sample(frac=1).reset_index(drop=True)
        labels = df['Category']
        text = df['Description']

        # X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=42, test_size=0.25)
        # count_vect = CountVectorizer()
        # X_train_counts = count_vect.fit_transform(X_train)
        # tf_transformer = TfidfTransformer().fit(X_train_counts)
        # X_train_transformed = tf_transformer.transform(X_train_counts)

        X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.3)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        selector = SelectKBest(chi2, k=270000)
        X_train_counts = selector.fit_transform(X_train_counts, y_train)
        tf_transformer = TfidfTransformer().fit(X_train_counts)
        X_train_transformed = tf_transformer.transform(X_train_counts)

        X_test_counts = count_vect.transform(X_test)
        X_test_counts = selector.transform(X_test_counts)
        X_test_transformed = tf_transformer.transform(X_test_counts)

        labels = LabelEncoder()
        y_train_labels_fit = labels.fit(y_train)
        y_train_lables_trf = labels.transform(y_train)
        tic()
        linear_svc_pl = LinearSVC()
        clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
        # calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc_pl, cv="prefit")
        # calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
        predicted = clf.predict(X_test_transformed)
        print('Model LinearSVC: {}'.format(np.mean(predicted == labels.transform(y_test))), round(toc(), 4))

        with open('models_transformer/model_LinearSVC_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/tf_transformer_LinearSVC_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(tf_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/count_vect_LinearSVC_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(count_vect, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/selector_LinearSVC_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(selector, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tic()
        passive_aggressive = PassiveAggressiveClassifier()
        clf = passive_aggressive.fit(X_train_transformed, y_train_lables_trf)
        # calibrated_svc = CalibratedClassifierCV(base_estimator=passive_aggressive, cv="prefit")
        # calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
        predicted = clf.predict(X_test_transformed)
        print('Model PassiveAggressive: {}'.format(np.mean(predicted == labels.transform(y_test))), round(toc(), 4))

        with open('models_transformer/model_PA_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/tf_transformer_PA_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(tf_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/count_vect_PA_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(count_vect, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/selector_PA_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(selector, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tic()
        sgd = SGDClassifier()
        clf = sgd.fit(X_train_transformed, y_train_lables_trf)
        # calibrated_svc = CalibratedClassifierCV(base_estimator=sgd, cv="prefit")
        # calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
        predicted = clf.predict(X_test_transformed)
        print('Model SGDClassifier: {}'.format(np.mean(predicted == labels.transform(y_test))), round(toc(), 4))
        print("\n")

        with open('models_transformer/model_SGD_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/tf_transformer_SGD_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(tf_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/count_vect_SGD_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(count_vect, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/selector_SGD_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(selector, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('models_transformer/labels.pickle', 'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)