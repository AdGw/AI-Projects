from ttictoc import tic, toc
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, Perceptron, RidgeClassifierCV, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

tic()
df = pd.read_csv("model_1.csv")
col = ['Category', 'Description']
df = df[col]
df = df[pd.notnull(df['Description'])]
df.columns = ['Category', 'Description']
df = df.sample(frac=1).reset_index(drop=True)
labels = df['Category']
text = df['Description']
print("csv loaded - ", toc())
tic()
X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.3)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_transformed = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_transformed = tf_transformer.transform(X_test_counts)

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)


# linear_svc_pl = MultinomialNB()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('MultinomialNB - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = ComplementNB()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('ComplementNB - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = BernoulliNB()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('BernoulliNB - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = LinearSVC()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('LinearSVC - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = Perceptron()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('Perceptron - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = SGDClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('SGD - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = KNeighborsClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('KNN - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = RandomForestClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('RandomForest - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = DecisionTreeClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('DecisionTree - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = ExtraTreeClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('ExtraTree - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = PassiveAggressiveClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('PassiveAggressiveClassifier - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = RidgeClassifier()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('RidgeClassifier - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))

# linear_svc_pl = RidgeClassifierCV()
# clf = linear_svc_pl.fit(X_train_transformed, y_train_lables_trf)
# # clf.fit(X_train_transformed, y_train_lables_trf)
# calibrated_svc = CalibratedClassifierCV(base_estimator=clf, cv="prefit")
# calibrated_svc.fit(X_train_transformed, y_train_lables_trf)
# predicted = calibrated_svc.predict(X_test_transformed)
# print('RidgeClassifierCV - dopasowanie dla danych na poziomie={}'.format(np.mean(predicted == labels.transform(
#     y_test))))


names = [
		 'MultinomialNB',
		 'ComplementNB',
		 'BernoulliNB',
         'Linear SVC',
         'Perceptron',
         'SGD',
         'KNN',
         'Random Forest',
         'DecisionTree',
         'ExtraTree',
         'Passive agressive',
         'Ridge',
         'RidgeClassifierCV']

classifiers = [
			   MultinomialNB(),
			   ComplementNB(),
			   BernoulliNB(),
               LinearSVC(),
               Perceptron(),
               SGDClassifier(),
               KNeighborsClassifier(),
               RandomForestClassifier(),
               DecisionTreeClassifier(),
               ExtraTreeClassifier(),
               PassiveAggressiveClassifier(),
               RidgeClassifier(),
               RidgeClassifierCV()]

parameters = [
              {
				   'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5],
	               'fit_prior': [True, False],
	               'class_prior': [(0.5, 0.5), (0.63, 0.34), (0.7, 0.3), (0.75, 0.25), (0.87, 0.13), (0.9, 0.1)]
              },
			  {
				   'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5],
	               'fit_prior': [True, False],
	               'class_prior': [(0.5, 0.5), (0.63, 0.34), (0.7, 0.3), (0.75, 0.25), (0.87, 0.13), (0.9, 0.1)],
	               'norm': [True, False]
              },
              {
				   'alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5],
				   'binarize': [0.01, 0.1, 0.5, 1],
	               'fit_prior': [True, False],
	               'class_prior': [(0.5, 0.5), (0.63, 0.34), (0.7, 0.3), (0.75, 0.25), (0.87, 0.13), (0.9, 0.1)]
              },
              {
	               'penalty': ['l1', 'l2'],
	               'loss': ['hinge', 'squared_hinge'],
	               'dual': [True, False],
	               'tol': [1e-5, 1e-4, 1e-3, 1e-2],
	               'C': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
	               'multi_class': ['ovr', 'crammer_singer'],
	               'intercept_scaling': [0.01, 0.1, 0.25, 0.5],
	               'fit_intercept': [True, False],
	               'max_iter': [100000]
               },
               {
	               'penalty': ['l1', 'l2', 'elasticnet'],
	               'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5],
	               'l1_ratio': [0.1, 0.25, 0.5, 0.75, 1],
	               'fit_intercept': [True, False],
	               'max_iter': [100000],
	               'tol': [1e-5, 1e-4, 1e-3, 1e-2],
	               'warm_start': [True, False]
               },
               {
	               'penalty': ['l1', 'l2', 'elasticnet'],
	               'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
	               'l1_ratio': [0.1, 0.25, 0.5, 0.75, 1],
	               'fit_intercept': [True, False],
	               'max_iter': [100000],
	               'tol': [1e-5, 1e-4, 1e-3, 1e-2],
	               'epsilon': [0.1, 0.25, 0.5, 0.75, 1],
	               'warm_start': [True, False]
               },
               {
	               'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
	               'weights': ['uniform', 'distance'],
	               'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	               'leaf_size': [5, 10, 20, 30, 40, 50, 75, 100],
	               'p': [0.5, 1.0, 2.0, 3.0]
               },
               {
	               'criterion': ['gini', 'entropy'],
	               'max_depth': [3, 5, 25, 50, 100],
	               'min_samples_split': [2, 3],
	               'min_samples_leaf': [2, 3],
	               'min_weight_fraction_leaf': [0, 0.5],
	               'max_features': ['auto', 'sqrt', 'log2'],
	               'max_leaf_nodes': [2, 3],
	               'warm_start': [True, False]
                },
                {
	               'criterion': ['gini', 'entropy'],
	               'splitter': ['best', 'random'],
	               'max_depth': [3, 5, 25, 50, 100],
	               'min_samples_split': [2, 3],
	               'min_samples_leaf': [2, 3],
	               'min_weight_fraction_leaf': [0, 0.5],
	               'max_features': ['auto', 'sqrt', 'log2'],
	               'max_leaf_nodes': [2, 3],
	               'ccp_alpha': [0, 1]
                },
                {
	               'criterion': ['gini', 'entropy'],
	               'splitter': ['best', 'random'],
	               'max_depth': [3, 5, 25, 50, 100],
	               'min_samples_split': [2, 3, 5],
	               'min_samples_leaf': [2, 3, 5],
	               'min_weight_fraction_leaf': [0, 0.5],
	               'max_features': ['auto', 'sqrt', 'log2'],
	               'max_leaf_nodes': [2, 3, 5],
	               'min_impurity_decrease': [2, 3, 5],
	               'ccp_alpha': [0, 1]
                },
                {
	               'C': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 2, 5],
	               'fit_intercept': [True, False],
	               'max_iter': [10000],
	               'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
	               'early_stopping': [True, False],
	               'validation_fraction': [0.01, 0.1, 0.5],
	               'loss': ['hinge', 'squared_hinge'],
	               'warm_start': [True, False],
	               'average':[True, False]
                },
                {
	               'alpha': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5],
	               'fit_intercept': [False],
	               'normalize': [True, False],
	               'max_iter': [10000],
	               'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
	               'solver': ['auto', 'lsqr', 'sparse_cg', 'sag', 'saga']
                },
                {
	               'fit_intercept': [True, False]
                }]

for i in range(len(classifiers)):
    print(f'--- {names[i].upper()} ---')
    gs_clf = GridSearchCV(classifiers[i], parameters[i], n_jobs=-1)
    gs_clf.fit(X_train_transformed, y_train)

    print(f'Score: {gs_clf.best_score_}\nParameters: {gs_clf.best_params_}')