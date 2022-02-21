import numpy as np
from scipy.sparse import load_npz
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from joblib import parallel_backend
from scipy.sparse import save_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from ttictoc import tic, toc

import os
import glob

def evaluate_joblib_models():
    [os.remove(i) for i in glob.glob('models_joblib/*')]
    NUM_MODELS = 3
    for i in range(1, 1 + NUM_MODELS):
        X = load_npz('models/X_'+str(i)+'.npz')
        Y = np.load('models/Y_'+str(i)+'.npz')['arr_0']
        features = [
            # 5000,
            270000,
            # X.shape[1]
        ]
        for j in features:
            print("features:", j)
            print("\n")

            model = SelectKBest(chi2, k=j)
            model.fit(X, Y)
            X_new = model.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_new, Y, test_size=0.3, random_state=0)

            tic()
            clf = LinearSVC().fit(X_train, y_train)
            # calibrated = CalibratedClassifierCV(clf)
            # calibrated.fit(X_train, y_train)
            s = clf.score(X_test, y_test)
            print("LinearSVC: \t\t {}".format(round(s, 4)) + " time:", round(toc(), 4))
            dump(clf, 'models_joblib/LinearSVC_'+str(i)+"_"+str(j)+'.joblib')

            tic()
            clf = PassiveAggressiveClassifier().fit(X_train, y_train)
            # calibrated = CalibratedClassifierCV(clf)
            # calibrated.fit(X_train, y_train)
            s = clf.score(X_test, y_test)
            print("PassiveAggressive: \t" + str(round(s, 4)) + " time:", round(toc(), 4))
            dump(clf, 'models_joblib/PassiveAggressive_'+str(i)+"_"+str(j)+'.joblib')

            tic()
            clf = SGDClassifier(loss='hinge').fit(X_train, y_train)
            # calibrated = CalibratedClassifierCV(clf)
            # calibrated.fit(X_train, y_train)
            s = clf.score(X_test, y_test)
            print("SGDClassifier: \t\t" + str(round(s, 4)) + " time:", round(toc(), 4))
            dump(clf, 'models_joblib/SGDClassifier_'+str(i)+"_"+str(j)+'.joblib')