from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier, Perceptron, \
    LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.calibration import CalibratedClassifierCV
from ttictoc import tic, toc
from keras.backend import sigmoid
import tensorflow
import pickle
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation


def evaluate_models_features():
    # df = pd.read_parquet('models/DL_features' + '.parquet', engine='pyarrow')
    # shape = df.shape
    # print(shape)
    # df.drop_duplicates()
    # shape = df.shape
    # print(shape)
    # feature_cols = df.columns
    # X = df.loc[:, feature_cols != 'Category']
    # y = df.Category
    # X = X.to_numpy()
    # y = y.to_numpy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # def mish(x, beta=1):
    #     return x * tensorflow.math.tanh(tensorflow.keras.activations.softplus(x))

    # get_custom_objects().update({'mish': mish})

    # model = Sequential()
    # model.add(Dense(256, activation='mish', input_shape=(shape[1] - 1,)))
    # model.add(Dropout(0.3))
    # model.add(Dense(128, activation='mish'))
    # model.add(Dropout(0.3))
    # model.add(Dense(120, activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # model.summary()

    # X_train = np.asarray(X_train)
    # y_train = np.asarray(y_train)

    # history = model.fit(X_train, y_train, epochs=20, validation_split=0.3, batch_size=32, verbose=1)
    # model.save("models_features/DL.h5")



    # # print(score)
    # import matplotlib.pyplot as plt
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model accuracy')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    NUM_MODELS = 3
    for i in range(1, 1 + NUM_MODELS):
        df = pd.read_parquet('models/ML_features_' + str(i) + '.parquet', engine='pyarrow')
        feature_cols = df.columns
        X = df.loc[:, feature_cols != 'Category']
        y = df.Category
        X = X.to_numpy()
        y = y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        tic()
        clf = MultinomialNB().fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        print("MultinomialNB: \t\t" + str(round(s, 4)) + " time:", round(toc(), 4))
        with open("models_features/MultinomialNB_"+str(i)+"_features", "wb") as f:
            pickle.dump(clf, f)

        tic()
        clf = ComplementNB().fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        print("ComplementNB: \t\t" + str(round(s, 4)) + " time:", round(toc(), 4))
        with open("models_features/ComplementNB_"+str(i)+"_features", "wb") as f:
            pickle.dump(clf, f)

        tic()
        clf = LinearSVC(max_iter=100000000).fit(X_train, y_train)
        # calibrated = CalibratedClassifierCV(clf)
        # calibrated.fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        print("LinearSVC: \t\t" + str(round(s, 4)) + " time:", round(toc(), 4))
        with open("models_features/LinearSVC_"+str(i)+"_features", "wb") as f:
            pickle.dump(clf, f)

        tic()
        clf = PassiveAggressiveClassifier(max_iter=1000).fit(X_train, y_train)
        # calibrated = CalibratedClassifierCV(clf)
        # calibrated.fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        print("PassiveAggressive: \t" + str(round(s, 4)) + " time:", round(toc(), 4))
        with open("models_features/PassiveAggressive_"+str(i)+"_features", "wb") as f:
            pickle.dump(clf, f)
