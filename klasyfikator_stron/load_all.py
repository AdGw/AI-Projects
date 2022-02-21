from joblib import load
# from keras.layers import Activation
# from keras.utils.generic_utils import get_custom_objects
# import keras
import os
import pickle
# import tensorflow

# def mish(x, beta = 1):
#     return (x * tensorflow.math.tanh(tensorflow.keras.activations.softplus(x)))
# get_custom_objects().update({'mish': mish})

with open("skip_fonts.txt", "r", encoding="utf-8-sig") as f:
    skip_fonts = f.read()
    
with open("models_transformer/labels.pickle", "rb") as f:
    labels = pickle.load(f)

# model_dl = keras.models.load_model('model_pl.h5', custom_objects ={"Activation": Activation(mish)})
# model_dl = keras.models.load_model('model_gru2.h5', custom_objects ={"Activation": Activation(mish)})

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('models/headers.pickle', 'rb') as fp:
    prepared = pickle.load(fp)
prepared.pop(0)

def load_models():
    p_vect = os.listdir("models_transformer/")
    arr_model = []
    c = 0
    for i in p_vect:
        if i.count("model_"):
            with open("models_transformer/"+i, "rb") as f:
                arr_model.append({c: pickle.load(f)})
            c += 1
    return arr_model

def load_transformers():
    p_vect = os.listdir("models_transformer/")
    arr_transformer = []
    c = 0
    for i in p_vect:
        if i.count("tf_transformer"):
            with open("models_transformer/"+i, "rb") as f:
                arr_transformer.append({c: pickle.load(f)})
            c += 1
    return arr_transformer

def load_vect():
    p_vect = os.listdir("models_transformer/")
    arr_vect = []
    c = 0
    for i in p_vect:
        if i.count("count_vect"):
            with open("models_transformer/"+i, "rb") as f:
                arr_vect.append({c: pickle.load(f)})
            c += 1
    return arr_vect

def load_selector():
    p_selector = os.listdir("models_transformer/")
    arr_selector = []
    c = 0
    for i in p_selector:
        if i.count("selector"):
            with open("models_transformer/"+i, "rb") as f:
                arr_selector.append({c: pickle.load(f)})
            c += 1
    return arr_selector

def load_models_joblib():
    model = os.listdir("models_joblib/")
    arr_model = []
    c = 0
    for i in model:
        clf = load('models_joblib/'+i)
        arr_model.append({c: clf})
        c += 1
    return arr_model

def load_models_features():
    arr_model = []
    c = 0
    path_features = os.listdir("models_features/")
    for i in path_features:
        if i.count("features"):
            with open("models_features/"+i, "rb") as f:
                arr_model.append({c: pickle.load(f)})
        else:
            model = keras.models.load_model('models_features/DL.h5', custom_objects ={"Activation": Activation(mish)})
            arr_model.append({c: model})
        c += 1
    return arr_model

categories = os.listdir("kategorie/")
vectorizer_1 = pickle.load(open('models/vectorized_1.dump', 'rb'))
vectorizer_2 = pickle.load(open('models/vectorized_2.dump', 'rb'))
vectorizer_3 = pickle.load(open('models/vectorized_3.dump', 'rb'))

arr_model_joblib = load_models_joblib()
arr_model = load_models()
arr_transformer = load_transformers()
arr_vect = load_vect()
arr_selector = load_selector()
arr_model_features = load_models_features()