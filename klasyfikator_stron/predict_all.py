# import tensorflow
# from keras.preprocessing.sequence import pad_sequences
# from keras_preprocessing.text import Tokenizer
from joblib import load
import numpy as np
import pickle
import os
from collections import Counter
from ttictoc import tic, toc
from load_all import labels, tokenizer, arr_model, arr_transformer, arr_vect, skip_fonts, arr_selector, \
    arr_model_joblib, vectorizer_1, vectorizer_2, vectorizer_3, prepared, categories, arr_model_features
    # model_dl, 
import threading, queue
import random

def machine_learning(prepared_txt, transformer, model, vect, labels, selector):
    p_count = vect.transform(prepared_txt)
    p_count = selector.transform(p_count)
    p_tfidf = transformer.transform(p_count)
    pred = model.predict(p_tfidf)
    ml_proba = categories[pred[0]]
    return ml_proba

def machine_learning_features(prepared, prepared_txt, model, categories):
    X_list = compare_arr(prepared, prepared_txt[0])
    if str(model)[0] == "<":
        pred = model.predict([X_list])
        ml_proba = categories[np.argmax(pred)]
    else:
        pred = model.predict([X_list])
        ml_proba = categories[pred[0]]
    return ml_proba

# def deep_learning(prepared_txt, tokenizer, max_length, categories, model):
#     sequences = tokenizer.texts_to_sequences(prepared_txt)
#     data = pad_sequences(sequences, maxlen=max_length)
#     example = data[0]
#     pred = model.predict(example.reshape(1, len(example)))
#     deep_proba = categories[np.argmax(pred)]
#     return deep_proba

def machine_learning_joblibs(prepared_txt, arr_model_joblib, vectorizer_1, vectorizer_2, vectorizer_3, c):
    p = os.listdir("kategorie/")
    if c == 1:
        x_test = vectorizer_1.transform(prepared_txt)
        y_pred = arr_model_joblib.predict(x_test)
        c += 1
        return c, p[y_pred[0]]
    elif c == 2:
        x_test = vectorizer_2.transform(prepared_txt)
        y_pred = arr_model_joblib.predict(x_test)
        c += 1
        return c, p[y_pred[0]]
    else:
        x_test = vectorizer_3.transform(prepared_txt)
        y_pred = arr_model_joblib.predict(x_test)
        c = 1
        return c, p[y_pred[0]]

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    added = d1_keys - d2_keys
    return added

def compare_arr(prepared, text):
    arr = []
    arr2 = []
    dict_file = {}
    dict_col = {}
    for i in prepared:
        dict_col.update(Convert([i, 0]))
    counter = Counter(text.split(" "))
    [i for i in prepared]
    for j in counter:
        res = j, counter[j]
        res = list(res)
        dict_file.update(Convert(res))
    z = {**dict_col, **dict_file}
    added = dict_compare(dict_file, dict_col)
    [z.pop(i) for i in added]
    arr2 = [z[i] for i in z]
    return arr2

def char_replace(text):
    for c in text:
        if not c.isalpha():
            text = text.replace(c, " ")
        else:
            pass
    return text

def main(item, text_original):
    arr_ml = []
    arr_ml_features = []
    arr_ml_joblib = []
    arr_ml_dict = []
    count = 1
    origin = "C:/Users/Adi/Desktop/dane_3/"
    # with open(origin + item, 'r', encoding="utf-8") as f:
    #     text_original = f.read()
    lines = (line for line in text_original.split() if
             not line.count("http://") and not line.count("www.") and not line.count("https://"))
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    line = ' '.join(char_replace(chunk.lower()) for chunk in chunks if 3 <= len(chunk) < 20)
    line = line.translate(str.maketrans(' ', ' ', skip_fonts)).lower()
    prepared_txt = ''
    for j in line.split(" "):
        if len(j) > 20:
            pass
        elif len(j) <= 2:
            pass
        else:
            prepared_txt += j + " "

    for i in range(len(arr_model_joblib)):
        count, res_ml_joblib = machine_learning_joblibs([prepared_txt], arr_model_joblib[i][i], vectorizer_1, vectorizer_2, vectorizer_3, count)
        arr_ml_joblib.append(res_ml_joblib)

    for i in range(len(arr_model)):
        res_ml = machine_learning([prepared_txt], arr_transformer[i][i], arr_model[i][i], arr_vect[i][i], labels, arr_selector[i][i])
        arr_ml.append(res_ml)

    for i in range(len(arr_model_features)):
       res_ml_features = machine_learning_features(prepared, [prepared_txt], arr_model_features[i][i], categories)
       arr_ml_features.append(res_ml_features)

    # res_dl = deep_learning([prepared_txt], tokenizer, 2000, categories, model_dl)
    # arr_ml.append(res_dl)
    
    final_list = arr_ml_joblib+arr_ml+arr_ml_features
    counter = Counter(final_list)
    print(counter)
    if list(counter.values())[0] >= 29:
        if not os.path.exists("new_data/" + list(counter)[0]):
            os.makedirs("new_data/" + list(counter)[0])
        with open("new_data/" + list(counter)[0] + "/" + item, "w", encoding="utf-8") as f:
            f.write(text_original)
        os.remove(origin + item)
    else:
        with open("przerobione/"+ item, "w", encoding="utf-8") as f:
            f.write(text_original)
        os.remove(origin + item)
    # q.task_done()


def worker(q):
    while True:
        item = q.get()
        # print(item)
        item_path = item["path"]
        content = item["content"]
        main(item_path, content)

THREAD_COUNT = 128

q = queue.Queue()
threads = [
    threading.Thread(target=worker, args=(q,))
    for _ in range(THREAD_COUNT)
]


for t in threads:
    t.start()

# for i in range(64):
#     thread = threading.Thread(target = worker)
#     thread.start()

cnt = 0

path_doc = os.listdir("C:/Users/Adi/Desktop/dane_3/")
arr = []
for i in path_doc:
    with open("C:/Users/Adi/Desktop/dane_3/"+i, 'r', encoding="utf-8") as f:
        text = f.read()
    dictionary = {i:text}
    arr.append(dictionary)
    print(len(arr), i)

random.shuffle(arr)

def get_keys(arr):
    return arr.keys()

def get_values(arr):
    return arr.values()    

while cnt < len(arr):
    try:
        d = {
            "path": list(arr[cnt].keys())[0],
            "content": list(arr[cnt].values())[0]
        }
        q.put(d)
        # if threading.active_count() < 5:
        #     print("reborn")
        #     threading.Thread(target=worker, daemon=True).start()
        cnt += 1
    except queue.Full as e:
        print("-full")
        pass
print('all task requests sent\n', end='')
q.join()
print('all work completed')
