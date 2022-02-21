import os
from collections import Counter
from ttictoc import tic, toc
import csv
import pandas as pd
import pickle
import numpy as np
import glob


def create():
    def Convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct

    def dict_compare(d1, d2):
        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
        added = d1_keys - d2_keys
        return added

    def compare_arr(prepared, text, c):
        ct = 0
        arr = []
        arr2 = [c]
        dict_file = {}
        dict_col = {}
        for i in prepared:
            dict_col.update(Convert([i, 0]))
        counter = Counter(text.split(" "))
        for i in prepared:
            arr.append(i)
        for j in counter:
            res = j, counter[j]
            res = list(res)
            dict_file.update(Convert(res))
        z = {**dict_col, **dict_file}
        added = dict_compare(dict_file, dict_col)
        for i in added:
            z.pop(i)
        for x in z:
            if ct == 0:
                pass
            else:
                arr2.append(z[x])
            ct += 1
        return arr2

    p = os.listdir('słowniki/')

    # if "headers" in check_file:
    #     with open('models/headers.pickle', 'rb') as fp:
    #         prepared = pickle.load(fp)
    # else:
    all_str = []
    for i in p:
        p2 = os.listdir('słowniki/' + i)
        for j in p2:
            with open('słowniki/' + i + '/' + j, 'r', encoding='utf-8') as f:
                file_r = f.read()
            file_r = file_r.split(",")
            for k in file_r:
                all_str.append(k)
    prepared = set(all_str)
    prepared = sorted(prepared)
    prepared.insert(0, "Category")
    with open('models/headers.pickle', 'wb') as fp:
        pickle.dump(prepared, fp)

    c = 0
    path = os.listdir("3_models/")
    for i in path:
        df = pd.DataFrame(columns=prepared, dtype=np.int16)
        listed = os.listdir("3_models/" + i)
        l = []
        for j in listed:
            listed_txt = os.listdir("3_models/" + i + "/" + j)
            for k in listed_txt:
                with open("3_models/" + str(i) + "/" + j + "/" + k, "r", encoding="utf-8") as f:
                    text_original = f.read()
                arr = compare_arr(prepared, text_original, c)
                l.append(arr)
            # print(len(l))
            c += 1
        c = 0
        df = pd.DataFrame(l, columns=prepared, dtype=np.int16)
        print(df)
        df.to_parquet('models/ML_features_' + i + '.parquet', index=False, engine='pyarrow')
