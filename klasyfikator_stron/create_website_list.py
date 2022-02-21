import pickle
import numpy as np
import os
import time


def create_list():
    p = os.listdir("3_models/")
    for i in p:
        all_sites_list = []
        p2 = os.listdir("3_models/" + i)
        for j in p2:
            p3 = os.listdir("3_models/" + i + "/" + j)
            for file in p3:
                with open("3_models/" + i + "/" + j + '/' + file, 'r', encoding='utf-8-sig') as f:
                    website = f.read()
                all_sites_list.append(website)
            # print('[done] ', dir)
        pickle.dump(all_sites_list, open("models/website_list_"+str(i)+".list", "wb"))
