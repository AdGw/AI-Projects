import pickle
from scipy.sparse import lil_matrix, save_npz
import numpy as np
import time
import os

def create_Y():
    # all_sites_list = []
    NUM_MODELS = 3
    for i in range(1, 1 + NUM_MODELS):
        p = os.listdir('3_models/'+str(i))
        cpt = sum([len(files) for r, d, files in os.walk('3_models/'+str(i))])
        Y = np.zeros((cpt,), dtype=np.uint8)
        idx = 0
        for j, dir in enumerate(p):
            for file in os.listdir('3_models/' + str(i) + "/" + dir):
                Y[idx] = j
                idx += 1

        np.savez_compressed('models/Y_'+str(i)+'.npz', Y)

