from scipy.sparse import csr_matrix, save_npz, vstack
import pickle
import numpy as np

def create_X():
    NUM_MODELS = 3
    for i in range(1, 1 + NUM_MODELS):
        all_sites_list = pickle.load(open("models/website_list_"+str(i)+".list", "rb"))
        vectorizer = pickle.load(open('models/vectorized_'+str(i)+'.dump', 'rb'))
        COLS = len(vectorizer.vocabulary_)
        X = csr_matrix((1, COLS), dtype=np.uint16)
        n = len(all_sites_list)
        BLOCK_SIZE = 5000
        for j in range(0, n // BLOCK_SIZE + 1):
            plus = BLOCK_SIZE
            if j == n // BLOCK_SIZE:
                plus = n % BLOCK_SIZE
            x_part = vectorizer.transform(all_sites_list[j * BLOCK_SIZE:(j * BLOCK_SIZE + plus)])
            X = vstack((X, x_part))

        save_npz('models/X_'+str(i)+'.npz', X[1:, :])
