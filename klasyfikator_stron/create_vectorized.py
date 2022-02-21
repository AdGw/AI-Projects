from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split


def create_vectorize():
    NUM_MODELS = 3
    for i in range(1, 1 + NUM_MODELS):
        all_sites_list = pickle.load(open("models/website_list_"+str(i)+".list", "rb"))
        n_samples = len(all_sites_list)
        data = np.random.randn(n_samples, 1)
        labels = np.random.randint(1, size=n_samples)
        indices = np.arange(n_samples)
        x1, x2, y1, y2, idx_train, idx_test = train_test_split(
            data, labels, indices, test_size=0.3, random_state=0, shuffle=True)
        all_sites_list_train = [all_sites_list[j] for j in idx_train]
        vectorizer = CountVectorizer(encoding='utf-8-sig', dtype=np.uint16, max_features=270000)
        vectorizer.fit(all_sites_list_train)
        pickle.dump(vectorizer, open("models/vectorized_"+str(i)+".dump", "wb"))