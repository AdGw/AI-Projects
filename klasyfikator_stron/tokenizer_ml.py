from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import os
import sklearn.datasets
import pickle
import numpy as np

datapath = "kategorie_cleaned/"
train = sklearn.datasets.load_files(datapath + "/", description=None, categories=None,
                                    load_content=True,
                                    shuffle=True, encoding='utf-8', decode_error='strict', random_state=0)

texts = train.data  # Extract text
target = train.target  # Extract target

vocab_size = 5807
tokenizer = Tokenizer(num_words = vocab_size)  # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # Generate sequences
word_index = tokenizer.word_index
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Found {:,} unique words.'.format(len(word_index)))
