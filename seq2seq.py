import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial.distance import cosine

from keras.layers import Dense, LSTM, Embedding
from keras import objectives
from keras import optimizers.Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Input hullaballoo
GLOVE_FILE = "wordvecs/glove.6B.100d.txt"
EMBEDDING_DIM = 100
INPUT_DIR = ""

# Create dictionary of all words in and vectors
embeddings_index = {}
f = open(GLOVE_FILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Compute the embedding matrix  
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Create LSTM encoder and decoder for sequence of Word Vectors
model = Sequential()

model.add(Embedding(len(word_index) + 2,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    mask_zero=True,
                    trainable=False)
model.add(LSTM(64, input_dim=EMBEDDING_DIM, 
               activation='relu', 
               dropout_W=0.1, dropout_U=0.1, 
               return_sequences=True))
model.add(LSTM(100, activation='relu', 
               dropout_W=0.1, dropout_U=0.1, 
               return_sequences=True))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='cosine_proximity', optimizer=adam)

model.fit(X_train, X_train, 
          n_epoch=100,
          batch_size=50, validation_split=0.1)

# Create NN for LSTM output to get the closest approximate word vector 
# This may be supplanted by a categorical cross entropy approach (usual way)

nbrs = NearestNeighbors(n_neighbors=5, metric='cosine')
nbrs.fit(X_train)

# For output predict with LSTM model and then run through neighbors model to get 
# closest word vectors 

for idx in LSTM_out:
    LSTM_out[idx] = kneighbors(1, LSTM_out[idx], return_distance=False)


