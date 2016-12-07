import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 

from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial.distance import cosine

from keras.layers import Dense, LSTM, Embedding, merge, Input, Masking
from keras import objectives
from keras import optimizers.Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.preproccessing import sequence 


# Input hullaballoo
reddit_df = pd.read_hdf("")
EMBEDDING_DIM = 50
MAX_LENGTH = 100
INPUT_FILE = "/mnt/data/datasets/RedditComments/Reddit4Reconstruction.pcl"
GLOVE_FILE = ""

df = pd.read_pickle(INPUT_FILE)

X_train = df['body'];
y_train_class = df['subreddit']

print(len(X_train))


# Pad the input sequences ro be uniform length
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH)
# y_train = sequence.pad_sequences(y_train, maxlen=MAX_LENGTH)

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

"""

# Create LSTM encoder and decoder for sequence of Word Vectors
model = Sequential()

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
"""

# Create LSTM encoder and decoder for sequence of Word Vectors

input_layer = Input(shape=(MAX_LENGTH, EMBEDDING_DIM), dtype='float32', name='input_layer')

embedding = Embedding(input_dim=len(word_index) + 2,
                      output_dim=EMBEDDING_DIM,
                      input_length=MAX_LENGTH,
                      weights=[embedding_matrix],
                      mask_zero=True,
                      trainable=False, name='embedding')(input_layer)

encoder = LSTM(64, 
               activation='relu', 
               dropout_W=0.1, dropout_U=0.1, 
               return_sequences=True, name='encoder')(embedding)

output_layer = LSTM(100, activation='relu', 
                    dropout_W=0.1, dropout_U=0.1, 
                    return_sequences=True, name='output_layer')(input_layer)

x = Dense(128, activation='relu')(input_layer)
class_layer = Dense(128, activation='softmax', name='class_layer')(x)

model = Model(input=[input_layer], output=[output_layer, class_layer])

model.compile(optimizer=adam, 
              loss={'output_layer': 'cosine_proximity', 'class_layer': 'categoircal_crossentropy'}, 
              loss_weights=[1., 1.])

model.fit({'input_layer': X_train}, 
          {'output_layer': X_train, 'class_layer': y_train_class},
          nb_epochs=100, batch_size=32)

# Create NN for LSTM output to get the closest approximate word vector 
# This may be supplanted by a categorical cross entropy approach (usual way)

nbrs = NearestNeighbors(n_neighbors=5, metric='cosine')
nbrs.fit(X_train)

# For output predict with LSTM model and then run through neighbors model to get 
# closest word vectors 

for idx in LSTM_out:
    LSTM_out[idx] = kneighbors(1, LSTM_out[idx], return_distance=False)


