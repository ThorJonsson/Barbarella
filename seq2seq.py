import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 
import pdb
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial.distance import cosine

from keras.layers import Dense, LSTM, Embedding, merge, Input, Masking
from keras import objectives
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.preprocessing import sequence
import nltk
nltk.data.path.append('/mnt/data/datasets/nltk_data/')
# Input hullaballoo
EMBEDDING_DIM = 50
MAX_LENGTH = 100
INPUT_FILE = "/mnt/data/datasets/RedditComments/Reddit4Reconstruction.pcl"
GLOVE_FILE = "/mnt/data/datasets/RedditComments/Barbarella/wordvecs/glove.6B.50d.txt"
print('Reading in reddit data')
df = pd.read_pickle(INPUT_FILE)

X_train = df['tokenized'][:10000]

y_train_class = df['subreddit'][:10000]

print(len(X_train))


# Pad the input sequences ro be uniform length
# y_train = sequence.pad_sequences(y_train, maxlen=MAX_LENGTH)

# Create dictionary of all words in and vectors
embeddings_index = {}
f = open(GLOVE_FILE)
cnt = 0
for line in f:
    cnt +=1
    if cnt > 10000:
        break
    if line == '\n':
        continue
    values = line.split()
    #pdb.set_trace()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Compute the embedding matrix
embedding_matrix = np.zeros((10000, EMBEDDING_DIM))

inv_vocab = {}
for idx,key in enumerate(embeddings_index):
    embedding_vector = embeddings_index.get(key)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        #pdb.set_trace()
        embedding_matrix[idx + 1] = embedding_vector
        inv_vocab[key] = idx + 1

X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH)
# Create LSTM encoder and decoder for sequence of Word Vectors

input_layer = Input(shape=(MAX_LENGTH, EMBEDDING_DIM), dtype='float32', name='input_layer')

embedding = Embedding(input_dim=len(10000) + 2,
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
                    return_sequences=True, name='output_layer')(encoder)

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


