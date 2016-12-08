import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 
import pdb
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial.distance import cosine

from keras.layers import Dense, LSTM, Embedding, merge, Input, Dropout
from keras import objectives
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.preprocessing import sequence
from keras import backend as K
import WordColour as wc

# Input hullaballoo
EMBEDDING_DIM = 50
MAX_LENGTH = 100
VOCAB_SIZE = 20000
INPUT_FILE = "/mnt/data/datasets/RedditComments/Reddit4Reconstruction.pcl"
GLOVE_FILE = "/mnt/data/datasets/RedditComments/Barbarella/wordvecs/glove.6B.50d.txt"
print('Reading in reddit data')
df = pd.read_pickle(INPUT_FILE)

X_train = df['tokenized'][:250000]

subreddit_list = ['reddevils', 'nottheonion', 'news', 'worldnews','science', 'canada', 'atheism', 'Music']
# create one hot vectors for the above ordering 
y_strings = df['subreddit'][:250000]


# One hot vectors for the subreddits for classfication 
y = y_strings.apply(wc.subreddit_vec)

print(len(X_train))

# Create dictionary of all words in and vectors
embeddings_index = {}
f = open(GLOVE_FILE)
cnt = 0
for line in f:
    cnt +=1
    if cnt > 20000:
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
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

inv_vocab = {}
for idx,key in enumerate(embeddings_index):
    embedding_vector = embeddings_index.get(key)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        #pdb.set_trace()
        embedding_matrix[idx + 1] = embedding_vector
        inv_vocab[key] = idx + 1

def comment2idx(tokenized_list):
    idx_list = []
    for w in tokenized_list:
        try:
            idx_list.append(inv_vocab[w])
        except KeyError:
            idx_list.append(0)
    return idx_list

X_train = X_train.apply(comment2idx)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LENGTH)
print(X_train.shape)
print(y.shape)

Y_list = y.tolist()
Y = np.array(Y_list)

#df = pd.DataFrame({'X': X_train, 'y':y})
#df.to_pickle("subreddit_class_data.pcl")

# Create LSTM encoder and decoder for sequence of Word Vectors
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE,
                      output_dim=EMBEDDING_DIM,
                      input_length=MAX_LENGTH,
                      weights=[embedding_matrix],
                      mask_zero=False,
                      trainable=False))

model.add(LSTM(64, activation='relu', dropout_W=0.1, dropout_U=0.1, return_sequences=False))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(8, activation='softmax'))

print(model.layers[0].output_dim)
print(model.layers[1].output_dim)

# Compile and train the model 
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y, batch_size=512, nb_epoch=100, validation_split=0.1, shuffle=True)

# Create NN for LSTM output to get the closest approximate word vector 
# This may be supplanted by a categorical cross entropy approach (usual way)

"""
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine')
nbrs.fit(X_train)

# For output predict with LSTM model and then run through neighbors model to get 
# closest word vectors 

for idx in LSTM_out:
    LSTM_out[idx] = kneighbors(1, LSTM_out[idx], return_distance=False)

"""
