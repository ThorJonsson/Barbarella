from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from WordColour import *
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import math
import random 
import itertools 

colour_dict = "NRC"
word_vec_dict = ""
data = pd.read_hdf("LexiChromeDataVector.hdf");
data = shuffle(data)
num = len(data)

''' Take out 'None' labels and insert them into the test data '''
# Divide the data into 80% training, 10% validation and 10% testing
?t
rain_data = data[['Colour Vec', 'GloVe Vector', 'colour']][0:math.floor(num*0.9)]
test_data  = data[['Colour Vec', 'GloVe Vector', 'colour']][math.floor(num*0.9):len(data)]

# Convert data frame into dictionary 
X_train = np.vstack(train_data['GloVe Vector'])
y_train = np.vstack(train_data['Colour Vec'])

X_test = np.vstack(test_data['GloVe Vector'])
y_test = np.vstack(test_data['Colour Vec'])

""" Thresholds the colour vector values such that all non-max 
    colours are suppressed 
"""
for i in range(0,len(y_train)):
    max_ind = y_train[i] == np.amax(y_train[i])
    y_train[i] = max_ind.astype(int)
    ind = np.where(y_train[i] == 1)
    if(len(ind[0] > 1)):
        temp = np.zeros([1,12])
        temp[0,ind[0][random.randint(0,len(ind[0])-1)]] = 1
        y_train[i] = temp

for i in range(0,len(y_test)):
    max_ind = y_test[i] == np.amax(y_test[i])
    y_test[i] = max_ind.astype(int)
    ind = np.where(y_test[i] == 1)
    if(len(ind[0] > 1)):
        temp = np.zeros([1,12])
        temp[0,ind[0][random.randint(0,len(ind[0])-1)]] = 1
        y_test[i] = tempwc.

#data = {'trX':train_matX, 'trY': train_matY}

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape :
# here, 20-dimensional vectors.
print(X_train.shape)
model.add(Dense(512, input_dim=X_train.shape[1], init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(12, init='uniform'))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          nb_epoch=1,
          batch_size=50, validation_split=0.05)

score = model.evaluate(X_test, y_test, batch_size=64)

print(score)


''' Show of the classes in a t-sne organized visualization '''



embed_model = TSNE(n_components=2)
embedding = embed_model.fit_transform(X_test)

list_o_colours=['black', 'brown', 'white', 'grey',\
                'pink', 'red','orange', 'yellow',\
                'green', 'blue', 'purple', 'none']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0, len(y_test)):
    # Use y_test as a mask for the list of colours 
    col_out = list(itertools.compress(list_o_colours, y_test[i]))
    if(col_out == 'none'):
        col_out = 'white'

    point = embedding[i]
    if(not (i % 10)):
        ax.scatter(point[0], point[1], point[2], c=col_out, s=5)

plt.show()
