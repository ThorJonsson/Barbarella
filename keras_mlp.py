from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

from WordColour import *
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import math

colour_dict = "NRC"
word_vec_dict = ""

data = pd.read_hdf("LexiChromeDataVector.hdf");

data = shuffle(data)
num = len(data)

# Divide the data into 80% training, 10% validation and 10% testing

train_data = data[['Colour Vec', 'GloVe Vector']][0:math.floor(num*0.9)]
test_data  = data[['Colour Vec', 'GloVe Vector']][math.floor(num*0.9):len(data)]

# Convert data frame into dictionary 
X_train = np.vstack(train_data['GloVe Vector'])
y_train = np.vstack(train_data['Colour Vec'])

X_test = np.vstack(test_data['GloVe Vector'])
y_test = np.vstack(test_data['Colour Vec'])

#data = {'trX':train_matX, 'trY': train_matY}

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=50, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(12, init='uniform'))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=100)

score = model.evaluate(X_test, y_test, batch_size=16)
