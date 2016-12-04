import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
#from tensorflow.models.rnn import rnn_cell, seq2seq

data = 'data/mpii/'

def normalize(x):
    x_norm = np.zeros(x.shape)
    d = (np.sum(x ** 2)**(0.5))
    x_norm = (x/d)
    return x_norm

# Make a vocabulary, this will make looking up words faster
def build_GloVe_vocab(dictionary = 'data/wordvecs/glove.6B.50d.txt'):
    GloVe = []
    with open(dictionary) as file_:
        for line in tqdm(file_):
            # To deal with empty lines
            if line == '\n':
                continue
            word = line.split(' ',1)[0]
            line = line.strip('\n')
            vector = line.split(' ',1)[1]
            x = np.array(vector.split(' '),dtype=float)
            vector = normalize(x)
            GloVe.append({'word' : word, 'vector': vector})

    return pd.DataFrame(GloVe)

def save_GloVe():
    GloVe_df = build_GloVe_vocab()
    GloVe_df.to_hdf('data/wordvecs/GloVe50d.hdf',key='a')

def get_GloVe():
    GloVe_df = pd.read_hdf('data/wordvecs/GloVe50d.hdf',key='a')
    return GloVe_df

def get_LexiChrome():
    LexiChrome_df = pd.read_hdf('data/LexiChrome/LexiChromeData.hdf')
    return LexiChrome_df

def get_Reddit(i):
    Reddit_df = pd.read_hdf('data/Reddit/CommentsMay2015_size2p_22_' + str(i))
    return Reddit_df




# Adds a column that has a sequence of word vectors corresponding to 'body' of
# comment for valid comments. A comment is valid if it can be viewed as a
# sequence of GloVe vectors.
def Reddit2GloVeColour(Reddit_df):
    # sort first by created_utc
    # sort first by parentid
    # then sort by link id
    # then sort by subreddit
    #Reddit_df.sort_values(['

#if __name__ == "__main__":
#    # Note that my_img can just as well be an array of images since this is a
#    # tensor
#    my_img = fetch_img_tensor()
#    init_op = tf.initialize_all_variables()
#    with tf.Session() as sess:
#        sess.run(init_op)
#        # Start populating the filename queue.
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(coord=coord)
#        ''' Build Graph'''
#        
#        ''' Evaluate Tensor'''
#        for i in range(1): #length of your filename list
#            image = my_img.eval() #here is your image Tensor :) 
#        print(image.shape)
#        Image.fromarray(np.asarray(image)).show()
#        coord.request_stop()
#        coord.join(threads)
#
#sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
#label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]
# 
#def make_example(sequence, labels):
#    # The object we return
#    ex = tf.train.SequenceExample()
#    # A non-sequential feature of our example
#    sequence_length = len(sequence)
#    ex.context.feature["length"].int64_list.value.append(sequence_length)
#    # Feature lists for the two sequential features of our example
#    fl_tokens = ex.feature_lists.feature_list["tokens"]
#    fl_labels = ex.feature_lists.feature_list["labels"]
#    for token, label in zip(sequence, labels):
#        fl_tokens.feature.add().int64_list.value.append(token)
#        fl_labels.feature.add().int64_list.value.append(label)
#    return ex
# 
## Write all examples into a TFRecords file
#with tempfile.NamedTemporaryFile() as fp:
#    writer = tf.python_io.TFRecordWriter(fp.name)
#    for sequence, label_sequence in zip(sequences, label_sequences):
#        ex = make_example(sequence, label_sequence)
#        writer.write(ex.SerializeToString())
#    writer.close()
