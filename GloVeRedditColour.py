import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
import pdb

GloVe = pd.read_hdf('data/wordvecs/GloVe50d.hdf',key='a')
def load_data():
    # To get nltk from /mnt/ drive - it doesn't look there by default
    nltk.data.path.append('/mnt/data/datasets/nltk_data/')
    print('Getting 50D GloVe vectors')
    GloVe = pd.read_hdf('data/wordvecs/GloVe50d.hdf',key='a')
    print('Done! Now getting Reddit data 0')
    Reddit = pd.read_hdf('data/Reddit/CommentsMay2015_size2p_22_' + str(3))
    print('Now getting Lexichrome')
    LexiChrome = pd.read_hdf('data/LexiChrome/LexiChromeData.hdf')
    return Glove, Reddit, LexiChrome

# Normalize vector - this is import for a parallelized word distance lookup
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

def get_Reddit(i):
    Reddit_df = pd.read_hdf('data/Reddit/CommentsMay2015_size2p_22_' + str(i))
    return Reddit_df
"""
Desc: Gets GloVe vectors for a given list of words
Input: List of tokenized lower case words
Output: List of corresponding GloVe vectors
"""
def GloVeComment(word_list):
  #  GloVe = get_GloVe()
    # Generate small dataframe containing the words we need
    comment_df = GloVe[GloVe['word'].isin(word_list)]
#    if len(comment_df) < 1:
    # The dataframe is in a different order so we query words individually
    glove_list = []
    # loop to preserve order
    for w in word_list:
        # Get vector from small dataframe
        #pdb.set_trace()
        if not comment_df[(GloVe.word == w)].empty:
            vector = comment_df[(GloVe.word == w)].vector.item()
            glove_list.append(vector)

    return glove_list


"""
Desc: Given a subreddit this function goes through a fixed link id (thread) and goes throughithe comments pertaining to
that link id. For each comment c_0 we look for a child comment c_1.
We say that a comment c_1 is a child of c_0 iff c_0.name == c_1.parent_id
Input: All dataframes
Output: (Parent, Child)
Where for both Parent and Child we have:
Comment:GloVe sequence, Color sequence, score, subreddit
"""
def reddit_glove_lexichrome(reddit_df):
    subreddits = ['reddevils', 'nottheonion', 'news']# 'worldnews','science', 'canada', 'atheism', 'Music']
    for sub in subreddits:
        sub_df = reddit_df[(reddit_df.subreddit == sub)]
        # clean
        sub_df = sub_df[['body', 'score', 'subreddit','name','parent_id', 'link_id']]
        sub_df = sub_df[(sub_df.body != '[deleted]')]
        # Get rid of comments that have no parent
        mask = sub_df['parent_id'].isin(sub_df.name.tolist()) | sub_df['name'].isin(sub_df.parent_id.tolist())
        sub_df = sub_df[mask]
        sub_df['lower'] = sub_df['body'].apply(str.lower)
        sub_df['tokenized'] = sub_df['lower'].apply(nltk.word_tokenize)
        sub_df['parent'] = sub_df['tokenized'].apply(GloVeComment)

    return sub_df





# Adds a column that has a sequence of word vectors corresponding to 'body' of
# comment for valid comments. A comment is valid if it can be viewed as a
# sequence of GloVe vectors.
#def Reddit2GloVeColour(Reddit_df):
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
