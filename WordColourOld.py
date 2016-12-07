import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm

# Gets an hdf file of reddit data
def load(filename):
    path = '/mnt/data/datasets/RedditComments/'
    Redditdf = pd.read_hdf(path + filename)
    return Redditdf

# It is very non-trivial to obtain temporal relation in reddit data

# Instead we propose to reconstruct each comment - thus obtaining a context
# vector


# Returns vector corresponding to a word
# GloVe[(GloVe.word == 'just')].vector.item()

# Sort values X.sort_values(['created_utc'],ascending=[True], inplace = True)

# Get column value and reset index Redditdf[(Reddit.subreddit == subreddit_name)].reset_index()

# Most frequent in a given column AskReddit['link_id'].value_counts().idxmax()

"""
Desc: Gets GloVe vectors for a given list of words
Input: List of tokenized lower case words
Output: List of corresponding GloVe vectors
"""
def GloVeComment(word_list):
    # Generate small dataframe containing the words we need
    comment_df = GloVe[GloVe['word'].isin(word_list)]
    if len(comment_df) != len(word_list):
        print('Comment does not have a complete GloVe representation')
    # The dataframe is in a different order so we query words individually
    glove_list = []
    # loop to preserve order
    for w in word_list:
        # Get vector from small dataframe
        comment_df[(GloVe.word == w)].vector.item()
        glove_list.append(w)

    return glove_list

"""
Desc: Given a subreddit it loads a thread

"""
# Objective, get a comment with associated colors and wordvectors
def colorize_comment():
    Reddit_df = load('CommentsMay2015_size2p_22_0')
    # Get list of comments
    Comments = Reddit_df['body'].tolist()
    # Go through comment and get a sequence of GloVe vectors
    # Here we need to preserve order so we make a list of numpy arrays



# Code that aims to import words and colours
def colour_vec(colour, list_o_colours):
    if type(colour) is str:
        """ Creates a one hot vector for colours """
        c_filter = lambda x, y: x == y

        """ Creates one-hot vector to represent colours """
        return  [int(c_filter(x, colour)) for x in list_o_colours]
    if type(colour) is dict:
        col_vec = [0]*len(list_o_colours)
        for idx, val in enumerate(list_o_colours):
            for key in colour:
                if(key == val):
                    col_vec[idx] = colour[key]
    # Consider the vector elements as probabilities which sum up to 1
    col_vec = [col_vec[x]/sum(col_vec) for x,val in enumerate(col_vec)]
    return col_vec

def word_vec(word, dictionary):
    """ Finds the word in the given dictionary
        and will assign appropriate word vector """
    with open(dictionary) as file_:
        for line in file_:
            dict_word = line.split(' ',1)[0]
            if(dict_word == word):
                line = line.strip('\n');
                return  line.split(' ',1)[1]
    return "Word not found"

def normalize(x):
    x_norm = np.zeros(x.shape)
    d = (np.sum(x ** 2)**(0.5))
    x_norm = (x/d)
    return x_norm

# Make a vocabulary, this will make looking up words faster
def build_glove_vocab(dictionary = 'data/wordvecs/glove.6B.50d.txt'):
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

def find_word_colour(word, dictionary):
    """ Finds the word in the given dictionary
        and will assign appropriate word vector
        This will continue until the end of the
        list to find other instances and colours
        for the multicolour words with score """
    #pdb.set_trace()
    with open(dictionary) as file_:
        colour = {}
        for line in file_:
            line_s = line.split('\t',3)
            dict_word = line_s[0].split('--',1)[0]

            if(dict_word == word):
                # Find 'VotesForThisColour' and 'TotalVotesCast'
                if(line_s[1].split('=',1)[1] == 'None'):
                    colour['none'] = 1
                    continue
                col_weight = float(line_s[2].split('=',1)[1]) / float(line_s[3].split('=', 1)[1])
                col = line_s[1].split('=',1)[1]
                colour[col] = col_weight
        # Check if there are any other colours besides "none" 
        # and ignore "none" if that is the case
        if 'none' in colour:
            if(len(colour) > 1 and colour['none'] == 1):
                del colour['none'] # why?
        elif not colour: # check if dict is empty
            colour['none'] = 1
    return colour

def find_word_colour(wordlist, dictionary):
    """ Finds the word in the given dictionary
        and will assign appropriate word vector
        This will continue until the end of the
        list to find other instances and colours
        for the multicolour words with score """
    with open(dictionary) as file_:
        colour = {}
        # Get the word we are looking for from colour dict
        for line in file_:
            line_s = line.split('\t',3)
            dict_word = line_s[0].split('--',1)[0]

            if(dict_word in wordlist):
                # Find 'VotesForThisColour' and 'TotalVotesCast'
                if(line_s[1].split('=',1)[1] == 'None'):
                    colour['none'] = 1
                    continue
                col_weight = float(line_s[2].split('=',1)[1]) / float(line_s[3].split('=', 1)[1])
                col = line_s[1].split('=',1)[1]
                colour[col] = col_weight
        # Check if there are any other colours besides "none"
        # and ignore "none" if that is the case
        if 'none' in colour:
            if(len(colour) > 1 and colour['none'] == 1):
                del colour['none'] # why?
        elif not colour: # check if dict is empty
            colour['none'] = 1
    return colour

class WordColour(object):
    """ dictionary is a link to a plain text dictionary like glove
        for pre-made word vectors """

    def __init__(self,\
                 word,\
                 vec_dict = 'wordvecs/glove.6B.50d.txt',\
                 col_dict = 'NRC-color-lexicon-senselevel-v0.92.txt',\
                 colour_list=['black', 'brown', 'white', 'grey',\
                              'pink', 'red','orange', 'yellow',\
                              'green', 'blue', 'purple', 'none']):
        # Any specific reason for using underscores?
        self.word   = word #Maybe redundant?
        self.colour = find_word_colour(self.word, col_dict)
        self.colour_vec = colour_vec(self.colour, colour_list)
        self.word_vec = word_vec(word, vec_dict)

    def print_wordcol(self):
        print('Word: %s \nColour: %s \nColVec: %s \nWordVec: %s \n' %(self.word,
                                                                         self.colour,
                                                                         ' '.join(str(x) for x in self.colour_vec),
                                                                         self.word_vec))
# Thoughts: Will we be able to find all the words in glove?
# This function simply gets the wordcolor object for every word in a sequence
# of words
def SequenceofWordsColour(word_seq):
    listofWC = []
    # Assuming that words are split by a space
    # We make all words lower case for glove compatibility
    for word in word_seq.lower().split():
        WC = WordColour(word)
        listofWC.append(WC)
    return listofWC

# To find the nearest neighbors for words with color we need to make
# a matrix that contains all word vectors of the words found in LexiChrome
# http://stackoverflow.com/questions/20083098/improve-pandas-pytables-hdf5-table-write-performance
def LexiChromeVocab(vec_dict = 'data/wordvecs/glove.6B.50d.txt',\
                    col_dict = 'data/LexiChrome/NRC-color-lexicon-senselevel-v0.92.txt',\
                    colour_list=['black', 'brown', 'white', 'grey',\
                              'pink', 'red','orange', 'yellow',\
                              'green', 'blue', 'purple', 'none']):

    # We go through the lexichrome data and find the word vector pertaining to
    # each entry, we store this in a list of dicts,
    listofLexiChrome = []
    i = 0
    with open(col_dict) as file_:
        for line in tqdm(file_):
            line_s = line.split('\t',3)
            dict_word = line_s[0].split('--',1)[0]
            vec = word_vec(dict_word,vec_dict)
            colour = find_word_colour(dict_word, col_dict)
            col_vec = colour_vec(colour,colour_list)
            listofLexiChrome.append({'word': dict_word,\
                                     'GloVe Vector': vec,\
                                     'Colour': colour,\
                                     'Colour Vector': col_vec})
    return pd.DataFrame(listofLexiChrome)

# From Stanford GloVe
def distance(W, vocab, ivocab, input_term):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


if __name__ == "__main__":
    N = 100;          # number of closest words that will be shown
    W, vocab, ivocab = generate()
    while True:
        input_term = raw_input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            distance(W, vocab, ivocab, input_term)



# Color mixing
# http://stackoverflow.com/questions/1351442/is-there-an-algorithm-for-color-mixing-that-works-like-mixing-real-colors


