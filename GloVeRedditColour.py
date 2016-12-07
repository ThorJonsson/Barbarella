import numpy as np
import pandas as pd
from tqdm import tqdm
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
Desc: Given a subreddit this function goes through a fixed link id (thread) and goes throughithe comments pertaining to
that link id. For each comment c_0 we look for a child comment c_1.
We say that a comment c_1 is a child of c_0 iff c_0.name == c_1.parent_id
Input: All dataframes
Output: (Parent, Child)
Where for both Parent and Child we have:
Comment:GloVe sequence, Color sequence, score, subreddit
"""
def reddit_clean():
    subreddits = ['reddevils', 'nottheonion', 'news', 'worldnews','science', 'canada', 'atheism', 'Music']
    # 11 is the number of dataframes we have
    reddit_df = []
    for i in range(11):
        print("Getting reddit file", i)
        reddit_df.append(get_Reddit(i))
        tmp_reddit = reddit_df[i]
        subreddit_mask = tmp_reddit['subreddit'].isin(subreddits)
        tmp_reddit = tmp_reddit[subreddit_mask]
            # clean
        tmp_reddit = tmp_reddit[['body', 'score', 'subreddit','name','parent_id', 'link_id']]
        tmp_reddit = tmp_reddit[(tmp_reddit.body != '[deleted]')]
        reddit_df[i] = tmp_reddit
    reddit_df = pd.concat(reddit_df,ignore_index=True)

    # Get rid of comments that have no parent
    mask = reddit_df['parent_id'].isin(reddit_df.name.tolist()) | reddit_df['name'].isin(reddit_df.parent_id.tolist())
    reddit_df = reddit_df[mask]
    reddit_df['body'] = reddit_df['body'].apply(str.lower)
    return reddit_df

