# Parse the reddit data

# Objective: Make an ordered list of comments. A subreddit thread will be considered as a conversation about a subject
# in the context of the theme/modality given by the subreddit itself.
# Thoughts
# Will there be a difference in perplexity between different modalities? How can we tune between different modalities?
#   In [31]: a.columns
#   Out[31]: 
#   Index(['archived', 'author', 'author_flair_css_class', 'author_flair_text',
#       'body', 'controversiality', 'created_utc', 'distinguished', 'downs',
#       'edited', 'gilded', 'id', 'link_id', 'name', 'parent_id',
#       'removal_reason', 'retrieved_on', 'score', 'score_hidden', 'subreddit',
#       'subreddit_id', 'ups'],
#      dtype='object')
#

# First consider how to get a single comment without using too much memory 
import json
import pandas as pd
# Get N number of comments
def hdf5_comments():
    with open('/mnt/data/datasets/RedditComments/RC_2015-05','r') as fname:
        batch = []
        i=0 # To keep track of line number
        batch_number=0
        p = 22
        break_batch = 2**(p)
        for line in fname:
            # fetches a comment as a dict
            line_dict = json.loads(line)
            batch.append(line_dict)
            i+=1
            if i == break_batch:
                print(batch_number)
                save_hdf_frame(batch, batch_number,p)
                batch_number +=1
                batch=[] # We have stored the previous batch on disk
                i = 0 

def save_hdf_frame(batch,batch_number,p):
    path = '/mnt/data/datasets/RedditComments/'
    name = 'CommentsMay2015' + '_size2p_'+ str(p) + '_' + str(batch_number)
    batch_frame = pd.DataFrame.from_dict(batch)
    # return as a dataframe
    # sort batchframe
    # 1. sort by link_id
    # 2. sort by created_utc
    a = batch_frame.sort_values(['link_id','created_utc'],ascending = [True, True],\
            inplace = True)
    batch_frame.to_hdf(path+name,key='a')



