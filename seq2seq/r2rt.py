import pandas as pd
import numpy as np
import tensorflow as tf
import blogs_data
import pdb

# Get data, in this example we use the collatz sequence
def collatz(n, max_steps=1000):
    seq = [n]
    steps = 0
    # Store the sequence as a list until it reaches one
    # The number of elements in this seq is its stopping time
    # It is conjectured that such a seq exists for all positive integers
    while n != 1 and steps < max_steps:
        if n % 2 == 0:
            n = n / 2
        else:
            n = 3 * n + 1
        steps += 1
        seq.append(n)
    return seq

# To mimic our objective we want to create a pandas dataframe which contains
# Column 1: N - random integer between 1 and M=1.000.000
# Column 2: Corresponding Collatz sequence
# Column 3: Context Vector: Manifold representation of the Collatz sequence
def make_collatz_data():
    x = np.random.randint(1000000,size=10000)
    df = pd.DataFrame(x)
    df['collatz'] = df[0].apply(collatz)
    df['length'] = df['collatz'].apply(len)
    train_len, test_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
    # Splits dataframe into train and test
    train, test = df.ix[:train_len-1], df.ix[train_len:train_len + test_len]
    return train, test, df

def load_blog_df():
    # Gets DataFrame, shuffles it and resets the index
    df = blogs_data.loadBlogs().sample(frac=1).reset_index(drop=True)
    vocab, reverse_vocab = blogs_data.loadVocab()
    train_len, test_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
    # Splits dataframe into train and test
    train, test = df.ix[:train_len-1], df.ix[train_len:train_len + test_len]
    return train, test, vocab, reverse_vocab, df

# Not good enough because sequence vary in length
class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    # n is the size of the batch
    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n
        return res['as_numbers'], res['gender']*3 + res['age_bracket'], res['length']

class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor+n-1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['gender']*3 + res['age_bracket'], res['length']

class BucketedDataIterator():
    def __init__(self, df, num_buckets = 5):
        df = df.sort_values('length').reset_index(drop=True)
        self.size = len(df)/num_buckets
        self.dfs = []
        # Put the shortest sequences in the first bucket etc
        for bucket in range(num_buckets):
            self.dfs.append(df.ix[bucket*self.size: (bucket+1)*self.size -1])
        self.num_buckets = num_buckets
        self.cursor = np.array([0]*num_buckets)
        self.shuffle()
        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, n):
        # if any of the buckets is full go to next epoch
        if np.any(self.cursor+n+1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0,self.num_buckets)
        res = self.dfs[i].ix[self.cursor[i]:self.cursor[i]+n-1]
        self.cursor[i] += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['collatz'].values[i]

        return x

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(vocab_size,\
                state_size = 64,\
                batch_size = 256,\
                num_classes = 6):
    # to reset put vocab as argument
    #vocab_size = len(vocab)
    reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None])  # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = x# tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder_with_default(1.0, [])

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1,state_size], initializer = tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell,\
                                                 rnn_inputs,\
                                                 sequence_length=seqlen,\
                                                 initial_state = init_state)
    # Add dropout
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    """
    Obtain the last relevant output. The best approach in the future will be to use:

        last_rnn_output = tf.gather_nd(rnn_outputs, tf.pack([tf.range(batch_size), seqlen-1], axis=1))

    which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :], but the
    gradient for this op has not been implemented as of this writing.

    The below solution works, but throws a UserWarning re: the gradient.
    """
    idx = tf.range(batch_size)*tf.shape(rnn_outputs)[1] + (seqlen - 1)
    last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]),idx)


    # Will be used for the colour later
    # Softmax layer
    #with tf.variable_scope('softmax'):
    #    W = tf.get_variable('W', [state_size, num_classes])
    #    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32),y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }

def build_seq2seq_graph(vocab_size,\
                        state_size = 64,\
                        batch_size = 256,\
                        num_classes = 6):
    # To reset put vocab in args
    # vocab_size = len(vocab)
    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size, None])
    keep_prob = tf.placeholder_with_default(1.0, [])

    # Tile the target indices
    #y_ = tf.tile(tf.expand_dims(y,1),[1, tf.shape(x)[1]])

    '''
    Create a mask that we will use for the cost function

        This mask is the same shape as x and y_, and is equal to 1 for all non-PAD time
        steps (where a prediction is made), and 0 for all PAD time steps (no pred -> no loss)
        The number 30, used when creating the lower_triangle_ones matrix, is the maximum
        sequence length in our dataset
    '''
    lower_triangular_ones = tf.constant(np.tril(np.ones([30,30])), dtype=tf.float32)
    seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, seqlen -1),\
                           [0,0], [batch_size, tf.reduce_max(seqlen)])
    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state',\
                                 [1, state_size],\
                                 initializer=tf.constant_initializer(0.0))

    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell,\
                                                 rnn_inputs,\
                                                 sequence_length=seqlen,\
                                                 initial_state=init_state)
    # Add dropout,
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])
    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(rnn_outputs, W) + b

    preds = tf.nn.softmax(logits)

    # To calculate the number correct, we want to count padded steps as incorrect
    correct = tf.cast(tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y_reshaped),tf.int32) *\
                tf.cast(tf.reshape(seqlen_mask, [-1]),tf.int32)

    # To calculate accuracy we want to divide by the number of non-padded time-steps,
    # rather than taking the mean
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))/tf.reduce_sum(tf.cast(seqlen, tf.float32))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped)
    loss = loss * tf.reshape(seqlen_mask, [-1])

    # To calculate average loss, we need to divide by number of non-padded time-steps,
    # rather than taking the mean
    loss = tf.reduce_sum(loss) / tf.reduce_sum(seqlen_mask)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }

def train_graph(graph, train, test, batch_size = 256, num_epochs = 10, iterator = PaddedDataIterator):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tr = iterator(train)
        te = iterator(test)

        step, accuracy = 0,0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seqlen']: batch[2], graph['dropout']: 0.6}

            accuracy_, _ = sess.run([graph['accuracy'],graph['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                # eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {graph['x']: batch[0], graph['y']: batch[1], graph['seqlen']: batch[2]}
                    accuracy_ = sess.run([graph['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0,0
                print('Accuracy after epoch', current_epoch, ' - tr:', tr_losses[-1], '- te:', te_losses[-1])

    return tr_losses, te_losses

if __name__ == "__main__":
    from time import time
    train, test, df = make_collatz_data()
    g = build_seq2seq_graph(len(df))
    t = time()
    tr_losses, te_losses = train_graph(g, train, test, num_epochs=100, iterator = BucketedDataIterator)
    print('Total time for 1 epoch with PaddedDataIterator:', time() - t)

