# Thor H. Jonsson
# Example of a seq2seq autoencoder in Tensorflow
import tensorflow as tf
import numpy as np
import pandas as pd

# Get data, in this example we use the collatz sequence
def collatz(n, max_steps=100):
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
def make_data():
    x = np.random.randint(100,size=10) + 1 # We add 1 so we never get 0
    df = pd.DataFrame(x)
    df['collatz'] = df[0].apply(collatz)
    input_sequences = df['collatz'].tolist()
    target_sequences = input_sequences # Sequence Autoencoder
    return input_sequences, target_sequences
"""
 We serialize the sequences by converting it into a sequence example
 Input: Pandas DataFrame with the collatz sequences in a column called 'collatz'
 Output: tf.train.SequenceExample()
 From the API: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/example/example.proto
"""
def make_example(input_sequence, target_sequence):
    ex = tf.train.SequenceExample()
    # The max length of our sequence is a feature that we have to include
    sequence_length = 100
    ex.context.feature['length'].int64_list.value.append(sequence_length)
    # Feature lists for the sequential features of our example
    fl_input = ex.feature_lists.feature_list['input_tokens']
    fl_target = ex.feature_lists.feature_list['target_tokens']
    for input_token, target_token in zip(input_sequence, target_sequence):
        fl_input.feature.add().int64_list.value.append(input_token)
        fl_target.feature.add().int64_list.value.append(target_token)
    return ex

def serialize(input_sequences, target_sequences, fnames = 'collatz_data'):
    # Write all examples into a TFRecords file
    writer = tf.python_io.TFRecordWriter(fname)
    for sequence, label_sequence in zip(input_sequences, target_sequences):
        ex = make_example(sequence, label_sequence)
        writer.write(ex.SerializeToString())
    writer.close()

# Input: Our sequences in tensor form
# Output: Batch of size 5
def get_batch(input_seq,batch_size = 5):
    batched_data = tf.train.batch(
        tensors = [input_seq],
        batch_size=batch_size,
        dynamic_pad = True,
        name ='collatz_batch'
    )


def get_examples(fnames = 'collatz_data'):
    filename_queue = tf.train.string_input_producer([fnames], num_epochs=1,shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Define how to parse the example
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "input_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "target_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    context = tf.contrib.learn.run_n(context_parsed,n=1,feed_dict=None)
    sequence = tf.contrib.learn.run_n(sequence_parsed, n=1, feed_dict=None)

def build_model():
    cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)

    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length = X_lengths,
        inputs=X)


