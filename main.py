from mlp import *
from WordColour import *
import tensorflow as tf

if __name__ == '__main__':

    data = LexiChromeVocab()
    test_X = data['GloVe Vector']
    test_ids = data['Colour Vector']

    sess = tf.Session()

    model = MLP(sess, data)

    sess.run(tf.initialize_all_variables())

    train(sess, model, data, show_plot = False)

    write_predictions(sess, model, test_X, test_ids,
                     'pred_nhid_%s_lr_%s_epochs_%s.txt' % (model.n_hidden,
                                                           model.lr,
                                                           model.epochs))
    sess.close()

