import tensorflow as tf
import numpy as np
import sys


class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.0001, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.model_name = model_name

        # build thy graph
        #  attach any part of the graph that needs to be exposed, to the self
        # def __graph__():
        with tf.variable_scope(self.model_name) as scope:

            # placeholders
            # tf.reset_default_graph()
            #  encoder inputs : list of indices of length xseq_len
            self.enc_ip = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(xseq_len) ]

            #  labels that represent the real outputs
            self.labels = [ tf.placeholder(shape=[None,],
                            dtype=tf.int64,
                            name='ei_{}'.format(t)) for t in range(yseq_len) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.dec_ip = [ tf.zeros_like(self.enc_ip[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


            # Basic LSTM cell wrapped in Dropout Wrapper
            self.keep_prob = tf.placeholder(tf.float32)
            # define the basic cell
            basic_cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                    tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(emb_dim, state_is_tuple=True),
                    output_keep_prob=self.keep_prob)
            # stack cells together : n layered model
            stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


            # for parameter sharing between training model
            #  and testing model
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.enc_ip,self.dec_ip, stacked_lstm,
                                                    xvocab_size, yvocab_size, emb_dim)
                # share parameters
                scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input
                #  to the next timestep
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.enc_ip, self.dec_ip, stacked_lstm, xvocab_size, yvocab_size,emb_dim,
                    feed_previous=True)

            # now, for training,
            #  build loss function

            # weighted loss
            #  TODO : add parameter hint
            loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)
            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        sys.stdout.write('<log> Building Graph ')
        # build comput graph
        # __graph__()
        sys.stdout.write('</log>\n')

        self.saver = tf.train.Saver()
        # create a session
        self.sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.sess.run(tf.global_variables_initializer())
        # return to user
        # return sess
