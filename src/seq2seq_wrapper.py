import tensorflow as tf
import numpy as np
import sys

def sequence_loss(logits, targets, weights, average_across_timesteps=True, average_across_batch=True, reward=None):
    time_steps = tf.shape(targets)[0]
    batch_size = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.stack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.stack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=targets_)
    crossent = tf.reshape(crossent, tf.stack([time_steps, batch_size]))

    if reward is not None:
        crossent *= tf.stop_gradient(reward)

    log_perp = tf.reduce_sum(crossent * weights, 0)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, 0)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        batch_size = tf.shape(targets)[1]
        return cost / tf.cast(batch_size, tf.float32)
    else:
        return cost

class Seq2Seq(object):

    def __init__(self, xseq_len, yseq_len,
            xvocab_size, yvocab_size,
            emb_dim, num_layers, ckpt_path,
            lr=0.01, model_name='seq2seq_model'):

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

            # rewards for dual learning learning
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")


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
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

            # weighted loss
            #  TODO : add parameter hint

            # normalize logits for computing probilitic
            self.logits = []
            for output in self.decode_outputs:
                self.logits.append(tf.nn.softmax(output))


            loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
            self.decode_outputs = tf.stack(self.decode_outputs)
            # loss for seq2seq
            self.loss_seq2seq = sequence_loss(self.decode_outputs, self.labels, loss_weights)
            # loss for dual learning
            self.loss_dual = sequence_loss(self.decode_outputs, self.labels, loss_weights, reward=self.rewards)

            # train op to minimize the loss
            self.train_op_seq2seq = self.optimizer.minimize(self.loss_seq2seq)
            self.train_op_dual = self.optimizer.minimize(self.loss_dual)

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


    # prediction
    def predict(self, X):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = self.sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)

    # get the feed dictionary
    def get_feed_seq2seq(self, X, Y, keep_prob, rewards=None):
        feed_dict = {self.enc_ip[t]: X[t] for t in range(self.xseq_len)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        feed_dict[self.keep_prob] = keep_prob # dropout prob
        if rewards is not None:
            feed_dict[self.rewards] = rewards
        return feed_dict

    # evaluate 'num_batches' batches
    def eval_batches(self, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    def eval_step(self, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed_seq2seq(batchX, batchY, keep_prob=1.)
        loss_v, dec_op_v = self.sess.run([self.loss_seq2seq, self.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    def train_step_seq2seq(self, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed_seq2seq(batchX, batchY, keep_prob=0.5)
        _, loss_v = self.sess.run([self.train_op_seq2seq, self.loss_seq2seq], feed_dict)
        return loss_v

    def update_step_dual(self, batchX, batchY, rewards):
        # build feed
        feed_dict = self.get_feed_seq2seq(batchX, batchY, keep_prob=0.5, rewards=rewards)
        _, loss_v = self.sess.run([self.train_op_dual, self.loss_dual], feed_dict)
        return loss_v

    # log probability of X->Y
    def test(self, X, Y):
        feed_dict = self.get_feed_seq2seq(X, Y, keep_prob=1.)
        logits = self.sess.run(self.logits, feed_dict)
        logits = np.stack(logits).transpose([1,0,2])
        Y = Y.transpose([1,0])
        probs = np.zeros(Y.shape)
        for idx in range(Y.shape[0]):
            for jdx in range(Y.shape[1]):
                probs[idx, jdx] = logits[idx, jdx, Y[idx,jdx]]
        return np.average(probs, axis=1)

