import tensorflow as tf
import numpy as np
import sys
import os

import lm
from lm_data_reader import Vocab, DataReader

class LM(object):
    def __init__(self, FLAGS):
        self.char_vocab = Vocab.load(os.path.join(FLAGS.train_dir, 'char_vocab.pkl'))
        self.word_vocab = Vocab.load(os.path.join(FLAGS.train_dir, 'word_vocab.pkl'))
        actual_max_word_length = 0
        for w in self.word_vocab._index2token:
            actual_max_word_length = max(actual_max_word_length, len(w) + 2)
        FLAGS.max_word_length = actual_max_word_length

        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                # tensorflow seed must be inside graph
                tf.set_random_seed(FLAGS.seed)
                np.random.seed(seed=FLAGS.seed)

                with tf.variable_scope("Model"):
                    self.m = lm.inference_graph(
                            char_vocab_size=self.char_vocab.size,
                            word_vocab_size=self.word_vocab.size,
                            char_embed_size=FLAGS.char_embed_size,
                            batch_size=FLAGS.batch_size,
                            num_highway_layers=FLAGS.highway_layers,
                            num_rnn_layers=FLAGS.rnn_layers,
                            rnn_size=FLAGS.rnn_size,
                            max_word_length=FLAGS.max_word_length,
                            kernels=eval(FLAGS.kernels),
                            kernel_features=eval(FLAGS.kernel_features),
                            num_unroll_steps=FLAGS.num_unroll_steps,
                            dropout=0)
                    self.m.update(lm.loss_graph(self.m.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))
                    global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

                saver = tf.train.Saver()
                saver.restore(self.sess, FLAGS.load_model)
                print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())
        self.FLAGS = FLAGS

    def load_data(self, lines, min_len=10):

        word_tokens = []
        char_tokens = []
        for line in lines:
            line = line.strip()
            line = line.replace('}', '').replace('{', '').replace('|', '')
            line = line.replace('<unk>', ' | ')
            if self.FLAGS.eos:
                line = line.replace(self.FLAGS.eos, '')
            line = line.split()
            if len(line) < min_len:
                line += ['|']*(min_len-len(line))

            for word in line:
                if len(word) > self.FLAGS.max_word_length - 2:  # space for 'start' and 'end' chars
                    word = word[:self.FLAGS.max_word_length-2]

                word_array = self.word_vocab.get(word, self.word_vocab.get('|'))
                word_tokens.append(word_array)

                char_array = [self.char_vocab.get(c) for c in '{' + word + '}']
                char_array = [c for c in char_array if c is not None]
                char_tokens.append(char_array)

            if self.FLAGS.eos:
                word_tokens.append(self.word_vocab.get(self.FLAGS.eos))
                char_array = [self.char_vocab.get(c) for c in '{' + self.FLAGS.eos + '}']
                char_tokens.append(char_array)

        # now we know the sizes, create tensors
        word_tensors = {}
        char_tensors = {}
        word_tensors = np.array(word_tokens, dtype=np.int32)
        char_tensors = np.zeros([len(char_tokens), self.FLAGS.max_word_length], dtype=np.int32)
        for i, char_array in enumerate(char_tokens):
            char_tensors[i,:len(char_array)] = char_array

        return word_tensors, char_tensors

    def evaluate(self, lines):
        probs = []
        for line in lines:
            rnn_state = self.sess.run(self.m.initial_rnn_state)
            word_tensors, char_tensors = self.load_data([line])
            test_reader = DataReader(word_tensors, char_tensors,
                                      self.FLAGS.batch_size, self.FLAGS.num_unroll_steps)
            prob = []
            for x, y in test_reader.iter():
                logits, rnn_state = self.sess.run([
                    self.m.logits,
                    self.m.final_rnn_state
                ], {
                    self.m.input  : x,
                    self.m.targets: y,
                    self.m.initial_rnn_state: rnn_state
                })
            for logit, yy in zip(logits, y[0,:]):
                prob.append(logit[0,yy])
            prob = np.mean(prob)
            probs.append(prob)

        return np.array(probs)

