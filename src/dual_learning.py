import tensorflow as tf
import numpy as np
import sys
import data_util
import seq2seq_wrapper
import lm_wrapper
import random

class Dual(object):
    def __init__(self, params):
        self.params = params
        self.lm_a = lm_wrapper.LM(params.lm_a)
        self.lm_b = lm_wrapper.LM(params.lm_b)

        '''
        lines = ['This is a test .', 'This is another test .']
        self.lm_a.evaluate(lines)
        lines = ['This is a test .', 'This is another test .']
        self.lm_b.evaluate(lines)
        '''

        self.seq2seq_ab = seq2seq_wrapper.Seq2Seq(xseq_len=params.seq2seq.max_len_A,
                                       yseq_len=params.seq2seq.max_len_B,
                                       xvocab_size=params.seq2seq.vocab_size_A,
                                       yvocab_size=params.seq2seq.vocab_size_B,
                                       ckpt_path=params.seq2seq.ckpt_path_AB,
                                       emb_dim=params.seq2seq.emb_dim,
                                       num_layers=params.seq2seq.num_layers,
                                       model_name='seq2seq_ab')
        self.seq2seq_ba = seq2seq_wrapper.Seq2Seq(xseq_len=params.seq2seq.max_len_B,
                                       yseq_len=params.seq2seq.max_len_A,
                                       xvocab_size=params.seq2seq.vocab_size_B,
                                       yvocab_size=params.seq2seq.vocab_size_A,
                                       ckpt_path=params.seq2seq.ckpt_path_BA,
                                       emb_dim=params.seq2seq.emb_dim,
                                       num_layers=params.seq2seq.num_layers,
                                       model_name='seq2seq_ba')


    # get the feed dictionary
    def get_feed_seq2seq(self, X, Y, model, keep_prob):
        feed_dict = {model.enc_ip[t]: X[t] for t in range(model.xseq_len)}
        feed_dict.update({model.labels[t]: Y[t] for t in range(model.yseq_len)})
        feed_dict[model.keep_prob] = keep_prob # dropout prob
        return feed_dict

    # evaluate 'num_batches' batches
    def eval_batches(self, model, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(model, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    def eval_step(self, model, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed_seq2seq(batchX, batchY, model, keep_prob=1.)
        loss_v, dec_op_v = model.sess.run([model.loss, model.decode_outputs_test], feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    def train_step(self, model, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed_dict = self.get_feed_seq2seq(batchX, batchY, model, keep_prob=0.5)
        _, loss_v = model.sess.run([model.train_op, model.loss], feed_dict)
        return loss_v

    def train(self, datas):
        train_batch_ab_gen = data_util.rand_batch_gen(datas.bi_train_A, datas.bi_train_B, self.params.seq2seq.batch_size)
        train_batch_ba_gen = data_util.rand_batch_gen(datas.bi_train_B, datas.bi_train_A, self.params.seq2seq.batch_size)

        valid_batch_ab_gen = data_util.rand_batch_gen(datas.bi_valid_A, datas.bi_valid_B, self.params.seq2seq.batch_size)
        valid_batch_ba_gen = data_util.rand_batch_gen(datas.bi_valid_B, datas.bi_valid_A, self.params.seq2seq.batch_size)

        ratio_dual = 0.5
        # run M epochs
        for i in range(self.params.seq2seq.steps):
            rand_point = random.uniform(0, 1)

            # if rand number larger than ratio of dual learning, using normal seq2seq pair to train model
            if rand_point > ratio_dual:
                train_loss_AB = self.train_step(self.seq2seq_ab, train_batch_ab_gen)
                train_loss_BA = self.train_step(self.seq2seq_ba, train_batch_ba_gen)
                if i%10 ==0:
                    print('step %d, seq2seq, AB train loss : %.6f' % (i, train_loss_AB))
                    print('step %d, seq2seq, BA train loss : %.6f' % (i, train_loss_BA))
            # using dual learning to train model
            else:
                # translate A-->B_mid
                if i%10 ==0:
                    print('step %d, dual, AB train loss : %.6f')
                    print('step %d, dual, BA train loss : %.6f')

            if i and i%100 == 0: # TODO : make this tunable by the user
                # evaluate to get validation loss
                val_loss_AB = self.eval_batches(self.seq2seq_ab, valid_batch_ab_gen, 16) # TODO : and this
                val_loss_BA = self.eval_batches(self.seq2seq_ba, valid_batch_ba_gen, 16) # TODO : and this
                # print stats
                print('AB val loss : {0:.6f}'.format(val_loss_AB))
                print('BA val loss : {0:.6f}'.format(val_loss_BA))
                sys.stdout.flush()

