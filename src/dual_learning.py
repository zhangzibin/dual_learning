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


    # a step of dual learning, we use A->B as varible names here, actually it supports B->A too
    # TODO: support beam search to generate K middle sentence B
    def train_step_dual(self, datas, params, mono_a_gen, seq2seq_ab, seq2seq_ba, lookup_b, lm_b):
        # translate A-->B_mid
        batch_A = mono_a_gen.__next__()[0]
        output = seq2seq_ab.predict(batch_A)
        batch_B_mid_raw = []
        for oi in output:
            batch_B_mid_raw.append(data_util.decode(sequence=oi, lookup=lookup_b, separator=' '))
        batch_B_mid = data_util.get_tensors_from_lines(batch_B_mid_raw, datas.bi_word2idx_B, params.seq2seq.max_len_B).T
        # evaluation B_mid on language model
        r1 = lm_b.evaluate(batch_B_mid_raw) # TODO: a better reward!!
        # log probability of A recovered from B_mid
        r2 = self.seq2seq_ba.test(batch_B_mid, batch_A)

        # update parameter using rewards
        total_r = params.seq2seq.alpha*r1 + (1-params.seq2seq.alpha)*r2
        loss_ab = seq2seq_ab.update_step_dual(batch_A, batch_B_mid, total_r)
        alpha_ = np.array([(1-params.seq2seq.alpha)]*params.seq2seq.batch_size)
        loss_ba = seq2seq_ba.update_step_dual(batch_B_mid, batch_A, alpha_)
        return loss_ab, loss_ba

    # test batches using bleu
    def test_batches(self, seq2seq_ab, test_batch_ab_gen, lookup_b, num_batches=32):
        all_B_predict = []
        all_B_raw = []
        for i in range(num_batches):
            batchA, batchB, rawB = test_batch_ab_gen.__next__()
            output = seq2seq_ab.predict(batchA)
            for oi,ooii in zip(output,rawB):
                oi = data_util.decode(sequence=oi, lookup=lookup_b, separator=' ')
                if len(oi) > 0:
                    all_B_predict.append(oi)
                    all_B_raw.append(ooii)
        print(1, all_B_raw[0])
        print(2, all_B_predict[0])
        print()
        if len(all_B_predict) > 0:
            bleu, bleu_log = data_util.corpus_bleu(all_B_predict, all_B_raw)
        else:
            bleu = 0.
        return bleu

    def test(self, datas, params):
        test_batch_ab_gen = data_util.rand_batch_gen3(datas.bi_test_A, datas.bi_test_B, datas.raw_test_B, self.params.seq2seq.batch_size)
        test_batch_ba_gen = data_util.rand_batch_gen3(datas.bi_test_B, datas.bi_test_A, datas.raw_test_A, self.params.seq2seq.batch_size)

        bleu_ab = self.test_batches(self.seq2seq_ab, test_batch_ab_gen, datas.bi_idx2word_B, num_batches=1)
        print('AB test bleu : {0:.6f}'.format(bleu_ab))
        bleu_ba = self.test_batches(self.seq2seq_ba, test_batch_ba_gen, datas.bi_idx2word_A, num_batches=1)
        print('BA test bleu : {0:.6f}'.format(bleu_ba))

    def train(self, datas, params):
        train_batch_ab_gen = data_util.rand_batch_gen(datas.bi_train_A, datas.bi_train_B, self.params.seq2seq.batch_size)
        train_batch_ba_gen = data_util.rand_batch_gen(datas.bi_train_B, datas.bi_train_A, self.params.seq2seq.batch_size)

        valid_batch_ab_gen = data_util.rand_batch_gen(datas.bi_valid_A, datas.bi_valid_B, self.params.seq2seq.batch_size)
        valid_batch_ba_gen = data_util.rand_batch_gen(datas.bi_valid_B, datas.bi_valid_A, self.params.seq2seq.batch_size)


        mono_a_gen = data_util.rand_batch_gen(datas.mono_A, datas.mono_A, self.params.seq2seq.batch_size)
        mono_b_gen = data_util.rand_batch_gen(datas.mono_B, datas.mono_B, self.params.seq2seq.batch_size)

        # run M epochs
        for i in range(self.params.seq2seq.steps):
            rand_point = random.uniform(0, 1)

            # if rand number larger than ratio of dual learning, using normal seq2seq pair to train model
            if rand_point > params.seq2seq.ratio_dual:
                train_loss_AB = self.seq2seq_ab.train_step_seq2seq(train_batch_ab_gen)
                train_loss_BA = self.seq2seq_ba.train_step_seq2seq(train_batch_ba_gen)
                if i%10 ==0:
                    print('step %d, seq2seq, AB train loss: %.6f' % (i, train_loss_AB))
                    print('step %d, seq2seq, BA train loss: %.6f' % (i, train_loss_BA))
            # using dual learning to train model
            else:
                train_loss_AB, train_loss_BA = self.train_step_dual(datas, params, mono_a_gen, self.seq2seq_ab, self.seq2seq_ba, datas.bi_idx2word_B, self.lm_b)
                if i%10 ==0:
                    print('step %d, dual_ab, AB train loss: %.6f, BA train loss: %.6f' % (i, train_loss_AB, train_loss_BA))
                train_loss_BA, train_loss_AB = self.train_step_dual(datas, params, mono_b_gen, self.seq2seq_ba, self.seq2seq_ab, datas.bi_idx2word_A, self.lm_a)
                if i%10 ==0:
                    print('step %d, dual_ba, AB train loss: %.6f, BA train loss: %.6f' % (i, train_loss_AB, train_loss_BA))

            if i and i%10 == 0: # TODO : make this tunable by the user
                # evaluate to get validation loss
                val_loss_AB = self.seq2seq_ab.eval_batches(valid_batch_ab_gen, 16) # TODO : and this
                val_loss_BA = self.seq2seq_ba.eval_batches(valid_batch_ba_gen, 16) # TODO : and this
                # print stats
                print('AB val loss : {0:.6f}'.format(val_loss_AB))
                print('BA val loss : {0:.6f}'.format(val_loss_BA))
                sys.stdout.flush()
                self.test(datas, params)

