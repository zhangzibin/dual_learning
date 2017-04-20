
import util
import tensorflow as tf
import data_util
import dual_learning
from pprint import pprint

params = util.AttrDict()
params.seq2seq = util.AttrDict(
        max_len_A = 21,
        max_len_B = 21,
        ckpt_path_AB = 'en_fr',
        ckpt_path_BA = 'fr_en',
        emb_dim = 16,
        num_layers = 1,
        batch_size = 32,
        steps = 100000
        )

datas = util.AttrDict()
datas.bi_word2idx_A, (datas.bi_train_A, datas.bi_valid_A, datas.bi_test_A) = \
        data_util.get_bi_data('../data/en_fr/', endwith='en', max_len=params.seq2seq.max_len_A)
datas.bi_word2idx_B, (datas.bi_train_B, datas.bi_valid_B, datas.bi_test_B) = \
        data_util.get_bi_data('../data/en_fr/', endwith='fr', max_len=params.seq2seq.max_len_B)

params.seq2seq.vocab_size_A = len(datas.bi_word2idx_A)
params.seq2seq.vocab_size_B = len(datas.bi_word2idx_B)
pprint(params)

dual_model = dual_learning.Dual(params)
dual_model.train(datas)
