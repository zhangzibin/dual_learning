
import util
import tensorflow as tf
import data_util
import dual_learning
from pprint import pprint

# parameters
params = util.AttrDict()
params.seq2seq = util.AttrDict(
        max_len_A = 21,
        max_len_B = 21,
        ckpt_path_AB = 'en_fr',
        ckpt_path_BA = 'fr_en',
        emb_dim = 1024,
        num_layers = 1,
        batch_size = 32,
        steps = 100000,
        beam_size = 2,
        alpha = 0.5,
        ratio_dual = 0.5
        )
params.lm_a = util.AttrDict(
        model_name = 'lm_a',
        load_model = '../cv2/A/epoch022_6.8524.model',
        train_dir = '../cv2/A',
        rnn_size = 650,
        highway_layers = 2,
        char_embed_size = 30,
        kernels = '[1,2,3,4,5,6,7]',
        kernel_features = '[50,100,150,200,200,200,200]',
        rnn_layers = 2,
        num_unroll_steps = 10,
        batch_size = 1,
        max_word_length = 65,
        seed = 3435,
        eos = '+'
        )
params.lm_b = util.AttrDict(
        model_name = 'lm_b',
        load_model = '../cv2/B/epoch024_7.7575.model',
        train_dir = '../cv2/B',
        rnn_size = 650,
        highway_layers = 2,
        char_embed_size = 30,
        kernels = '[1,2,3,4,5,6,7]',
        kernel_features = '[50,100,150,200,200,200,200]',
        rnn_layers = 2,
        num_unroll_steps = 10,
        batch_size = 1,
        max_word_length = 65,
        seed = 3435,
        eos = '+'
        )

# prepare data
datas = util.AttrDict()
# bilingual data for warm start
(datas.bi_idx2word_A, datas.bi_word2idx_A), (datas.bi_train_A, datas.bi_valid_A, datas.bi_test_A), datas.raw_test_A = \
        data_util.get_bi_data('../data/en_fr/', endwith='aa', max_len=params.seq2seq.max_len_A)
(datas.bi_idx2word_B, datas.bi_word2idx_B), (datas.bi_train_B, datas.bi_valid_B, datas.bi_test_B), datas.raw_test_B = \
        data_util.get_bi_data('../data/en_fr/', endwith='b', max_len=params.seq2seq.max_len_B)
# monolingual data for reinforcement learning
datas.mono_A = data_util.get_mono_data('../data/en/train.txt', vocab=datas.bi_word2idx_A, max_len=params.seq2seq.max_len_A)
datas.mono_B = data_util.get_mono_data('../data/fr/train.txt', vocab=datas.bi_word2idx_B, max_len=params.seq2seq.max_len_B)

# update vocab size to params for embedding layer
params.seq2seq.vocab_size_A = len(datas.bi_idx2word_A)
params.seq2seq.vocab_size_B = len(datas.bi_idx2word_B)
pprint(params)

# define model
dual_model = dual_learning.Dual(params)

# start training
dual_model.train(datas, params)
