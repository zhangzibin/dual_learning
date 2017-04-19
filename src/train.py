
import tensorflow as tf
import data_util


# path of bilingual pairwise data A<->B
# vocab can be any word not in training data, as training data is small
bi_A_vocab = '../data/en_fr/vocab.en'
bi_A_files = ('../data/en_fr/train.en', '../data/en_fr/valid.en', '../data/en_fr/test.en')
bi_B_vocab = '../data/en_fr/vocab.fr'
bi_B_files = ('../data/en_fr/train.fr', '../data/en_fr/valid.fr', '../data/en_fr/test.fr')

# path of monolingual data
mo_A_file = '../data/en/train.txt'
mo_B_file = '../data/fr/train.txt'

A_idx2word, A_word2idx = data_util.get_vocab(bi_A_vocab)
B_idx2word, B_word2idx = data_util.get_vocab(bi_B_vocab)
print(len(A_idx2word))
