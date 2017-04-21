import os
import numpy as np
from random import sample

def get_vocab_from_file(filename):
    with open(filename) as f:
        idx2word = ['<pad>', '<unk>']
        idx2word += [w.strip() for w in f.readlines()]
        word2idx = dict([(w, idx) for idx,w in enumerate(idx2word)])
    return idx2word, word2idx

def get_tensors(filenames, word2idx, max_len):
    all_result = []
    for filename in filenames:
        with open(filename) as f:
            data = [line.split(' ') for line in f.readlines()]
        result = []
        vocab = set(word2idx.keys())
        for line in data:
            line = [word2idx[w] if w in vocab else word2idx['<unk>'] for w in line[:max_len]]
            if len(line) < max_len:
                line += [word2idx['<pad>']] * (max_len-len(line))
            result.append(line)
        all_result.append(np.asarray(result))
    return all_result

def get_bi_data(path, endwith='en', max_len=50):
    vocab_file = os.path.join(path, 'vocab.'+endwith)
    data_files = [os.path.join(path, 'train.')+endwith,
            os.path.join(path, 'dev.')+endwith,
            os.path.join(path, 'test.')+endwith]
    idx2word, word2idx = get_vocab_from_file(vocab_file)
    tensors = get_tensors(data_files, word2idx, max_len)
    return word2idx, tensors

def get_mono_data(path, vocab, max_len=50):
    return get_tensors([path], vocab, max_len)[0]

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T
