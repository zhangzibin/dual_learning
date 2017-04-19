
def get_vocab(filename):
    with open(filename) as f:
        idx2word = ['<unk>']
        idx2word += [w.strip() for w in f.readlines()]
        word2idx = [(w.idx)for idx,w in enumerate(idx2word)]
    return idx2word, word2idx

def get_data(filename, word2idx, min_len, max_len):
    pass



