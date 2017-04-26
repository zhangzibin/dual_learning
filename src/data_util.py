import os
import math
import numpy as np
from random import sample
from collections import Counter

def get_vocab_from_file(filename):
    with open(filename) as f:
        idx2word = ['<pad>', '<unk>']
        idx2word += [w.strip() for w in f.readlines()]
        word2idx = dict([(w, idx) for idx,w in enumerate(idx2word)])
    return idx2word, word2idx

def get_tensors_from_file(filenames, word2idx, max_len):
    all_result = []
    for filename in filenames:
        with open(filename) as f:
            data = [line.strip().split(' ') for line in f.readlines()]
        result = []
        vocab = set(word2idx.keys())
        for line in data:
            line = [word2idx[w] if w in vocab else word2idx['<unk>'] for w in line[:max_len]]
            if len(line) < max_len:
                line += [word2idx['<pad>']] * (max_len-len(line))
            result.append(line)
        all_result.append(np.asarray(result))
    return all_result

def get_tensors_from_lines(lines, word2idx, max_len):
    result = []
    vocab = set(word2idx.keys())
    for line in lines:
        line = [word2idx[w] if w in vocab else word2idx['<unk>'] for w in line[:max_len]]
        if len(line) < max_len:
            line += [word2idx['<pad>']] * (max_len-len(line))
        result.append(line)
    return np.array(result)

def get_bi_data(path, endwith='en', max_len=50):
    vocab_file = os.path.join(path, 'vocab.'+endwith)
    data_files = [os.path.join(path, 'train.')+endwith,
            os.path.join(path, 'dev.')+endwith,
            os.path.join(path, 'test.')+endwith]
    idx2word, word2idx = get_vocab_from_file(vocab_file)
    tensors = get_tensors_from_file(data_files, word2idx, max_len)
    # the test set should keep raw text to calculate bleu
    with open(data_files[2]) as f:
        raw_testset = [line.strip() for line in f.readlines()]
    return (idx2word, word2idx), tensors, raw_testset

def get_mono_data(path, vocab, max_len=50):
    return get_tensors_from_file([path], vocab, max_len)[0]

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

def rand_batch_gen3(x, y, z, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T, [z[idx] for idx in sample_idx]

def decode(sequence, lookup, separator=' '):
    return separator.join([lookup[element] for element in sequence if element])

def corpus_bleu(hypotheses, references, smoothing=False, order=4, **kwargs):
    """
    Computes the BLEU score at the corpus-level between a list of translation hypotheses and references.
    With the default settings, this computes the exact same score as `multi-bleu.perl`.

    All corpus-based evaluation functions should follow this interface.

    :param hypotheses: list of strings
    :param references: list of strings
    :param smoothing: apply +1 smoothing
    :param order: count n-grams up to this value of n. `multi-bleu.perl` uses a value of 4.
    :param kwargs: additional (unused) parameters
    :return: score (float), and summary containing additional information (str)
    """
    total = np.zeros((4,))
    correct = np.zeros((4,))

    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        hyp_length += len(hyp)
        ref_length += len(ref)

        for i in range(order):
            hyp_ngrams = Counter(zip(*[hyp[j:] for j in range(i + 1)]))
            ref_ngrams = Counter(zip(*[ref[j:] for j in range(i + 1)]))

            total[i] += sum(hyp_ngrams.values())
            correct[i] += sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())

    if smoothing:
        total += 1
        correct += 1

    scores = correct / total

    score = math.exp(
        sum(math.log(score) if score > 0 else float('-inf') for score in scores) / order
    )

    bp = min(1, math.exp(1 - ref_length / hyp_length))
    bleu = 100 * bp * score

    return bleu, 'penalty={:.3f} ratio={:.3f}'.format(bp, hyp_length / ref_length)
