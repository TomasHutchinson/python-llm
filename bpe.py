from collections import Counter, defaultdict
from tqdm import tqdm

def get_initial_vocab(corpus):
    vocab = Counter()
    for text in tqdm(corpus, desc="Current Test"):
        for word in text.strip().split():
            #split word into characters + </w>
            symbols = ' '.join(list(word)) + ' </w>'
            vocab[symbols] += 1
    return vocab

def get_initial_vocab_stream(dataset, max_examples=None):
    vocab = Counter()
    dataset_iter = dataset
    if max_examples is not None:
        dataset_iter = tqdm(dataset, total=max_examples)
    else:
        dataset_iter = tqdm(dataset)
    
    for i, example in enumerate(dataset_iter):
        text = example['text']
        for word in text.strip().split():
            symbols = ' '.join(list(word)) + ' </w>'
            vocab[symbols] += 1
        if max_examples is not None and i + 1 >= max_examples:
            break
    return vocab

def get_pair_stats(vocab):
    """Count symbol pairs across all words"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split(' ')
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, vocab):
    """Merge the most frequent pair in vocab"""
    new_vocab = {}
    old = ' '.join(pair)
    new = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(old, new)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe_stream(corpus, num_entries=100, num_merges=100):
    vocab = get_initial_vocab_stream(corpus, num_entries)
    merges = []

    for _ in tqdm(range(num_merges), desc="BPE merges"):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        vocab = merge_pair(best, vocab)
        merges.append(best)

    return vocab, merges
