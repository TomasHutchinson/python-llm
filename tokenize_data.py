import pickle
from tqdm import tqdm

from bpe import train_bpe_stream

load = False
tokenize = False
subset_size = 10000
num_merges = 5000


def apply_bpe(word, merge_map):
    symbols = list(word) + ['</w>']
    while True:
        pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
        merge_found = False
        for pair in pairs:
            if pair in merge_map:
                i = pairs.index(pair)
                symbols = symbols[:i] + [merge_map[pair]] + symbols[i+2:]
                merge_found = True
                break
        if not merge_found:
            break
    return symbols

def tokenize_dataset(dataset, merges, max_examples=None):
    merge_map = {pair: ''.join(pair) for pair in merges}
    tokenized_dataset = []
    for i, example in enumerate(tqdm(dataset, total=max_examples)):
        words = example['text'].split()
        tokens = []
        for word in words:
            tokens.extend(apply_bpe(word, merge_map))
        tokenized_dataset.append(tokens)
        if max_examples is not None and i + 1 >= max_examples:
            break
    return tokenized_dataset

if load:
    from datasets_load import dataset
    vocab, merges = train_bpe_stream(dataset, num_entries=subset_size, num_merges=num_merges)


    with open("bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

if tokenize:
    from datasets_load import dataset
    with open("bpe_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("bpe_merges.pkl", "rb") as f:
        merges = pickle.load(f)
    
    tokenized_subset = tokenize_dataset(dataset, merges, max_examples=subset_size)

    tokens = set()
    for word in vocab.keys():
        tokens.update(word.split(' '))


    for merge in merges:
        tokens.add(''.join(merge))

    token2id = {token: idx for idx, token in enumerate(sorted(tokens))}
    id2token = {idx: token for token, idx in token2id.items()}
    tokenized_ids = [[token2id[t] for t in word] for word in tokenized_subset]
    merge_map = {pair: ''.join(pair) for pair in merges}

    token_id = [token2id, id2token, tokenized_ids, merge_map]
    with open("token_ids.pkl", "wb") as f:
        pickle.dump(token_id, f)
    