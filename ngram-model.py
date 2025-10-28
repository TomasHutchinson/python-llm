import random
from collections import defaultdict, Counter
import pickle
from tqdm import tqdm
import os
import unicodedata

from tokenize_data import apply_bpe

token_ids = []

model_name = ""
path = "models/" + (model_name + "/" if model_name else "")

with open(path + "token_ids.pkl", "rb") as f:
    token_ids = pickle.load(f)

token2id = token_ids[0]
id2token = token_ids[1]
tokenized_ids = token_ids[2]
merge_map = token_ids[3]

def generate_ngram_text_backoff(ngram_probs, id2token, n=10, seed=None, max_tokens=100):
    generated = list(seed)
    
    for _ in range(max_tokens):
        for k in range(n-1, 0, -1):
            context = tuple(generated[-k:])
            probs = ngram_probs.get(context)
            if probs:
                tokens, weights = zip(*probs.items())
                next_token = random.choices(tokens, weights=weights)[0]
                generated.append(next_token)
                break
        else:
            context = random.choice(list(ngram_probs.keys()))
            next_token = random.choices(list(ngram_probs[context].keys()), 
                                        weights=list(ngram_probs[context].values()))[0]
            generated.append(next_token)

    return ''.join([id2token[t].replace('</w>', ' ') for t in generated])

def seed_text_to_id(text):
    seed_tokens = []
    for word in text.strip().split():
        bpe_tokens = apply_bpe(word, merge_map)
        seed_tokens.extend(bpe_tokens)
    return [token2id[t] for t in seed_tokens]

def build_ngram_counts(tokenized_seqs, n=3):
    ngram_counts = defaultdict(Counter)
    for seq in tqdm(tokenized_seqs, desc="Building n-gram counts"):
        if len(seq) < n:
            continue
        for i in range(len(seq) - n + 1):
            context = tuple(seq[i:i + n - 1])
            target = seq[i + n - 1]
            ngram_counts[context][target] += 1
    return ngram_counts

def counts_to_probs(ngram_counts):
    ngram_probs = {}
    for context, counter in tqdm(ngram_counts.items(), desc="Converting counts to probs"):
        total = sum(counter.values())
        ngram_probs[context] = {token: count / total for token, count in counter.items()}
    return ngram_probs

def clean_text(text):
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace common encoding artifacts
    text = text.replace("â€”", "—")  # em-dash
    text = text.replace("â€“", "–")  # en-dash
    text = text.replace("â€˜", "'")  # left single quote
    text = text.replace("â€™", "'")  # right single quote
    text = text.replace("â€œ", '"')  # left double quote
    text = text.replace("â€\u201d", '"')  # right double quote
    text = text.replace("â€¢", "•")
    return text


n = 6
ngram_counts = build_ngram_counts(tokenized_ids, n=n)
ngram_probs = counts_to_probs(ngram_counts)

start_text = input("Enter start text: ") + " "
seed_ids = seed_text_to_id(start_text)
print(generate_ngram_text_backoff(ngram_probs, id2token, seed=tuple(seed_ids[-(n-1):]), n=max(n, len(seed_ids)), max_tokens=1000))
