import os
import random
from typing import List

import torch
from torch.utils.data import Dataset


class ToyTextDataset(Dataset):
    """A tiny dataset that either reads from a text file or generates synthetic sequences.

    Each sample is a sequence of token ids (integers). The tokenizer used should produce ids.
    """

    def __init__(self, texts: List[str], tokenizer, seq_len=32):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # encode all texts into ids and concatenate
        all_ids = []
        for t in texts:
            ids = tokenizer.encode(t)
            all_ids.extend(ids)
        # create sequences by sliding window
        self.seqs = []
        for i in range(0, max(1, len(all_ids) - seq_len), seq_len):
            chunk = all_ids[i:i + seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + \
                    [tokenizer.vocab.get(tokenizer.pad_token)] * \
                    (seq_len - len(chunk))
            self.seqs.append(chunk)
        if len(self.seqs) == 0:
            # fallback: generate random sequences from vocab
            vocab = list(tokenizer.vocab.values())
            for _ in range(100):
                self.seqs.append([random.choice(vocab)
                                 for _ in range(seq_len)])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        y = x.clone()
        return x, y


def make_toy_texts(num_sentences=200):
    # generate simple repetitive sentences to build a toy vocabulary
    words = ['hello', 'world', 'this', 'is', 'a', 'toy',
             'dataset', 'for', 'testing', 'gpt', 'model']
    texts = []
    for i in range(num_sentences):
        length = random.randint(5, 20)
        texts.append(' '.join(random.choices(words, k=length)))
    return texts


def get_dataloader(tokenizer, batch_size=8, seq_len=32):
    texts = make_toy_texts(500)
    tokenizer.build_vocab(texts, max_vocab=2000)
    ds = ToyTextDataset(texts, tokenizer, seq_len=seq_len)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
