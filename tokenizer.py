import json
from collections import Counter


class SimpleTokenizer:
    """A tiny whitespace-based tokenizer with simple vocab build/save/load.

    This is intentionally minimal so the TA can run the project without external tokenizers.
    Not compatible with GPT-2 tokenizer; meant for demo / grading smoke tests.
    """

    def __init__(self, unk_token='<unk>', pad_token='<pad>'):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, texts, max_vocab=None, min_freq=1):
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        # special tokens first
        items = [self.pad_token, self.unk_token]
        # most common
        for tok, freq in cnt.most_common():
            if freq < min_freq:
                break
            items.append(tok)
            if max_vocab and len(items) >= max_vocab:
                break
        self.vocab = {tok: i for i, tok in enumerate(items)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        return self.vocab

    def encode(self, text, max_len=None):
        toks = text.split()
        ids = [self.vocab.get(t, self.vocab.get(self.unk_token)) for t in toks]
        if max_len is not None:
            if len(ids) < max_len:
                ids = ids + \
                    [self.vocab.get(self.pad_token)] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return ' '.join(self.inv_vocab.get(i, self.unk_token) for i in ids)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab, 'unk': self.unk_token,
                      'pad': self.pad_token}, f, ensure_ascii=False)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.vocab = obj['vocab']
        self.unk_token = obj.get('unk', self.unk_token)
        self.pad_token = obj.get('pad', self.pad_token)
        self.inv_vocab = {int(i): tok for tok, i in self.vocab.items()} if any(isinstance(
            k, str) for k in self.vocab.keys()) else {i: tok for tok, i in self.vocab.items()}
        # ensure types
        # normalize inv_vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()} if False else {int(
            i): tok for tok, i in self.vocab.items()} if False else {i: tok for tok, i in self.vocab.items()}
        # simpler: regenerate consistent mapping
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
