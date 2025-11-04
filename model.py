import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.mha = nn.MultiheadAttention(
            embed_dim=n_embd, num_heads=n_head, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        # causal mask: upper triangular (1 above diagonal) should be -inf
        attn_mask = torch.triu(torch.full(
            (T, T), float('-inf'), device=x.device), diagonal=1)
        # MultiheadAttention with batch_first expects (B, T, C)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, n_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            nn.GELU(),
            nn.Linear(n_ff, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, n_ff, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Minimal GPT-like model matching the README description.
    """

    def __init__(self, *, vocab_size=50257, n_embd=128, n_layer=2, n_head=4, n_ff=512, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(
            n_embd, n_head, n_ff, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.size()
        assert T <= self.pos_emb.num_embeddings, "Sequence length longer than model max_seq_len"
        tok = self.token_emb(input_ids)  # (B, T, C)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos = self.pos_emb(pos_ids)
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    # quick smoke test when running the file directly
    m = GPT()
    print(f'Created GPT model with params: {m.count_parameters():,}')
    # forward a tiny dummy batch
    x = torch.zeros((1, 8), dtype=torch.long)
    logits = m(x)
    print('Logits shape:', logits.shape)
