"""
Simple inference / generation script that loads a checkpoint and generates tokens autoregressively.
Usage:
  python inference.py --ckpt mini_gpt_checkpoint_new.pt --prompt "hello world" --length 20
"""
import argparse
import os

import torch
import torch.nn.functional as F

from model import GPT
from tokenizer import SimpleTokenizer


def top_k_logits(logits, k):
    if k == 0:
        return logits
    v, _ = torch.topk(logits, k)
    minv = v[..., -1, None]
    return torch.where(logits < minv, torch.full_like(logits, -1e10), logits)


def generate(model, tokenizer, prompt, length=20, temperature=1.0, top_k=0, device='cpu'):
    model.eval()
    ids = tokenizer.encode(prompt, max_len=tokenizer.max_len if hasattr(
        tokenizer, 'max_len') else None)
    # pad or truncate
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    generated = cur
    for _ in range(length):
        logits = model(generated)
        logits = logits[:, -1, :]
        logits = logits / max(1e-8, temperature)
        if top_k > 0:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_id), dim=1)
    out = generated[0].tolist()
    return tokenizer.decode(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='mini_gpt_checkpoint_new.pt')
    parser.add_argument('--prompt', default='hello world')
    parser.add_argument('--length', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print('Checkpoint not found:', args.ckpt)
        raise SystemExit(1)

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    tokenizer = SimpleTokenizer()
    if isinstance(ckpt, dict) and 'tokenizer_vocab' in ckpt:
        tokenizer.vocab = ckpt['tokenizer_vocab']
        tokenizer.inv_vocab = {i: tok for tok, i in tokenizer.vocab.items()}
    else:
        print('No tokenizer vocab found in checkpoint; using default toy tokenizer (may produce <unk>).')

    model = GPT(vocab_size=len(tokenizer.vocab) if tokenizer.vocab else 50257)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.to(args.device)

    out = generate(model, tokenizer, args.prompt,
                   length=args.length, device=args.device)
    print('Generated:', out)
