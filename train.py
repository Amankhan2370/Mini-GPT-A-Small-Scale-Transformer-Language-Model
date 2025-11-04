"""
Minimal training script for the toy GPT model.
Run locally in an environment with PyTorch installed.
Example:
  python train.py --epochs 2 --batch_size 8 --save_path ckpt.pt
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import GPT
from tokenizer import SimpleTokenizer
from data import get_dataloader


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.cpu else 'cpu')
    print('Using device:', device)

    tokenizer = SimpleTokenizer()
    dataloader = get_dataloader(
        tokenizer, batch_size=args.batch_size, seq_len=args.seq_len)

    model = GPT(vocab_size=len(tokenizer.vocab), n_embd=args.n_embd, n_layer=args.n_layer,
                n_head=args.n_head, n_ff=args.n_ff, max_seq_len=args.seq_len)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)  # (B, T, V)
            # reshape for loss: (B*T, V) vs (B*T)
            B, T, V = logits.size()
            loss = loss_fn(logits.view(B*T, V), yb.view(B*T))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        print(
            f'Epoch {epoch+1} finished, loss: {running_loss/len(dataloader):.4f}')
        # save checkpoint
        ckpt = {'model_state_dict': model.state_dict(), 'args': vars(
            args), 'tokenizer_vocab': tokenizer.vocab}
        torch.save(ckpt, args.save_path)
        print('Saved checkpoint to', args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--save_path', type=str,
                        default='mini_gpt_checkpoint_new.pt')
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_ff', type=int, default=512)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    train(args)
