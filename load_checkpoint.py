"""
Script to inspect and attempt to load a checkpoint into the minimal GPT model defined in `model.py`.
Run: python load_checkpoint.py --ckpt mini_gpt_checkpoint.pt
"""
import argparse
import torch
import os
from model import GPT


def show_ckpt_keys(ckpt):
    if isinstance(ckpt, dict):
        print('Top-level keys in checkpoint:')
        for k in ckpt.keys():
            print(' -', k)
    else:
        print('Checkpoint is not a dict; likely a state_dict mapping of tensors')


def try_load(ckpt_path, device='cpu'):
    print('Loading checkpoint from', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    show_ckpt_keys(ckpt)

    # heuristics for common naming
    candidate = None
    if isinstance(ckpt, dict):
        for name in ('model_state_dict', 'state_dict', 'model'):
            if name in ckpt:
                candidate = ckpt[name]
                print(f"Found nested state dict under key '{name}'")
                break
    if candidate is None:
        # maybe ckpt is already a state_dict (mapping of tensors)
        candidate = ckpt

    # instantiate model with defaults that match README
    model = GPT()
    print('Model created with', model.count_parameters(), 'parameters')

    # try loading (non-strict) so we can see mismatches
    try:
        res = model.load_state_dict(candidate, strict=False)
        print('load_state_dict result:', res)
    except Exception as e:
        print('load_state_dict failed with exception:', e)

    # show top-level tensor keys (limit)
    if isinstance(candidate, dict):
        print('\nSample of state_dict keys (first 40):')
        for i, k in enumerate(candidate.keys()):
            if i >= 40:
                break
            print(' -', k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='mini_gpt_checkpoint.pt')
    parser.add_argument('--path', type=str, default='.')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    ckpt_path = os.path.join(args.path, args.ckpt)
    if not os.path.exists(ckpt_path):
        print('Checkpoint not found at', ckpt_path)
    else:
        try_load(ckpt_path, device=args.device)
