Quick run instructions for the repo

1. Create a Python virtual environment and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a tiny training demo (toy dataset) to verify everything works:

```bash
python train.py --epochs 1 --batch_size 8 --seq_len 32
```

3. Run the demo inference after training (or use the existing checkpoint file):

```bash
python inference.py --ckpt mini_gpt_checkpoint_new.pt --prompt "hello world" --length 20
```

Notes:
- The repository includes a saved trained checkpoint `mini_gpt_checkpoint.pt` (student-provided). A new demo checkpoint `mini_gpt_checkpoint_new.pt` will be created when you run `train.py`.
- The tokenizer used in the demo is a tiny whitespace-based tokenizer (`tokenizer.py`) intended for grading/demo purposes. It is not compatible with GPT-2 tokenization.
- If you need me to update `README.md` instead of creating `RUNNING.md`, tell me and I'll try again.
