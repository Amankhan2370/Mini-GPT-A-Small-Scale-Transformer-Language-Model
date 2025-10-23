# Mini-GPT: Small-Scale Transformer Language Model

A compact, educational implementation of a GPT-style transformer trained from scratch in PyTorch – designed to demystify the core mechanics of large-scale language models and experiment with foundational NLP architecture on accessible hardware.

***

### Table of Contents

- Overview
- Model Architecture
- Dataset
- Training Details
- Results & Visualization
- Challenges Overcome
- Key Takeaways
- Future Work

***

## Overview

This project implements and trains a small-scale transformer, inspired by GPT models, highlighting each component’s role from token embedding to multi-head attention. The codebase and training workflow provide a practical, research-driven learning tool for understanding modern language models without inaccessible computational barriers.

***

## Model Architecture

- **Embedding Layer**: Transforms token IDs into 128-dim dense vectors.
- **Positional Encoding**: Learnable embeddings, supporting sequence length up to 512 tokens.
- **Transformer Encoder**: Two layers, each with:
  - Multi-head self-attention (4 heads)
  - Feedforward network (512 units)
  - Layer norm, residuals, dropout (0.1)
- **Output**: Linear layer projects to vocabulary logits.

**Parameters:** ~6.7M (suitable for experimentation, much smaller than GPT-3’s 175B).

***

## Dataset

- **Source**: Preprocessed text corpus for language modeling tasks.
- **Stats**:
  - Sequences: 50,191 (length: 512)
  - Vocabulary: 50,257 (GPT-2 tokenizer)
  - Train/Val Split: 90/10
  - Size: ~196MB

Task: Causal language modeling (predict next token in sequence).

***

## Training Details

- **Platform**: Google Colab
- **Hardware**: Tesla T4 (15GB VRAM)
- **Framework**: PyTorch 2.6.0
- **Batch Size**: 8 (optimized for memory)
- **Learning Rate**: $$5 \times 10^{-4}$$
- **Optimizer**: Adam, gradient clipping
- **Epochs**: 10
- **Loss**: Cross-entropy

Training involves classic forward/backward passes, optimizer steps, and per-epoch validation.

***

## Results & Visualization

Performance metrics demonstrate robust learning and generalization.

| Epoch | Train Loss | Val Loss | Train Perplexity | Val Perplexity |
|-------|------------|----------|------------------|----------------|
| 1     | 10.30      | 9.78     | 29,964           | 17,674         |
| 5     | 6.93       | 7.02     | 1,024            | 1,121          |
| 10    | 6.10       | 6.54     | 446              | 693            |

**Visuals:**  
<img ![Training Metrics](training_metrics.png)>

- Loss and perplexity curves show stable training, absence of overfitting, and strong model improvement.

***

## Challenges Overcome

- **Memory Management**: Reduced batch size to resolve CUDA out-of-memory errors with negligible slowdown.
- **Data Loading**: Streamlined file organization after early runtime issues.
- **Hardware Utilization**: Explicit code to confirm GPU runtime and monitor usage.

***

## Key Takeaways

- Built tangible intuition for transformer internals (self-attention, optimization, etc.).
- Developed skills in advanced PyTorch, model evaluation, GPU training, and troubleshooting.
- Understood the balance of architectural complexity versus hardware limits in real deep learning projects.

***

## Future Work

- Train for more epochs (20+), try deeper models (4–6 layers), and experiment with learning rate schedules.
- Implement mixed-precision or gradient accumulation for further memory efficiency.
- Add text generation and qualitative evaluation scripts.

***

### License

Specify your preferred open-source license here.

***

This project is an open, hands-on resource for anyone aiming to understand and experiment with modern NLP architectures on their own terms – feedback and collaboration welcome!

***

**For best results, add this to your `README.md`, update license and contact/credit sections as needed, and link to your code/demo.**

[1](https://github.com/othneildrew/Best-README-Template)
[2](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
[3](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
[4](https://github.com/mhucka/readmine)
[5](https://github.com/jehna/readme-best-practices)
[6](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
[7](https://www.makeareadme.com)
[8](https://www.reddit.com/r/github/comments/uulygm/what_are_some_really_nice_github_profile_readmes/)
[9](https://www.readme-templates.com)
[10](https://github.com/banesullivan/README)
