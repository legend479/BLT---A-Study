# BLT: Byte Latent Transformer - Implementation Study

A comparative study implementing BLT (Byte Latent Transformer) and a baseline character-level transformer for string reversal tasks.

## Overview

This project implements two sequence-to-sequence models:
1. **BLT Model**: Uses entropy-based byte patching with n-gram hashing for efficient byte-level processing
2. **Baseline Model**: Standard character-level encoder-decoder transformer

Both models are trained and evaluated on a string reversal task to demonstrate BLT's efficiency advantages.

## Directory Structure

```
BLT---A-Study/
├─ data/
│  ├─ train.csv              # Training data (input, target pairs)
│  ├─ test.csv               # Test data
│  └─ processed/             # Preprocessed data (auto-generated)
├─ src/
│  ├─ patcher.py             # Entropy-based byte patching
│  ├─ tokenizer.py           # Character tokenizer with special tokens
│  ├─ baseline_model.py      # Baseline transformer model
│  ├─ blt_model.py           # BLT model implementation
│  ├─ dataset.py             # Data preprocessing and loading
│  ├─ train.py               # Training script
│  ├─ infer.py               # Interactive inference
│  └─ eval.py                # Evaluation and metrics
├─ checkpoints/              # Model checkpoints (auto-generated)
├─ predictions/              # Prediction outputs (auto-generated)
├─ docs/
│  ├─ REPORT.md              # Detailed analysis report
│  └─ LMA Assignment 1.pdf   # Assignment specification
├─ requirements.txt
└─ README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd BLT---A-Study

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models

**Train BLT Model:**
```bash
python src/train.py --mode blt --epochs 10 --batch_size 32 --lr 1e-3
```

**Train Baseline Model:**
```bash
python src/train.py --mode char --epochs 10 --batch_size 32 --lr 1e-3
```

### 2. Evaluate Models

**Evaluate BLT:**
```bash
python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv
```

**Evaluate Baseline:**
```bash
python src/eval.py --mode char --checkpoint checkpoints/best_char_model.pt --test_csv data/test.csv
```

### 3. Interactive Inference

**BLT Model:**
```bash
python src/infer.py --mode blt --checkpoint checkpoints/best_blt_model.pt
```

**Baseline Model:**
```bash
python src/infer.py --mode char --checkpoint checkpoints/best_char_model.pt --beam_width 5
```

## Key Features

### BLT Model
- **Entropy-based Patching**: Dynamically segments input bytes based on Shannon entropy
- **N-gram Hashing**: Efficient representation using 1-gram, 2-gram, and 3-gram hashes
- **Patch Embeddings**: Separate embedding tables for each n-gram size
- **Reduced Sequence Length**: Processes patches instead of individual characters

### Baseline Model
- **Character-level Processing**: Standard token-by-token encoding
- **Transformer Architecture**: Encoder-decoder with multi-head attention
- **Positional Embeddings**: Learned position encodings

### Both Models Support
- Greedy decoding
- Beam search decoding with length penalty
- Gradient clipping and weight decay
- Automatic mixed precision (if available)

## Training Arguments

```bash
python src/train.py --help
```

Key arguments:
- `--mode`: Model type (`blt` or `char`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--d_model`: Model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 4)
- `--num_encoder_layers`: Encoder layers (default: 2)
- `--num_decoder_layers`: Decoder layers (default: 2)
- `--dim_feedforward`: FFN dimension (default: 256)
- `--dropout`: Dropout rate (default: 0.1)

## Evaluation Metrics

The evaluation script computes:
- **Exact Match Accuracy**: Percentage of perfectly reversed strings
- **Character Accuracy**: Character-level accuracy across all predictions
- **Per-sample Results**: Saved to CSV with input, target, prediction, and correctness

## Data Format

CSV files should have two columns:
```csv
input,target
hello,olleh
world,dlrow
```

For the string reversal task, target is the reversed input.

## Implementation Details

### Patching Algorithm
- **Window Size (W)**: 10 bytes
- **Entropy Threshold**: 2.0 bits
- **Max Patch Length**: 15 bytes
- Patches are created when entropy exceeds threshold or max length is reached

### N-gram Hashing
- **Bucket Size**: 4096 per n-gram size
- **N-gram Sizes**: 1, 2, 3
- **Hash Function**: SHA256-based deterministic hashing
- **Seed**: 1337 for reproducibility

### Tokenizer
- **Vocabulary**: Printable ASCII (32-126)
- **Special Tokens**: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`, `<PATCH>`
- **Total Vocab Size**: 100 tokens

## Performance Optimization

- **Preprocessing**: Data is preprocessed once and cached as `.pt` files
- **DataLoader**: Multi-worker data loading with pin_memory
- **Gradient Clipping**: Prevents exploding gradients
- **Efficient Collation**: Custom collate functions for each model type

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size`
- Reduce `--d_model` or `--dim_feedforward`
- Use fewer `--num_workers`

**Slow Training:**
- Increase `--batch_size` if memory allows
- Increase `--num_workers` for data loading
- Ensure CUDA is available for GPU acceleration

**Poor Performance:**
- Increase `--epochs`
- Try different learning rates (`--lr`)
- Increase model capacity (`--d_model`, `--num_encoder_layers`)
- Use beam search for inference (`--beam_width 5`)

## Citation

This implementation is based on the BLT paper:
```
Byte Latent Transformer: Patches Scale Better Than Tokens
https://arxiv.org/pdf/2412.09871
```


