# Quick Start Guide

This guide will help you get started with the BLT implementation in under 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

## Quick Test (2 minutes)

Run a quick test with minimal epochs to verify everything works:

```bash
python run_experiments.py --quick
```

This will:
- Train both BLT and Baseline models for 2 epochs
- Evaluate on test set
- Generate predictions and metrics

## Full Training (10-30 minutes depending on hardware)

For full training with 10 epochs:

```bash
python run_experiments.py --full
```

## Step-by-Step Manual Execution

If you prefer to run each step manually:

### 1. Train BLT Model

```bash
python src/train.py \
    --mode blt \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3
```

### 2. Train Baseline Model

```bash
python src/train.py \
    --mode char \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-3
```

### 3. Evaluate BLT

```bash
python src/eval.py \
    --mode blt \
    --checkpoint checkpoints/best_blt_model.pt \
    --test_csv data/test.csv
```

### 4. Evaluate Baseline

```bash
python src/eval.py \
    --mode char \
    --checkpoint checkpoints/best_char_model.pt \
    --test_csv data/test.csv
```

## Interactive Inference

Try the models interactively:

```bash
# BLT model
python src/infer.py --mode blt --checkpoint checkpoints/best_blt_model.pt

# Baseline model
python src/infer.py --mode char --checkpoint checkpoints/best_char_model.pt
```

Then type any string and press Enter to see the reversed output.

## Understanding the Output

### Training Output

During training, you'll see:
- Progress bars for each epoch
- Training loss per batch
- Validation loss (if validation set provided)
- Best model checkpoints saved

### Evaluation Output

After evaluation, you'll get:
- **Exact Match Accuracy**: % of perfectly reversed strings
- **Character Accuracy**: Character-level accuracy
- **Predictions CSV**: Detailed results for each sample

### Files Generated

```
checkpoints/
├── best_blt_model.pt          # Best BLT checkpoint
├── best_char_model.pt         # Best baseline checkpoint
├── latest_blt_model.pt        # Latest BLT checkpoint
├── latest_char_model.pt       # Latest baseline checkpoint
├── history_blt.json           # BLT training history
└── history_char.json          # Baseline training history

predictions/
├── predictions_blt.csv        # BLT predictions
├── predictions_baseline.csv   # Baseline predictions
├── predictions_blt_metrics.json
├── predictions_baseline_metrics.json
└── comparison_results.json    # Side-by-side comparison

data/processed/
├── train_blt.pt              # Preprocessed BLT training data
├── train_char.pt             # Preprocessed baseline training data
├── test_blt.pt               # Preprocessed BLT test data
├── test_char.pt              # Preprocessed baseline test data
├── tokenizer_blt.json        # BLT tokenizer
└── tokenizer_char.json       # Baseline tokenizer
```

## Common Issues

### Out of Memory

If you get OOM errors:
```bash
python src/train.py --mode blt --batch_size 16  # Reduce batch size
```

### Slow Training

If training is slow:
```bash
python src/train.py --mode blt --num_workers 0  # Disable multiprocessing
```

### CUDA Not Available

The code automatically falls back to CPU if CUDA is not available. Training will be slower but still work.

## Next Steps

1. **Analyze Results**: Check `predictions/comparison_results.json` for detailed metrics
2. **Read Report**: See `docs/REPORT.md` for in-depth analysis
3. **Experiment**: Try different hyperparameters in `src/train.py`
4. **Visualize**: Use the training history JSONs to plot learning curves

## Example Session

```bash
$ python run_experiments.py --quick

============================================================
BLT Experimental Pipeline
============================================================
Configuration:
  Epochs: 2
  Batch size: 32
  Beam width: 1
  Skip training: False
============================================================

============================================================
Training BLT Model
============================================================
Using device: cuda
Preprocessing data in blt mode...
Vocabulary size: 100
Train batches: 156

Creating BLT model...
Model parameters: 1,234,567

Starting training for 2 epochs...

Epoch 1/2
Training: 100%|████████████| 156/156 [00:45<00:00, 3.45it/s, loss=0.1234]
Train Loss: 0.1234

Epoch 2/2
Training: 100%|████████████| 156/156 [00:45<00:00, 3.45it/s, loss=0.0567]
Train Loss: 0.0567

✓ Completed in 90.5s

[... similar output for baseline ...]

============================================================
Experimental Results Summary
============================================================

BLT Model:
  Exact Match Accuracy: 95.67%
  Character Accuracy: 98.23%
  Total Samples: 1000

Baseline Model:
  Exact Match Accuracy: 94.12%
  Character Accuracy: 97.89%
  Total Samples: 1000

Comparison:
  Exact Match Δ: +1.55%
  Character Accuracy Δ: +0.34%

✓ Comparison saved to predictions/comparison_results.json

============================================================
Pipeline Complete!
============================================================
```

## Tips for Best Results

1. **Use GPU**: Training is 10-20x faster on GPU
2. **Increase Epochs**: 10+ epochs for better convergence
3. **Tune Learning Rate**: Try 5e-4 or 2e-3 if default doesn't work well
4. **Use Beam Search**: Add `--beam_width 5` for better inference quality
5. **Monitor Training**: Check loss curves in history JSONs

## Getting Help

- Check `README.md` for detailed documentation
- See `docs/REPORT.md` for implementation details
- Review code comments in `src/` directory
- Open an issue if you encounter bugs

Happy experimenting! 🚀
