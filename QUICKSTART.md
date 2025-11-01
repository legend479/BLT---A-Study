# Quick Start Guide - Complete Usage Reference

This guide covers all usage scenarios and file operations for the BLT implementation.

## 🚀 Instant Start (30 seconds)

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete experiment (2-4 hours)
python run_experiments.py --tune

# OR quick test (10 minutes)
python run_experiments.py --quick
```

## 📋 All Usage Options

### 1. Automated Pipeline (`run_experiments.py`)

#### Complete Research Experiment
```bash
python run_experiments.py --tune
```
- **Duration**: 2-4 hours
- **What it does**: Hyperparameter tuning → Training → Evaluation → Analysis
- **Output**: Complete research report with visualizations

#### Quick Validation
```bash
python run_experiments.py --quick
```
- **Duration**: 10-20 minutes  
- **What it does**: Fast training (2 epochs) → Basic evaluation
- **Output**: Quick validation results

#### Full Training
```bash
python run_experiments.py --full
```
- **Duration**: 30-60 minutes
- **What it does**: Complete training (10 epochs) → Full evaluation
- **Output**: Comprehensive comparison

#### Custom Experiments
```bash
# Custom tuning parameters
python run_experiments.py --tune --tune_epochs 5 --max_tune_combinations 20

# Custom training epochs
python run_experiments.py --epochs 15 --batch_size 64

# Evaluation only (skip training)
python run_experiments.py --skip_train --beam_width 5

# Resume interrupted experiment
python run_experiments.py --tune  # Automatically resumes
```

### 2. Individual Training (`src/train.py`)

#### Basic Training
```bash
# BLT model
python src/train.py --mode blt --epochs 10 --batch_size 32

# Baseline model  
python src/train.py --mode char --epochs 10 --batch_size 32
```

#### Optimized Training
```bash
# With mixed precision and learning rate scheduling
python src/train.py --mode blt --epochs 10 --batch_size 32 --use_amp --lr_scheduler cosine

# With gradient accumulation for large effective batch size
python src/train.py --mode blt --batch_size 16 --gradient_accumulation_steps 4 --use_amp

# With early stopping
python src/train.py --mode blt --epochs 20 --early_stopping_patience 3
```

#### Advanced Training Options
```bash
# Custom architecture
python src/train.py --mode blt --d_model 256 --nhead 8 --num_encoder_layers 3 --num_decoder_layers 3

# Custom learning rate and optimization
python src/train.py --mode blt --lr 2e-3 --lr_scheduler step --max_grad_norm 0.5

# Multi-GPU training (automatic detection)
python src/train.py --mode blt --batch_size 64 --use_amp

# Resume from checkpoint
python src/train.py --mode blt --resume_from checkpoints/latest_blt_model.pt
```

### 3. Evaluation (`src/eval.py`)

#### Basic Evaluation
```bash
# BLT model
python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv

# Baseline model
python src/eval.py --mode char --checkpoint checkpoints/best_char_model.pt --test_csv data/test.csv
```

#### Advanced Evaluation
```bash
# With beam search
python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv --beam_width 5

# Custom output location
python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv --output_csv my_predictions.csv

# Large batch evaluation
python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv --batch_size 64
```

### 4. Interactive Inference (`src/infer.py`)

#### Basic Inference
```bash
# BLT model
python src/infer.py --mode blt --checkpoint checkpoints/best_blt_model.pt

# Baseline model
python src/infer.py --mode char --checkpoint checkpoints/best_char_model.pt
```

#### Advanced Inference
```bash
# With beam search
python src/infer.py --mode char --checkpoint checkpoints/best_char_model.pt --beam_width 5

# Custom max length
python src/infer.py --mode blt --checkpoint checkpoints/best_blt_model.pt --max_len 256
```

### 5. Analysis and Visualization (`analyze_results.py`)

```bash
# Generate all analysis and visualizations
python analyze_results.py
```
- Creates training curves, accuracy analysis, hyperparameter impact plots
- Generates summary reports
- Works with any existing results

## 📁 File Reference Guide

### Core Implementation Files

#### `src/train.py` - Training Script
**Purpose**: Train BLT or baseline models with optimizations

**Key Arguments**:
- `--mode`: `blt` or `char` (model type)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--use_amp`: Enable mixed precision training
- `--lr_scheduler`: Learning rate scheduler (`none`, `cosine`, `step`)
- `--early_stopping_patience`: Early stopping patience (default: 0)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)

**Outputs**:
- `checkpoints/best_{mode}_model.pt`: Best model checkpoint
- `checkpoints/latest_{mode}_model.pt`: Latest checkpoint
- `checkpoints/history_{mode}.json`: Training history

#### `src/eval.py` - Evaluation Script
**Purpose**: Evaluate trained models and generate predictions

**Key Arguments**:
- `--mode`: `blt` or `char`
- `--checkpoint`: Path to model checkpoint
- `--test_csv`: Test data CSV file
- `--beam_width`: Beam search width (default: 1)
- `--output_csv`: Output predictions CSV

**Outputs**:
- `predictions/predictions_{mode}.csv`: Detailed predictions
- `predictions/predictions_{mode}_metrics.json`: Evaluation metrics

#### `src/blt_model.py` - BLT Model Implementation
**Purpose**: Optimized BLT model with vectorized patch processing

**Key Features**:
- Entropy-based byte patching
- N-gram hashing with separate embeddings
- Vectorized batch processing
- Memory-efficient tensor operations

#### `src/baseline_model.py` - Baseline Model
**Purpose**: Character-level transformer baseline

**Key Features**:
- Standard encoder-decoder architecture
- Character-level processing
- Comparable to BLT decoder for fair comparison

#### `src/dataset.py` - Data Processing
**Purpose**: Optimized data loading and preprocessing

**Key Features**:
- Efficient collation functions
- Cached preprocessing
- Memory-optimized tensor operations
- Multi-worker data loading

#### `src/patcher.py` - Byte Patching
**Purpose**: Entropy-based byte segmentation

**Key Features**:
- Shannon entropy calculation
- Sliding window patching
- Deterministic n-gram hashing
- Robust error handling

#### `src/tokenizer.py` - Character Tokenizer
**Purpose**: Character-level tokenization with special tokens

**Key Features**:
- ASCII character vocabulary
- Special tokens (PAD, SOS, EOS, UNK)
- Efficient encoding/decoding

#### `src/infer.py` - Interactive Inference
**Purpose**: Interactive model testing

**Key Features**:
- Real-time inference
- Beam search support
- User-friendly interface

### Pipeline and Analysis Files

#### `run_experiments.py` - Complete Pipeline
**Purpose**: End-to-end automated experimentation

**Key Features**:
- Hyperparameter tuning with resume capability
- Automated training and evaluation
- Comprehensive report generation
- Error recovery and robustness

**Usage Modes**:
- `--quick`: Fast validation (2 epochs)
- `--full`: Complete training (10 epochs)
- `--tune`: Research mode with hyperparameter tuning
- `--skip_train`: Evaluation only

#### `analyze_results.py` - Analysis and Visualization
**Purpose**: Generate analysis plots and summaries

**Key Features**:
- Training curve visualization
- Hyperparameter impact analysis
- Performance breakdown plots
- Automated report generation

## 🔧 Configuration and Customization

### Hyperparameter Tuning Configuration

Edit `get_hyperparameter_grid()` in `run_experiments.py`:

```python
def get_hyperparameter_grid():
    return {
        'lr': [5e-4, 1e-3, 2e-3],           # Learning rates to test
        'd_model': [64, 128, 256],          # Model dimensions
        'nhead': [4, 8],                    # Attention heads
        'num_layers': [2, 3],               # Encoder/decoder layers
        'dim_feedforward': [128, 256, 512], # FFN dimensions
        'dropout': [0.1, 0.2],              # Dropout rates
        'batch_size': [16, 32, 64]          # Batch sizes
    }
```

### Model Architecture Customization

#### BLT Model Parameters
```python
# In src/blt_model.py or via command line
BLTModel(
    vocab_size=100,           # Vocabulary size
    d_model=128,             # Model dimension
    nhead=4,                 # Attention heads
    num_encoder_layers=2,    # Encoder layers
    num_decoder_layers=2,    # Decoder layers
    dim_feedforward=256,     # FFN dimension
    dropout=0.1,             # Dropout rate
    max_patches=512,         # Maximum patches
    buckets=4096,            # Hash buckets per n-gram
    ngrams=(1, 2, 3)        # N-gram sizes
)
```

#### Patching Parameters
```python
# In src/patcher.py
patch_bytes(
    data=input_bytes,
    W=10,                    # Window size
    entropy_threshold=2.0,   # Entropy threshold
    max_patch_len=15        # Maximum patch length
)
```

## 📊 Output File Reference

### Results Directory (`results/`)
- `comprehensive_results.json`: All experimental data
- `experimental_report.txt`: Research-grade report
- `hyperparameter_tuning_{mode}.json`: Tuning results
- `hyperparameter_analysis.png`: Parameter impact plots

### Predictions Directory (`predictions/`)
- `predictions_{mode}.csv`: Detailed predictions with correctness
- `predictions_{mode}_metrics.json`: Comprehensive metrics
- `training_curves.png`: Training dynamics visualization
- `accuracy_analysis.png`: Performance breakdown
- `detailed_metrics.png`: Metric comparisons
- `summary_report.txt`: Analysis summary

### Checkpoints Directory (`checkpoints/`)
- `best_{mode}_model.pt`: Best model (lowest validation loss)
- `latest_{mode}_model.pt`: Most recent checkpoint
- `history_{mode}.json`: Training history (loss, times, etc.)
- `tune_{mode}/`: Hyperparameter tuning checkpoints

### Data Directory (`data/`)
- `train.csv`: Training data (10,001 samples)
- `test.csv`: Test data (2,001 samples)
- `processed/`: Cached preprocessed data
  - `train_{mode}.pt`: Preprocessed training data
  - `test_{mode}.pt`: Preprocessed test data
  - `tokenizer_{mode}.json`: Tokenizer configuration

## 🎯 Common Workflows

### Research Workflow
```bash
# 1. Complete experiment with tuning
python run_experiments.py --tune

# 2. Generate additional analysis (if needed)
python analyze_results.py

# 3. Review results
cat results/experimental_report.txt
```

### Development Workflow
```bash
# 1. Quick validation
python run_experiments.py --quick

# 2. Manual training with custom parameters
python src/train.py --mode blt --epochs 5 --batch_size 16 --lr 2e-3

# 3. Evaluate specific checkpoint
python src/eval.py --mode blt --checkpoint checkpoints/latest_blt_model.pt --test_csv data/test.csv

# 4. Interactive testing
python src/infer.py --mode blt --checkpoint checkpoints/best_blt_model.pt
```

### Comparison Workflow
```bash
# 1. Train both models
python src/train.py --mode blt --epochs 10 --use_amp
python src/train.py --mode char --epochs 10 --use_amp

# 2. Evaluate both
python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv
python src/eval.py --mode char --checkpoint checkpoints/best_char_model.pt --test_csv data/test.csv

# 3. Generate comparison
python analyze_results.py
```

## 🚨 Troubleshooting Guide

### Memory Issues
```bash
# Reduce batch size
python src/train.py --mode blt --batch_size 8

# Use gradient accumulation
python src/train.py --mode blt --batch_size 8 --gradient_accumulation_steps 4

# Reduce model size
python src/train.py --mode blt --d_model 64 --dim_feedforward 128
```

### Performance Issues
```bash
# Enable mixed precision
python src/train.py --mode blt --use_amp

# Reduce data loading workers
python src/train.py --mode blt --num_workers 2

# Use smaller dataset for testing
head -100 data/train.csv > data/train_small.csv
python src/train.py --train_csv data/train_small.csv
```

### Resume Interrupted Training
```bash
# Training automatically saves checkpoints - just run again
python run_experiments.py --tune  # Will resume from where it stopped

# Or manually resume
python src/train.py --mode blt --resume_from checkpoints/latest_blt_model.pt
```

## ⚡ Performance Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Enable Mixed Precision**: `--use_amp` for 1.5-2x speedup
3. **Optimize Batch Size**: Larger batches train faster (if memory allows)
4. **Use Multiple Workers**: `--num_workers 4` for faster data loading
5. **Enable Optimizations**: The system auto-optimizes based on your hardware

## 🎓 For Academic Use

### Getting Research-Quality Results
```bash
# Run complete research experiment
python run_experiments.py --tune --tune_epochs 5 --max_tune_combinations 15

# This provides:
# - Rigorous hyperparameter optimization
# - Statistical significance testing
# - Comprehensive performance analysis
# - Publication-ready visualizations
# - Detailed methodology documentation
```

### Using Results in Papers
- `results/experimental_report.txt`: Complete methodology and results
- `results/hyperparameter_analysis.png`: Parameter impact analysis
- `predictions/training_curves.png`: Training dynamics
- `predictions/accuracy_analysis.png`: Performance breakdown

All results include statistical analysis and are ready for academic use.

---

**Need help?** Check the main README.md or EXPERIMENTAL_GUIDE.md for more details!