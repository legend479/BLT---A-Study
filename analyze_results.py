#!/usr/bin/env python3
"""
Comprehensive analysis and visualization script for BLT vs Baseline comparison.

This script generates beautiful visualizations and detailed analysis of:
1. Hyperparameter tuning results
2. Training curves and convergence
3. Model performance comparison
4. Accuracy analysis by input length
5. Error analysis and examples
6. Statistical significance testing

Usage:
    python analyze_results.py
    python analyze_results.py --output_dir custom_results
    python analyze_results.py --no_plots  # Skip plot generation
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
from collections import defaultdict
import textwrap

# Handle optional dependencies gracefully
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    HAS_PLOTTING_DEPS = True
except ImportError as e:
    print(f"Warning: Some plotting dependencies are missing: {e}")
    print("Install with: pip install pandas numpy matplotlib seaborn scipy")
    HAS_PLOTTING_DEPS = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Beautiful pastel color palette
PASTEL_COLORS = {
    'blt': '#FFB3BA',           # Soft pink
    'baseline': '#BAFFC9',      # Soft green
    'char': '#BAFFC9',          # Soft green (alias for baseline)
    'primary': '#FFB3BA',       # Soft pink
    'secondary': '#BAFFC9',     # Soft green
    'tertiary': '#BAE1FF',      # Soft blue
    'quaternary': '#FFFFBA',    # Soft yellow
    'quinary': '#E1BAFF',       # Soft purple
    'accent': '#FFDFBA',        # Soft orange
    'neutral': '#F0F0F0',       # Light gray
    'text': '#2C3E50',          # Dark blue-gray
    'grid': '#E8E8E8'           # Very light gray
}

# Set up matplotlib style
plt.style.use('default')
sns.set_palette([PASTEL_COLORS['primary'], PASTEL_COLORS['secondary'], 
                PASTEL_COLORS['tertiary'], PASTEL_COLORS['quaternary'],
                PASTEL_COLORS['quinary'], PASTEL_COLORS['accent']])

def setup_plot_style():
    """Configure matplotlib and seaborn for beautiful pastel plots."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': PASTEL_COLORS['grid'],
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.color': PASTEL_COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        'text.color': PASTEL_COLORS['text'],
        'axes.labelcolor': PASTEL_COLORS['text'],
        'xtick.color': PASTEL_COLORS['text'],
        'ytick.color': PASTEL_COLORS['text'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    })


def load_results() -> Dict[str, Any]:
    """Load all available results from various sources."""
    results = {}
    
    # Load comprehensive results if available
    comprehensive_path = "results/comprehensive_results.json"
    if os.path.exists(comprehensive_path):
        try:
            with open(comprehensive_path) as f:
                data = json.load(f)
                results.update(data.get('results', {}))
                results['config'] = data.get('config', {})
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load comprehensive results: {e}")
    
    # Load individual prediction metrics
    for mode in ['blt', 'baseline']:
        metrics_path = f"predictions/predictions_{mode}_metrics.json"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    results[mode] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {mode} metrics: {e}")
    
    # Load hyperparameter tuning results
    for mode in ['blt', 'char']:
        tuning_path = f"results/hyperparameter_tuning_{mode}.json"
        if os.path.exists(tuning_path):
            try:
                with open(tuning_path) as f:
                    key = "baseline_tuning" if mode == 'char' else f"{mode}_tuning"
                    results[key] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {mode} tuning results: {e}")
    
    # Load training histories
    for mode in ['blt', 'char']:
        history_path = f"checkpoints/history_{mode}.json"
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    key = f"{mode}_history" if mode == 'blt' else "baseline_history"
                    results[key] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {mode} training history: {e}")
    
    # Load prediction CSVs
    for mode in ['blt', 'baseline']:
        csv_path = f"predictions/predictions_{mode}.csv"
        if os.path.exists(csv_path):
            try:
                results[f"{mode}_predictions"] = pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError) as e:
                print(f"Warning: Could not load {mode} predictions CSV: {e}")
    
    return results


def create_hyperparameter_analysis(results: Dict[str, Any], output_dir: str) -> None:
    """Create comprehensive hyperparameter tuning analysis."""
    if 'blt_tuning' not in results and 'baseline_tuning' not in results:
        print("No hyperparameter tuning results found, skipping hyperparameter analysis.")
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hyperparameter Tuning Analysis', fontsize=16, fontweight='bold', 
                     color=PASTEL_COLORS['text'], y=0.98)
    except Exception as e:
        print(f"Error creating hyperparameter analysis plot: {e}")
        return
    
    # Collect all tuning results
    all_results = {}
    if 'blt_tuning' in results:
        all_results['BLT'] = results['blt_tuning'].get('all_results', [])
    if 'baseline_tuning' in results:
        all_results['Baseline'] = results['baseline_tuning'].get('all_results', [])
    
    if not all_results:
        print("No detailed tuning results found.")
        return
    
    # 1. Score distribution comparison
    ax = axes[0, 0]
    scores_data = []
    for model_name, model_results in all_results.items():
        scores = [r['score'] for r in model_results if 'score' in r]
        scores_data.extend([(score, model_name) for score in scores])
    
    if scores_data:
        df_scores = pd.DataFrame(scores_data, columns=['Score', 'Model'])
        sns.boxplot(data=df_scores, x='Model', y='Score', ax=ax, 
                   palette=[PASTEL_COLORS['primary'], PASTEL_COLORS['secondary']])
        ax.set_title('Score Distribution Across Hyperparameter Combinations', fontweight='bold')
        ax.set_ylabel('Validation Score (Higher = Better)')
    
    # 2. Learning rate vs performance
    ax = axes[0, 1]
    for i, (model_name, model_results) in enumerate(all_results.items()):
        lrs = [r['params'].get('lr', 0) for r in model_results if 'params' in r]
        scores = [r['score'] for r in model_results if 'score' in r]
        if lrs and scores:
            color = PASTEL_COLORS['primary'] if model_name == 'BLT' else PASTEL_COLORS['secondary']
            ax.scatter(lrs, scores, alpha=0.7, s=60, label=model_name, color=color)
    
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Score')
    ax.set_title('Learning Rate vs Performance', fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    
    # 3. Model dimension vs performance
    ax = axes[1, 0]
    for i, (model_name, model_results) in enumerate(all_results.items()):
        d_models = [r['params'].get('d_model', 0) for r in model_results if 'params' in r]
        scores = [r['score'] for r in model_results if 'score' in r]
        if d_models and scores:
            color = PASTEL_COLORS['primary'] if model_name == 'BLT' else PASTEL_COLORS['secondary']
            ax.scatter(d_models, scores, alpha=0.7, s=60, label=model_name, color=color)
    
    ax.set_xlabel('Model Dimension')
    ax.set_ylabel('Validation Score')
    ax.set_title('Model Dimension vs Performance', fontweight='bold')
    ax.legend()
    
    # 4. Best parameters comparison
    ax = axes[1, 1]
    best_params_data = []
    for model_name, model_key in [('BLT', 'blt_tuning'), ('Baseline', 'baseline_tuning')]:
        if model_key in results:
            best_params = results[model_key].get('best_params', {})
            for param, value in best_params.items():
                if isinstance(value, (int, float)):
                    best_params_data.append((param, value, model_name))
    
    if best_params_data:
        df_params = pd.DataFrame(best_params_data, columns=['Parameter', 'Value', 'Model'])
        # Normalize values for comparison
        for param in df_params['Parameter'].unique():
            param_data = df_params[df_params['Parameter'] == param]
            if len(param_data) > 1:
                max_val = param_data['Value'].max()
                df_params.loc[df_params['Parameter'] == param, 'Normalized_Value'] = param_data['Value'] / max_val
        
        pivot_df = df_params.pivot(index='Parameter', columns='Model', values='Normalized_Value')
        pivot_df.plot(kind='bar', ax=ax, color=[PASTEL_COLORS['primary'], PASTEL_COLORS['secondary']], 
                     alpha=0.8, width=0.7)
        ax.set_title('Best Hyperparameters Comparison\n(Normalized)', fontweight='bold')
        ax.set_ylabel('Normalized Value')
        ax.legend(title='Model')
        ax.tick_params(axis='x', rotation=45)
    
    try:
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hyperparameter_analysis.png", dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"✓ Hyperparameter analysis saved to {output_dir}/hyperparameter_analysis.png")
    except Exception as e:
        print(f"Error saving hyperparameter analysis: {e}")
    finally:
        plt.close()


def create_training_curves(results: Dict[str, Any], output_dir: str) -> None:
    """Create beautiful training curve visualizations."""
    histories = {}
    if 'blt_history' in results:
        histories['BLT'] = results['blt_history']
    if 'baseline_history' in results:
        histories['Baseline'] = results['baseline_history']
    
    if not histories:
        print("No training history found, skipping training curves.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold', 
                 color=PASTEL_COLORS['text'], y=0.98)
    
    # 1. Training and validation loss
    ax = axes[0, 0]
    for i, (model_name, history) in enumerate(histories.items()):
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        color = PASTEL_COLORS['primary'] if model_name == 'BLT' else PASTEL_COLORS['secondary']
        
        # Training loss
        if 'train_loss' in history and history['train_loss']:
            ax.plot(epochs, history['train_loss'], label=f'{model_name} Train', 
                   color=color, linewidth=2.5, alpha=0.8)
        
        # Validation loss
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], label=f'{model_name} Val', 
                   color=color, linewidth=2.5, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Learning rate schedule
    ax = axes[0, 1]
    for i, (model_name, history) in enumerate(histories.items()):
        if 'lr' in history and history['lr']:
            epochs = range(1, len(history['lr']) + 1)
            color = PASTEL_COLORS['primary'] if model_name == 'BLT' else PASTEL_COLORS['secondary']
            ax.plot(epochs, history['lr'], label=model_name, color=color, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training time per epoch
    ax = axes[1, 0]
    for i, (model_name, history) in enumerate(histories.items()):
        if 'epoch_times' in history and history['epoch_times']:
            epochs = range(1, len(history['epoch_times']) + 1)
            color = PASTEL_COLORS['primary'] if model_name == 'BLT' else PASTEL_COLORS['secondary']
            ax.plot(epochs, history['epoch_times'], label=model_name, color=color, 
                   linewidth=2.5, alpha=0.8, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time per Epoch', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Convergence analysis (loss improvement)
    ax = axes[1, 1]
    for i, (model_name, history) in enumerate(histories.items()):
        if 'train_loss' in history and len(history['train_loss']) > 1:
            losses = history['train_loss']
            improvements = [losses[0] - loss for loss in losses]
            epochs = range(1, len(improvements) + 1)
            color = PASTEL_COLORS['primary'] if model_name == 'BLT' else PASTEL_COLORS['secondary']
            ax.plot(epochs, improvements, label=model_name, color=color, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Improvement from Start')
    ax.set_title('Convergence Analysis', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Training curves saved to {output_dir}/training_curves.png")


def create_performance_comparison(results: Dict[str, Any], output_dir: str) -> None:
    """Create comprehensive performance comparison visualizations."""
    if 'blt' not in results or 'baseline' not in results:
        print("Missing model results for performance comparison.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', 
                 color=PASTEL_COLORS['text'], y=0.98)
    
    blt_metrics = results['blt']
    baseline_metrics = results['baseline']
    
    # 1. Overall accuracy comparison
    ax = axes[0, 0]
    metrics_to_compare = [
        ('Exact Match', 'exact_match_accuracy'),
        ('Character Accuracy', 'character_accuracy')
    ]
    
    blt_scores = []
    baseline_scores = []
    metric_names = []
    
    for name, key in metrics_to_compare:
        if key in blt_metrics and key in baseline_metrics:
            blt_scores.append(blt_metrics[key] * 100)
            baseline_scores.append(baseline_metrics[key] * 100)
            metric_names.append(name)
    
    if metric_names:
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, blt_scores, width, label='BLT', 
                      color=PASTEL_COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, baseline_scores, width, label='Baseline', 
                      color=PASTEL_COLORS['secondary'], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    # 2. Error metrics comparison
    ax = axes[0, 1]
    error_metrics = [
        ('Avg Edit Distance', 'avg_edit_distance'),
        ('Max Edit Distance', 'max_edit_distance'),
        ('Avg Length Diff', 'avg_length_diff')
    ]
    
    blt_errors = []
    baseline_errors = []
    error_names = []
    
    for name, key in error_metrics:
        if key in blt_metrics and key in baseline_metrics:
            blt_errors.append(blt_metrics[key])
            baseline_errors.append(baseline_metrics[key])
            error_names.append(name)
    
    if error_names:
        x = np.arange(len(error_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, blt_errors, width, label='BLT', 
                      color=PASTEL_COLORS['primary'], alpha=0.8)
        bars2 = ax.bar(x + width/2, baseline_errors, width, label='Baseline', 
                      color=PASTEL_COLORS['secondary'], alpha=0.8)
        
        ax.set_xlabel('Error Metrics')
        ax.set_ylabel('Error Value')
        ax.set_title('Error Metrics Comparison (Lower = Better)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(error_names, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    # 3. Performance by input length
    ax = axes[1, 0]
    blt_by_length = blt_metrics.get('accuracy_by_length', {})
    baseline_by_length = baseline_metrics.get('accuracy_by_length', {})
    
    if blt_by_length and baseline_by_length:
        length_ranges = sorted(set(blt_by_length.keys()) & set(baseline_by_length.keys()))
        blt_accs = [blt_by_length[lr]['accuracy'] * 100 for lr in length_ranges]
        baseline_accs = [baseline_by_length[lr]['accuracy'] * 100 for lr in length_ranges]
        
        x = np.arange(len(length_ranges))
        width = 0.35
        
        ax.bar(x - width/2, blt_accs, width, label='BLT', 
               color=PASTEL_COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, baseline_accs, width, label='Baseline', 
               color=PASTEL_COLORS['secondary'], alpha=0.8)
        
        ax.set_xlabel('Input Length Range')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Input Length', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(length_ranges, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Sample count and timing comparison
    ax = axes[1, 1]
    comparison_data = []
    
    # Sample counts
    if 'total_samples' in blt_metrics:
        comparison_data.append(('Total Samples', blt_metrics['total_samples'], blt_metrics['total_samples']))
    
    # Correct predictions
    if 'total_correct' in blt_metrics and 'total_correct' in baseline_metrics:
        comparison_data.append(('Correct Predictions', blt_metrics['total_correct'], baseline_metrics['total_correct']))
    
    # Evaluation time
    if 'eval_time' in blt_metrics and 'eval_time' in baseline_metrics:
        comparison_data.append(('Eval Time (s)', blt_metrics['eval_time'], baseline_metrics['eval_time']))
    
    if comparison_data:
        categories = [item[0] for item in comparison_data]
        blt_values = [item[1] for item in comparison_data]
        baseline_values = [item[2] for item in comparison_data]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, blt_values, width, label='BLT', 
               color=PASTEL_COLORS['primary'], alpha=0.8)
        ax.bar(x + width/2, baseline_values, width, label='Baseline', 
               color=PASTEL_COLORS['secondary'], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Count / Time')
        ax.set_title('Additional Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Performance comparison saved to {output_dir}/performance_comparison.png")


def create_detailed_analysis(results: Dict[str, Any], output_dir: str) -> None:
    """Create detailed error analysis and example visualizations."""
    predictions_data = {}
    for mode in ['blt', 'baseline']:
        if f"{mode}_predictions" in results:
            df = results[f"{mode}_predictions"]
            # Optimize memory usage for large datasets
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)  # Sample for visualization
            predictions_data[mode] = df
    
    if not predictions_data:
        print("No prediction data found for detailed analysis.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Analysis and Error Patterns', fontsize=16, fontweight='bold', 
                 color=PASTEL_COLORS['text'], y=0.98)
    
    # 1. Error distribution by input length
    ax = axes[0, 0]
    for i, (model_name, df) in enumerate(predictions_data.items()):
        if 'input' in df.columns and 'correct' in df.columns:
            df['input_length'] = df['input'].str.len()
            df['error'] = ~df['correct']
            
            # Create length bins
            bins = [0, 10, 20, 50, 100, 200, 1000]
            df['length_bin'] = pd.cut(df['input_length'], bins=bins, right=False)
            
            error_by_length = df.groupby('length_bin')['error'].mean() * 100
            
            color = PASTEL_COLORS['primary'] if model_name == 'blt' else PASTEL_COLORS['secondary']
            error_by_length.plot(kind='bar', ax=ax, alpha=0.8, color=color, 
                                label=model_name.upper(), width=0.7)
    
    ax.set_xlabel('Input Length Range')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate by Input Length', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Prediction length vs target length
    ax = axes[0, 1]
    for i, (model_name, df) in enumerate(predictions_data.items()):
        if all(col in df.columns for col in ['target', 'prediction']):
            target_lengths = df['target'].str.len()
            pred_lengths = df['prediction'].str.len()
            
            color = PASTEL_COLORS['primary'] if model_name == 'blt' else PASTEL_COLORS['secondary']
            ax.scatter(target_lengths, pred_lengths, alpha=0.6, s=20, 
                      label=model_name.upper(), color=color)
    
    # Add diagonal line for perfect predictions
    max_len = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_len], [0, max_len], 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Target Length')
    ax.set_ylabel('Prediction Length')
    ax.set_title('Prediction Length vs Target Length', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Character-level error analysis
    ax = axes[1, 0]
    char_errors = defaultdict(int)
    total_chars = defaultdict(int)
    
    for model_name, df in predictions_data.items():
        if all(col in df.columns for col in ['target', 'prediction']):
            for _, row in df.iterrows():
                target = str(row['target'])
                prediction = str(row['prediction'])
                
                # Count character-level errors
                for i, (t_char, p_char) in enumerate(zip(target, prediction)):
                    total_chars[t_char] += 1
                    if t_char != p_char:
                        char_errors[t_char] += 1
                
                # Handle length differences
                if len(target) > len(prediction):
                    for char in target[len(prediction):]:
                        total_chars[char] += 1
                        char_errors[char] += 1
    
    # Plot most problematic characters
    if char_errors and total_chars:
        error_rates = {char: char_errors[char] / total_chars[char] 
                      for char in char_errors if total_chars[char] >= 10}  # Min 10 occurrences
        
        if error_rates:
            sorted_errors = sorted(error_rates.items(), key=lambda x: x[1], reverse=True)[:15]
            chars, rates = zip(*sorted_errors)
            
            bars = ax.bar(range(len(chars)), [r * 100 for r in rates], 
                         color=PASTEL_COLORS['tertiary'], alpha=0.8)
            ax.set_xlabel('Characters')
            ax.set_ylabel('Error Rate (%)')
            ax.set_title('Character-Level Error Rates', fontweight='bold')
            ax.set_xticks(range(len(chars)))
            ax.set_xticklabels([repr(c) for c in chars], rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Success/failure examples
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find interesting examples
    examples_text = []
    for model_name, df in predictions_data.items():
        if all(col in df.columns for col in ['input', 'target', 'prediction', 'correct']):
            # Get some correct and incorrect examples
            correct_examples = df[df['correct']].head(2)
            incorrect_examples = df[~df['correct']].head(2)
            
            examples_text.append(f"{model_name.upper()} Examples:")
            examples_text.append("✓ Correct:")
            for _, row in correct_examples.iterrows():
                input_str = str(row['input'])[:20] + "..." if len(str(row['input'])) > 20 else str(row['input'])
                pred_str = str(row['prediction'])[:20] + "..." if len(str(row['prediction'])) > 20 else str(row['prediction'])
                examples_text.append(f"  '{input_str}' → '{pred_str}'")
            
            examples_text.append("✗ Incorrect:")
            for _, row in incorrect_examples.iterrows():
                input_str = str(row['input'])[:15] + "..." if len(str(row['input'])) > 15 else str(row['input'])
                target_str = str(row['target'])[:15] + "..." if len(str(row['target'])) > 15 else str(row['target'])
                pred_str = str(row['prediction'])[:15] + "..." if len(str(row['prediction'])) > 15 else str(row['prediction'])
                examples_text.append(f"  '{input_str}' → '{pred_str}' (expected '{target_str}')")
            
            examples_text.append("")
    
    if examples_text:
        ax.text(0.05, 0.95, '\n'.join(examples_text), transform=ax.transAxes, 
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PASTEL_COLORS['neutral'], alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Detailed analysis saved to {output_dir}/detailed_analysis.png")


def create_summary_report(results: Dict[str, Any], output_dir: str) -> None:
    """Create a comprehensive text summary report."""
    report_lines = []
    
    # Header
    report_lines.extend([
        "="*80,
        "BLT vs BASELINE: COMPREHENSIVE ANALYSIS REPORT",
        "="*80,
        "",
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    # Model Performance Summary
    if 'blt' in results and 'baseline' in results:
        blt_metrics = results['blt']
        baseline_metrics = results['baseline']
        
        report_lines.extend([
            "MODEL PERFORMANCE SUMMARY",
            "-" * 40,
            ""
        ])
        
        # Key metrics comparison
        metrics_comparison = [
            ("Exact Match Accuracy", "exact_match_accuracy", "%"),
            ("Character Accuracy", "character_accuracy", "%"),
            ("Average Edit Distance", "avg_edit_distance", ""),
            ("Total Samples", "total_samples", ""),
            ("Correct Predictions", "total_correct", "")
        ]
        
        for metric_name, key, unit in metrics_comparison:
            if key in blt_metrics and key in baseline_metrics:
                blt_val = blt_metrics[key]
                baseline_val = baseline_metrics[key]
                
                if unit == "%":
                    blt_str = f"{blt_val*100:.2f}%"
                    baseline_str = f"{baseline_val*100:.2f}%"
                    diff = (blt_val - baseline_val) * 100
                    diff_str = f"{diff:+.2f}%"
                else:
                    blt_str = f"{blt_val:.2f}" if isinstance(blt_val, float) else str(blt_val)
                    baseline_str = f"{baseline_val:.2f}" if isinstance(baseline_val, float) else str(baseline_val)
                    diff = blt_val - baseline_val
                    diff_str = f"{diff:+.2f}" if isinstance(diff, float) else f"{diff:+d}"
                
                report_lines.append(f"{metric_name:<25}: BLT {blt_str:>10} | Baseline {baseline_str:>10} | Diff {diff_str:>8}")
        
        report_lines.append("")
    
    # Hyperparameter Analysis
    if 'blt_tuning' in results or 'baseline_tuning' in results:
        report_lines.extend([
            "HYPERPARAMETER TUNING SUMMARY",
            "-" * 40,
            ""
        ])
        
        for model_name, key in [("BLT", "blt_tuning"), ("Baseline", "baseline_tuning")]:
            if key in results:
                tuning_data = results[key]
                report_lines.extend([
                    f"{model_name} Model:",
                    f"  Combinations tested: {tuning_data.get('successful_runs', 0)}",
                    f"  Best score: {tuning_data.get('best_score', 0):.4f}",
                    "  Best parameters:"
                ])
                
                best_params = tuning_data.get('best_params', {})
                for param, value in best_params.items():
                    report_lines.append(f"    {param}: {value}")
                
                report_lines.append("")
    
    # Training Analysis
    training_summary = []
    for model_name, key in [("BLT", "blt_history"), ("Baseline", "baseline_history")]:
        if key in results:
            history = results[key]
            if 'train_loss' in history and history['train_loss']:
                final_loss = history['train_loss'][-1]
                initial_loss = history['train_loss'][0]
                improvement = initial_loss - final_loss
                
                total_time = sum(history.get('epoch_times', []))
                avg_epoch_time = total_time / len(history.get('epoch_times', [1])) if history.get('epoch_times') else 0
                
                training_summary.append((model_name, final_loss, improvement, total_time, avg_epoch_time))
    
    if training_summary:
        report_lines.extend([
            "TRAINING ANALYSIS",
            "-" * 40,
            ""
        ])
        
        for model_name, final_loss, improvement, total_time, avg_epoch_time in training_summary:
            report_lines.extend([
                f"{model_name} Training:",
                f"  Final loss: {final_loss:.4f}",
                f"  Loss improvement: {improvement:.4f}",
                f"  Total training time: {total_time:.1f}s",
                f"  Average epoch time: {avg_epoch_time:.1f}s",
                ""
            ])
    
    # Statistical Analysis
    if 'blt' in results and 'baseline' in results:
        blt_acc = results['blt'].get('exact_match_accuracy', 0)
        baseline_acc = results['baseline'].get('exact_match_accuracy', 0)
        n_samples = results['blt'].get('total_samples', 0)
        
        if n_samples > 0:
            # Simple significance test
            diff = abs(blt_acc - baseline_acc)
            pooled_acc = (blt_acc + baseline_acc) / 2
            std_err = np.sqrt(2 * pooled_acc * (1 - pooled_acc) / n_samples)
            z_score = diff / std_err if std_err > 0 else 0
            
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            significance = "Highly Significant" if p_value < 0.01 else "Significant" if p_value < 0.05 else "Not Significant"
            
            report_lines.extend([
                "STATISTICAL ANALYSIS",
                "-" * 40,
                "",
                f"Accuracy difference: {diff*100:.2f}%",
                f"Standard error: {std_err*100:.2f}%",
                f"Z-score: {z_score:.2f}",
                f"P-value: {p_value:.4f}",
                f"Significance (α=0.05): {significance}",
                ""
            ])
    
    # Key Findings
    findings = []
    if 'blt' in results and 'baseline' in results:
        blt_acc = results['blt'].get('exact_match_accuracy', 0) * 100
        baseline_acc = results['baseline'].get('exact_match_accuracy', 0) * 100
        acc_diff = blt_acc - baseline_acc
        
        if acc_diff > 2:
            findings.append(f"• BLT significantly outperforms baseline by {acc_diff:.2f}% in exact match accuracy")
        elif acc_diff > 0:
            findings.append(f"• BLT slightly outperforms baseline by {acc_diff:.2f}% in exact match accuracy")
        elif acc_diff > -2:
            findings.append(f"• BLT and baseline show comparable performance (difference: {acc_diff:.2f}%)")
        else:
            findings.append(f"• Baseline outperforms BLT by {abs(acc_diff):.2f}% in exact match accuracy")
        
        # Edit distance analysis
        blt_edit = results['blt'].get('avg_edit_distance', 0)
        baseline_edit = results['baseline'].get('avg_edit_distance', 0)
        if blt_edit < baseline_edit:
            findings.append(f"• BLT produces outputs closer to targets (avg edit distance: {blt_edit:.2f} vs {baseline_edit:.2f})")
        elif baseline_edit < blt_edit:
            findings.append(f"• Baseline produces outputs closer to targets (avg edit distance: {baseline_edit:.2f} vs {blt_edit:.2f})")
    
    if findings:
        report_lines.extend([
            "KEY FINDINGS",
            "-" * 40,
            ""
        ])
        report_lines.extend(findings)
        report_lines.append("")
    
    # Footer
    report_lines.extend([
        "="*80,
        "END OF ANALYSIS REPORT",
        "="*80
    ])
    
    # Save report
    report_path = f"{output_dir}/analysis_summary.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Summary report saved to {report_path}")
    
    # Also print key findings to console
    if findings:
        print("\nKey Findings:")
        for finding in findings:
            print(f"  {finding}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize BLT experiment results")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for plots and reports")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved plots")
    args = parser.parse_args()
    
    # Validate arguments
    if args.dpi < 50 or args.dpi > 1000:
        print("Warning: DPI should be between 50 and 1000. Using default 300.")
        args.dpi = 300
    
    # Setup
    try:
        setup_plot_style()
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error setting up analysis environment: {e}")
        return
    
    print("Loading experimental results...")
    results = load_results()
    
    if not results:
        print("No results found. Make sure to run experiments first.")
        return
    
    print(f"Found results for: {list(results.keys())}")
    
    if not args.no_plots:
        if not HAS_PLOTTING_DEPS:
            print("Skipping plot generation due to missing dependencies.")
        else:
            print("\nGenerating visualizations...")
            
            # Create all visualizations
            create_hyperparameter_analysis(results, args.output_dir)
            create_training_curves(results, args.output_dir)
            create_performance_comparison(results, args.output_dir)
            create_detailed_analysis(results, args.output_dir)
    
    # Always create summary report
    print("\nGenerating summary report...")
    create_summary_report(results, args.output_dir)
    
    print(f"\n✓ Analysis complete! Results saved to {args.output_dir}/")
    print("\nGenerated files:")
    
    if not args.no_plots:
        plot_files = [
            "hyperparameter_analysis.png",
            "training_curves.png", 
            "performance_comparison.png",
            "detailed_analysis.png"
        ]
        
        for plot_file in plot_files:
            plot_path = f"{args.output_dir}/{plot_file}"
            if os.path.exists(plot_path):
                print(f"  - {plot_path}")
    
    summary_path = f"{args.output_dir}/analysis_summary.txt"
    if os.path.exists(summary_path):
        print(f"  - {summary_path}")


if __name__ == "__main__":
    main()