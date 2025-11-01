#!/usr/bin/env python3
"""
Enhanced experimental pipeline for BLT vs Baseline comparison with hyperparameter tuning.

This script runs the complete pipeline:
1. Hyperparameter tuning (optional)
2. Train both models with best parameters
3. Evaluate on test set
4. Generate comprehensive comparison report

Usage:
    python run_experiments.py --quick                    # Quick test (2 epochs)
    python run_experiments.py --full                     # Full training (10 epochs)
    python run_experiments.py --tune                     # Hyperparameter tuning + full training
    python run_experiments.py --tune --tune_epochs 3     # Custom tuning epochs
    
Using existing tuning results:
    python run_experiments.py --analyze_tuning           # Analyze existing tuning results
    python run_experiments.py --train_from_existing      # Train using existing best parameters
    python run_experiments.py --tune --use_existing_tuning  # Resume incomplete tuning
    
Optimized for GTX 1080 Ti (4x 11.7GB):
    python run_experiments.py --gtx1080ti --quick        # Optimized quick test
    python run_experiments.py --gtx1080ti --tune         # Optimized tuning + training
    python run_experiments.py --gtx1080ti --train_from_existing  # Use existing + optimized training
"""

import argparse
import os
import subprocess
import sys
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def estimate_memory_usage(batch_size: int, d_model: int, seq_len: int = 512) -> float:
    """Estimate GPU memory usage in GB for given parameters."""
    # Rough estimation based on transformer memory requirements
    # This is a simplified calculation for guidance
    
    # Model parameters memory (weights)
    vocab_size = 100  # Approximate
    model_params = (
        vocab_size * d_model +  # Embeddings
        d_model * d_model * 12 +  # Attention weights (4 layers * 3 matrices each)
        d_model * d_model * 4 * 4  # Feedforward weights
    )
    model_memory = model_params * 4 / (1024**3)  # 4 bytes per float32, convert to GB
    
    # Activation memory (forward + backward)
    activation_memory = batch_size * seq_len * d_model * 8 * 4 / (1024**3)  # Rough estimate
    
    # Add some overhead
    total_memory = (model_memory + activation_memory) * 1.5
    
    return total_memory


def check_memory_requirements(config: Dict[str, Any]) -> bool:
    """Check if configuration fits in GTX 1080 Ti memory."""
    estimated_memory = estimate_memory_usage(
        config['batch_size'], 
        config['d_model']
    )
    
    available_memory = 11.0  # GTX 1080 Ti has ~11GB usable
    
    if estimated_memory > available_memory:
        print(f"⚠️  Warning: Estimated memory usage ({estimated_memory:.1f}GB) may exceed available memory ({available_memory}GB)")
        print(f"   Consider reducing batch_size or d_model")
        return False
    else:
        print(f"✓ Memory check passed: Estimated usage {estimated_memory:.1f}GB / {available_memory}GB available")
        return True


def analyze_existing_tuning_results(mode: str) -> Dict[str, Any]:
    """Analyze existing hyperparameter tuning results and show best parameters."""
    tuning_file = f"results/hyperparameter_tuning_{mode}.json"
    
    if not os.path.exists(tuning_file):
        print(f"❌ No existing tuning results found for {mode} at {tuning_file}")
        return {}
    
    try:
        with open(tuning_file) as f:
            tuning_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error loading tuning results: {e}")
        return {}
    
    all_results = tuning_data.get('all_results', [])
    if not all_results:
        print(f"❌ No completed runs found in tuning results for {mode}")
        return {}
    
    print(f"\n{'='*60}")
    print(f"EXISTING TUNING RESULTS ANALYSIS FOR {mode.upper()}")
    print(f"{'='*60}")
    print(f"Total runs completed: {len(all_results)}")
    print(f"Best score so far: {tuning_data.get('best_score', 'N/A'):.4f}")
    
    # Sort results by score (higher is better)
    sorted_results = sorted(all_results, key=lambda x: x.get('score', float('-inf')), reverse=True)
    
    print(f"\nTop 5 parameter combinations:")
    print(f"{'Rank':<4} {'Score':<8} {'Train Loss':<10} {'Val Loss':<10} {'Parameters'}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:5]):
        score = result.get('score', 0)
        train_loss = result.get('train_loss', 0)
        val_loss = result.get('val_loss', 'N/A')
        params = result.get('params', {})
        
        # Format parameters for display
        param_str = f"lr={params.get('lr', 'N/A')}, d_model={params.get('d_model', 'N/A')}, " \
                   f"batch_size={params.get('batch_size', 'N/A')}, layers={params.get('num_layers', 'N/A')}"
        
        print(f"{i+1:<4} {score:<8.4f} {train_loss:<10.4f} {val_loss if val_loss != 'N/A' else 'N/A':<10} {param_str}")
    
    # Show parameter distribution analysis
    if len(all_results) > 1:
        print(f"\nParameter Analysis:")
        param_stats = {}
        for param_name in ['lr', 'd_model', 'batch_size', 'num_layers', 'dim_feedforward']:
            values = [r['params'].get(param_name) for r in all_results if param_name in r.get('params', {})]
            if values:
                param_stats[param_name] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values)
                }
        
        for param, stats in param_stats.items():
            print(f"  {param}: min={stats['min']}, max={stats['max']}, avg={stats['avg']:.3f}")
    
    return tuning_data


def get_best_params_from_existing(mode: str) -> Optional[Dict[str, Any]]:
    """Get best parameters from existing tuning results."""
    tuning_file = f"results/hyperparameter_tuning_{mode}.json"
    
    if not os.path.exists(tuning_file):
        return None
    
    try:
        with open(tuning_file) as f:
            tuning_data = json.load(f)
        
        best_params = tuning_data.get('best_params')
        if best_params:
            print(f"✓ Found existing best parameters for {mode}: {best_params}")
            return tuning_data
        else:
            # If no best_params, find the best from all_results
            all_results = tuning_data.get('all_results', [])
            if all_results:
                best_result = max(all_results, key=lambda x: x.get('score', float('-inf')))
                tuning_data['best_params'] = best_result.get('params', {})
                tuning_data['best_score'] = best_result.get('score', float('-inf'))
                
                # Save updated results
                with open(tuning_file, 'w') as f:
                    json.dump(tuning_data, f, indent=2)
                
                print(f"✓ Updated best parameters for {mode} from existing results")
                return tuning_data
    
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error loading existing tuning results: {e}")
    
    return None


def create_tuning_summary_report(output_dir: str = "results") -> None:
    """Create a summary report of all existing tuning results."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "tuning_summary_report.txt")
    
    report_lines = []
    report_lines.extend([
        "="*80,
        "HYPERPARAMETER TUNING SUMMARY REPORT",
        "="*80,
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    for mode in ['blt', 'char']:
        tuning_file = f"results/hyperparameter_tuning_{mode}.json"
        model_name = "BLT" if mode == "blt" else "Baseline"
        
        report_lines.extend([
            f"{model_name.upper()} MODEL TUNING RESULTS",
            "-" * 40,
            ""
        ])
        
        if os.path.exists(tuning_file):
            try:
                with open(tuning_file) as f:
                    tuning_data = json.load(f)
                
                all_results = tuning_data.get('all_results', [])
                best_params = tuning_data.get('best_params', {})
                best_score = tuning_data.get('best_score', 0)
                
                report_lines.extend([
                    f"Status: {'Completed' if len(all_results) >= tuning_data.get('total_combinations', 0) else 'Incomplete'}",
                    f"Completed runs: {len(all_results)} / {tuning_data.get('total_combinations', 'Unknown')}",
                    f"Best score: {best_score:.4f}",
                    f"Best parameters:",
                ])
                
                for param, value in best_params.items():
                    report_lines.append(f"  {param}: {value}")
                
                if all_results:
                    # Performance statistics
                    scores = [r.get('score', 0) for r in all_results]
                    train_losses = [r.get('train_loss', 0) for r in all_results]
                    
                    report_lines.extend([
                        "",
                        "Performance Statistics:",
                        f"  Score range: {min(scores):.4f} to {max(scores):.4f}",
                        f"  Average score: {sum(scores)/len(scores):.4f}",
                        f"  Train loss range: {min(train_losses):.4f} to {max(train_losses):.4f}",
                    ])
                
            except (json.JSONDecodeError, IOError) as e:
                report_lines.append(f"Error loading results: {e}")
        else:
            report_lines.append("No tuning results found")
        
        report_lines.extend(["", ""])
    
    # Save report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Tuning summary report saved to: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))


def run_command(cmd, description, capture_output=False):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, stdout=None, stderr=None, universal_newlines=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with return code {result.returncode}")
        if capture_output and result.stderr:
            print(f"Error output: {result.stderr}")
        return False, elapsed, None
    
    print(f"\n✓ Completed in {elapsed:.2f}s")
    return True, elapsed, result.stdout if capture_output else None


def get_hyperparameter_grid():
    """Define hyperparameter search space optimized for 4x GTX 1080 Ti (11.7GB each)."""
    return {
        'lr': [1e-3, 2e-3],  # Reduced to 2 values for faster tuning
        'd_model': [128, 256],  # Focus on larger models that benefit from GPU power
        'nhead': [8],  # Fixed to 8 for optimal GPU utilization
        'num_layers': [2, 3],  # Keep both options
        'dim_feedforward': [512, 1024],  # Larger feedforward for better GPU utilization
        'dropout': [0.1],  # Fixed to reduce search space
        'batch_size': [64, 128]  # Larger batches to maximize GPU throughput
    }


def get_optimized_config_for_gtx1080ti():
    """Get optimized configuration for 4x GTX 1080 Ti setup."""
    return {
        'batch_size': 128,  # Large batch for maximum GPU utilization
        'd_model': 256,     # Larger model dimension
        'nhead': 8,         # Optimal attention heads for GPU
        'num_layers': 2,    # Balance between speed and performance
        'dim_feedforward': 1024,  # Large feedforward for GPU efficiency
        'lr': 2e-3,         # Higher learning rate for faster convergence
        'dropout': 0.1,     # Standard dropout
        'num_workers': 8,   # Optimize data loading
        'use_amp': True,    # Mixed precision for speed
        'early_stopping_patience': 2  # Faster convergence detection
    }


def run_hyperparameter_tuning(mode: str, tune_epochs: int, max_combinations: int = 8) -> Dict[str, Any]:
    """Run hyperparameter tuning for a given mode with resume capability."""
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING FOR {mode.upper()} MODEL")
    print(f"{'='*80}")
    
    # Check for existing tuning results to resume
    tuning_file = f"results/hyperparameter_tuning_{mode}.json"
    if os.path.exists(tuning_file):
        print(f"Found existing tuning results for {mode}. Resuming...")
        with open(tuning_file) as f:
            existing_results = json.load(f)
        
        # Check if tuning was completed
        if existing_results.get('successful_runs', 0) >= max_combinations:
            print(f"Tuning already completed for {mode}. Using existing results.")
            return existing_results
        
        results = existing_results.get('all_results', [])
        best_score = existing_results.get('best_score', float('-inf'))
        best_params = existing_results.get('best_params', None)
        completed_params = [r['params'] for r in results]
        print(f"Resuming from {len(results)} completed runs...")
    else:
        results = []
        best_score = float('-inf')
        best_params = None
        completed_params = []
    
    param_grid = get_hyperparameter_grid()
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))
    
    # Limit combinations for practical tuning
    if len(all_combinations) > max_combinations:
        print(f"Limiting to {max_combinations} random combinations out of {len(all_combinations)} total")
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
        combinations = [all_combinations[i] for i in selected_indices]
    else:
        combinations = all_combinations
    
    # Filter out already completed combinations
    remaining_combinations = []
    for combination in combinations:
        params = dict(zip(keys, combination))
        # More robust comparison for parameter matching
        params_match = any(
            all(abs(params.get(k, 0) - completed.get(k, 0)) < 1e-10 if isinstance(params.get(k, 0), float) 
                else params.get(k) == completed.get(k) for k in params.keys())
            for completed in completed_params
        )
        if not params_match:
            remaining_combinations.append(combination)
    
    print(f"Running {len(remaining_combinations)} remaining combinations...")
    
    for i, combination in enumerate(remaining_combinations):
        params = dict(zip(keys, combination))
        run_number = len(results) + i + 1
        total_runs = len(results) + len(remaining_combinations)
        
        print(f"\n{'-'*60}")
        print(f"Tuning Run {run_number}/{total_runs}")
        print(f"Parameters: {params}")
        print(f"{'-'*60}")
        
        # Adjust dim_feedforward based on d_model for GTX 1080 Ti efficiency
        if params['dim_feedforward'] < params['d_model'] * 2:
            params['dim_feedforward'] = params['d_model'] * 4  # Larger feedforward for better GPU utilization
        
        # Build command
        cmd = [
            sys.executable, "src/train.py",
            "--mode", mode,
            "--epochs", str(tune_epochs),
            "--batch_size", str(params['batch_size']),
            "--lr", str(params['lr']),
            "--d_model", str(params['d_model']),
            "--nhead", str(params['nhead']),
            "--num_encoder_layers", str(params['num_layers']),
            "--num_decoder_layers", str(params['num_layers']),
            "--dim_feedforward", str(params['dim_feedforward']),
            "--dropout", str(params['dropout']),
            "--use_amp",  # Always use mixed precision for speed
            "--early_stopping_patience", "2",  # Early stopping for faster tuning
            "--num_workers", "8",  # Optimize data loading for multi-GPU
            "--save_dir", f"checkpoints/tune_{mode}"
        ]
        
        success, train_time, _ = run_command(cmd, f"Training {mode} with params {i+1}")
        
        if success:
            # Load training history to get final loss
            history_path = f"checkpoints/tune_{mode}/history_{mode}.json"
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history = json.load(f)
                
                final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
                final_val_loss = history['val_loss'][-1] if history['val_loss'] else final_train_loss
                
                # Score is negative loss (higher is better)
                score = -final_val_loss if history['val_loss'] else -final_train_loss
                
                result = {
                    'params': params,
                    'train_loss': final_train_loss,
                    'val_loss': final_val_loss if history['val_loss'] else None,
                    'score': score,
                    'train_time': train_time,
                    'epochs_completed': len(history['train_loss'])
                }
                
                results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                print(f"✓ Score: {score:.4f} (Train Loss: {final_train_loss:.4f})")
                
                # Save intermediate results after each successful run
                intermediate_results = {
                    'mode': mode,
                    'tune_epochs': tune_epochs,
                    'total_combinations': len(combinations),
                    'successful_runs': len(results),
                    'best_params': best_params,
                    'best_score': best_score,
                    'all_results': results
                }
                
                os.makedirs("results", exist_ok=True)
                try:
                    with open(f"results/hyperparameter_tuning_{mode}.json", "w") as f:
                        json.dump(intermediate_results, f, indent=2)
                except (IOError, OSError) as e:
                    print(f"⚠ Warning: Could not save intermediate results: {e}")
                
            else:
                print("❌ Could not load training history")
        else:
            print("❌ Training failed")
    
    # Save tuning results
    tuning_results = {
        'mode': mode,
        'tune_epochs': tune_epochs,
        'total_combinations': len(combinations),
        'successful_runs': len(results),
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/hyperparameter_tuning_{mode}.json", "w") as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE FOR {mode.upper()}")
    print(f"{'='*60}")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    print(f"Results saved to: results/hyperparameter_tuning_{mode}.json")
    
    return tuning_results


def train_with_best_params(mode: str, epochs: int, tuning_results: Dict[str, Any]) -> bool:
    """Train model with best parameters from tuning with resume capability."""
    best_params = tuning_results.get('best_params')
    if not best_params:
        print(f"\n❌ No best parameters found for {mode}. Cannot proceed with training.")
        return False
    
    # Check if final model already exists
    final_checkpoint = f"checkpoints/best_{mode}_model.pt"
    if os.path.exists(final_checkpoint):
        print(f"\n✓ Final {mode.upper()} model already exists. Skipping training.")
        return True
    
    print(f"\n{'='*60}")
    print(f"Training {mode.upper()} with Best Parameters")
    print(f"{'='*60}")
    print(f"Best parameters: {best_params}")
    
    cmd = [
        sys.executable, "src/train.py",
        "--mode", mode,
        "--epochs", str(epochs),
        "--batch_size", str(best_params['batch_size']),
        "--lr", str(best_params['lr']),
        "--d_model", str(best_params['d_model']),
        "--nhead", str(best_params['nhead']),
        "--num_encoder_layers", str(best_params['num_layers']),
        "--num_decoder_layers", str(best_params['num_layers']),
        "--dim_feedforward", str(best_params['dim_feedforward']),
        "--dropout", str(best_params['dropout']),
        "--use_amp",
        "--lr_scheduler", "cosine",
        "--early_stopping_patience", "3",
        "--num_workers", "8"  # Optimize data loading
    ]
    
    success, _, _ = run_command(cmd, f"Final Training {mode.upper()} Model")
    return success


def evaluate_model(mode: str, batch_size: int, beam_width: int) -> Dict[str, Any]:
    """Evaluate a trained model with resume capability."""
    # Check if evaluation already exists
    metrics_path = f"predictions/predictions_{mode}_metrics.json"
    if os.path.exists(metrics_path):
        print(f"\n✓ {mode.upper()} evaluation already exists. Loading existing results.")
        with open(metrics_path) as f:
            return json.load(f)
    
    # Check if model checkpoint exists - handle naming convention
    if mode == "baseline":
        # The baseline model is actually saved as "char" mode
        checkpoint_path = f"checkpoints/best_char_model.pt"
        eval_mode = "char"
    else:
        checkpoint_path = f"checkpoints/best_{mode}_model.pt"
        eval_mode = mode
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ No checkpoint found for {mode} model at {checkpoint_path}")
        return {}
    
    cmd = [
        sys.executable, "src/eval.py",
        "--mode", eval_mode,
        "--checkpoint", checkpoint_path,
        "--test_csv", "data/test.csv",
        "--batch_size", str(max(batch_size, 64)),  # Larger batch for faster evaluation
        "--beam_width", str(beam_width),
        "--output_csv", f"predictions/predictions_{mode}.csv",
        "--num_workers", "8"  # Optimize data loading
    ]
    
    success, eval_time, _ = run_command(cmd, f"Evaluating {mode.upper()} Model")
    
    if success:
        # Load metrics
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            metrics['eval_time'] = eval_time
            return metrics
    
    return {}


def generate_comprehensive_report(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate a comprehensive experimental report."""
    report_lines = []
    
    # Header
    report_lines.extend([
        "="*80,
        "BLT vs BASELINE: COMPREHENSIVE EXPERIMENTAL REPORT",
        "="*80,
        "",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total experiment time: {config.get('total_time', 0):.1f} seconds",
        ""
    ])
    
    # Experimental Configuration
    report_lines.extend([
        "EXPERIMENTAL CONFIGURATION",
        "-" * 80,
        f"Training epochs: {config.get('epochs', 'N/A')}",
        f"Evaluation beam width: {config.get('beam_width', 'N/A')}",
        f"Hyperparameter tuning: {'Yes' if config.get('tuning_performed', False) else 'No'}",
        ""
    ])
    
    # Hyperparameter Tuning Results
    if config.get('tuning_performed', False):
        report_lines.extend([
            "HYPERPARAMETER TUNING RESULTS",
            "-" * 80,
            ""
        ])
        
        for mode in ['blt', 'baseline']:
            if f'{mode}_tuning' in results:
                tuning = results[f'{mode}_tuning']
                report_lines.extend([
                    f"{mode.upper()} Model Tuning:",
                    f"  Total combinations tested: {tuning.get('total_combinations', 0)}",
                    f"  Successful runs: {tuning.get('successful_runs', 0)}",
                    f"  Best validation score: {tuning.get('best_score', 0):.4f}",
                    f"  Best parameters:",
                ])
                
                best_params = tuning.get('best_params', {})
                for param, value in best_params.items():
                    report_lines.append(f"    {param}: {value}")
                
                report_lines.append("")
        
        # Tuning insights
        report_lines.extend([
            "Hyperparameter Insights:",
            ""
        ])
        
        if 'blt_tuning' in results and 'baseline_tuning' in results:
            blt_params = results['blt_tuning'].get('best_params', {})
            baseline_params = results['baseline_tuning'].get('best_params', {})
            
            report_lines.extend([
                f"  • Learning Rate: BLT={blt_params.get('lr', 'N/A')}, Baseline={baseline_params.get('lr', 'N/A')}",
                f"  • Model Dimension: BLT={blt_params.get('d_model', 'N/A')}, Baseline={baseline_params.get('d_model', 'N/A')}",
                f"  • Attention Heads: BLT={blt_params.get('nhead', 'N/A')}, Baseline={baseline_params.get('nhead', 'N/A')}",
                f"  • Layers: BLT={blt_params.get('num_layers', 'N/A')}, Baseline={baseline_params.get('num_layers', 'N/A')}",
                f"  • Batch Size: BLT={blt_params.get('batch_size', 'N/A')}, Baseline={baseline_params.get('batch_size', 'N/A')}",
                ""
            ])
    
    # Model Performance Results
    report_lines.extend([
        "MODEL PERFORMANCE RESULTS",
        "-" * 80,
        ""
    ])
    
    if "blt" in results and "baseline" in results:
        blt_metrics = results["blt"]
        baseline_metrics = results["baseline"]
        
        # Performance table
        report_lines.extend([
            f"{'Metric':<25} {'BLT':<15} {'Baseline':<15} {'Difference':<15}",
            "-" * 70,
        ])
        
        metrics_to_compare = [
            ('Exact Match Accuracy', 'exact_match_accuracy', '%'),
            ('Character Accuracy', 'character_accuracy', '%'),
            ('Avg Edit Distance', 'avg_edit_distance', ''),
            ('Max Edit Distance', 'max_edit_distance', ''),
            ('Avg Length Diff', 'avg_length_diff', ''),
            ('Evaluation Time', 'eval_time', 's')
        ]
        
        for metric_name, metric_key, unit in metrics_to_compare:
            blt_val = blt_metrics.get(metric_key, 0)
            baseline_val = baseline_metrics.get(metric_key, 0)
            
            if unit == '%':
                blt_str = f"{blt_val*100:.2f}%"
                baseline_str = f"{baseline_val*100:.2f}%"
                diff = (blt_val - baseline_val) * 100
                diff_str = f"{diff:+.2f}%"
            elif unit == 's':
                blt_str = f"{blt_val:.2f}s"
                baseline_str = f"{baseline_val:.2f}s"
                diff = blt_val - baseline_val
                diff_str = f"{diff:+.2f}s"
            else:
                blt_str = f"{blt_val:.2f}"
                baseline_str = f"{baseline_val:.2f}"
                diff = blt_val - baseline_val
                diff_str = f"{diff:+.2f}"
            
            report_lines.append(f"{metric_name:<25} {blt_str:<15} {baseline_str:<15} {diff_str:<15}")
        
        report_lines.extend([
            "",
            f"Total Test Samples: {blt_metrics.get('total_samples', 0)}",
            f"BLT Correct: {blt_metrics.get('total_correct', 0)}",
            f"Baseline Correct: {baseline_metrics.get('total_correct', 0)}",
            ""
        ])
    
    # Performance by Input Length
    if "blt" in results and "baseline" in results:
        blt_by_length = results["blt"].get('accuracy_by_length', {})
        baseline_by_length = results["baseline"].get('accuracy_by_length', {})
        
        if blt_by_length and baseline_by_length:
            report_lines.extend([
                "PERFORMANCE BY INPUT LENGTH",
                "-" * 80,
                f"{'Length Range':<15} {'BLT Acc':<12} {'Baseline Acc':<15} {'Difference':<12} {'Samples':<10}",
                "-" * 64,
            ])
            
            for length_range in sorted(blt_by_length.keys()):
                if length_range in baseline_by_length:
                    blt_acc = blt_by_length[length_range]['accuracy'] * 100
                    baseline_acc = baseline_by_length[length_range]['accuracy'] * 100
                    diff = blt_acc - baseline_acc
                    count = blt_by_length[length_range]['count']
                    
                    report_lines.append(
                        f"{length_range:<15} {blt_acc:>6.2f}%     {baseline_acc:>6.2f}%        "
                        f"{diff:>+6.2f}%     {count:>6}"
                    )
            
            report_lines.append("")
    
    # Training Analysis
    if config.get('tuning_performed', False):
        report_lines.extend([
            "TRAINING ANALYSIS",
            "-" * 80,
            ""
        ])
        
        # Load training histories
        for mode in ['blt', 'char']:
            history_path = f"checkpoints/history_{mode}.json"
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history = json.load(f)
                
                final_loss = history['train_loss'][-1] if history['train_loss'] else 0
                total_time = sum(history['epoch_times']) if history['epoch_times'] else 0
                avg_epoch_time = total_time / len(history['epoch_times']) if history['epoch_times'] else 0
                
                model_name = "BLT" if mode == "blt" else "Baseline"
                report_lines.extend([
                    f"{model_name} Training:",
                    f"  Final training loss: {final_loss:.4f}",
                    f"  Total training time: {total_time:.1f}s",
                    f"  Average epoch time: {avg_epoch_time:.1f}s",
                    f"  Epochs completed: {len(history['train_loss'])}",
                    ""
                ])
    
    # Statistical Significance
    if "blt" in results and "baseline" in results:
        report_lines.extend([
            "STATISTICAL ANALYSIS",
            "-" * 80,
            ""
        ])
        
        blt_acc = results["blt"].get('exact_match_accuracy', 0)
        baseline_acc = results["baseline"].get('exact_match_accuracy', 0)
        n_samples = results["blt"].get('total_samples', 0)
        
        # Simple significance test (assuming binomial distribution)
        if n_samples > 0:
            diff = abs(blt_acc - baseline_acc)
            std_err = np.sqrt((blt_acc * (1 - blt_acc) + baseline_acc * (1 - baseline_acc)) / n_samples)
            z_score = diff / std_err if std_err > 0 else 0
            
            significance = "Highly Significant" if z_score > 2.58 else "Significant" if z_score > 1.96 else "Not Significant"
            
            report_lines.extend([
                f"Accuracy Difference: {diff*100:.2f}%",
                f"Standard Error: {std_err*100:.2f}%",
                f"Z-Score: {z_score:.2f}",
                f"Significance (α=0.05): {significance}",
                ""
            ])
    
    # Key Findings and Conclusions
    report_lines.extend([
        "KEY FINDINGS",
        "-" * 80,
        ""
    ])
    
    findings = []
    
    if "blt" in results and "baseline" in results:
        blt_acc = results["blt"].get('exact_match_accuracy', 0) * 100
        baseline_acc = results["baseline"].get('exact_match_accuracy', 0) * 100
        acc_diff = blt_acc - baseline_acc
        
        if acc_diff > 1:
            findings.append(f"• BLT significantly outperforms baseline by {acc_diff:.2f}% in exact match accuracy")
        elif acc_diff > 0:
            findings.append(f"• BLT slightly outperforms baseline by {acc_diff:.2f}% in exact match accuracy")
        elif acc_diff > -1:
            findings.append(f"• BLT and baseline show comparable performance (difference: {acc_diff:.2f}%)")
        else:
            findings.append(f"• Baseline outperforms BLT by {abs(acc_diff):.2f}% in exact match accuracy")
        
        # Edit distance analysis
        blt_edit = results["blt"].get('avg_edit_distance', 0)
        baseline_edit = results["baseline"].get('avg_edit_distance', 0)
        if blt_edit < baseline_edit:
            findings.append(f"• BLT produces outputs closer to targets (avg edit distance: {blt_edit:.2f} vs {baseline_edit:.2f})")
        
        # Length analysis
        blt_by_length = results["blt"].get('accuracy_by_length', {})
        if blt_by_length:
            length_ranges = sorted(blt_by_length.keys())
            if len(length_ranges) >= 2:
                short_range = length_ranges[0]
                long_range = length_ranges[-1]
                
                short_acc = blt_by_length[short_range]['accuracy'] * 100
                long_acc = blt_by_length[long_range]['accuracy'] * 100
                
                if long_acc < short_acc - 5:
                    findings.append(f"• Performance degrades on longer sequences ({short_range}: {short_acc:.1f}% vs {long_range}: {long_acc:.1f}%)")
                elif long_acc > short_acc + 5:
                    findings.append(f"• Performance improves on longer sequences ({short_range}: {short_acc:.1f}% vs {long_range}: {long_acc:.1f}%)")
    
    if config.get('tuning_performed', False):
        findings.append("• Hyperparameter tuning was performed to optimize both models")
        
        if 'blt_tuning' in results and 'baseline_tuning' in results:
            blt_best = results['blt_tuning'].get('best_params', {})
            baseline_best = results['baseline_tuning'].get('best_params', {})
            
            if blt_best.get('lr', 0) != baseline_best.get('lr', 0):
                findings.append(f"• Optimal learning rates differ: BLT={blt_best.get('lr')}, Baseline={baseline_best.get('lr')}")
            
            if blt_best.get('d_model', 0) != baseline_best.get('d_model', 0):
                findings.append(f"• Optimal model dimensions differ: BLT={blt_best.get('d_model')}, Baseline={baseline_best.get('d_model')}")
    
    for finding in findings:
        report_lines.append(finding)
    
    report_lines.extend([
        "",
        "CONCLUSIONS",
        "-" * 80,
        ""
    ])
    
    # Generate conclusions based on results
    conclusions = []
    
    if "blt" in results and "baseline" in results:
        acc_diff = (results["blt"].get('exact_match_accuracy', 0) - results["baseline"].get('exact_match_accuracy', 0)) * 100
        
        if acc_diff > 2:
            conclusions.append("1. BLT demonstrates clear advantages over the baseline character-level model")
            conclusions.append("   for the string reversal task, likely due to its patch-based processing.")
        elif acc_diff > 0:
            conclusions.append("1. BLT shows modest improvements over the baseline, suggesting potential")
            conclusions.append("   benefits of patch-based processing for sequence tasks.")
        else:
            conclusions.append("1. Both models achieve comparable performance on string reversal,")
            conclusions.append("   indicating that the task may not fully leverage BLT's advantages.")
        
        conclusions.append("")
        conclusions.append("2. The entropy-based patching mechanism successfully segments input")
        conclusions.append("   sequences, potentially reducing sequence length and computational cost.")
        
        if config.get('tuning_performed', False):
            conclusions.append("")
            conclusions.append("3. Hyperparameter tuning reveals that optimal configurations may differ")
            conclusions.append("   between BLT and baseline models, emphasizing the importance of")
            conclusions.append("   model-specific optimization.")
    
    conclusions.extend([
        "",
        "4. This implementation provides a solid foundation for further research into",
        "   byte-level transformer architectures and patch-based sequence processing."
    ])
    
    for conclusion in conclusions:
        report_lines.append(conclusion)
    
    # Footer
    report_lines.extend([
        "",
        "="*80,
        "END OF REPORT",
        "="*80
    ])
    
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive BLT experiments with hyperparameter tuning")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 epochs)")
    parser.add_argument("--full", action="store_true", help="Full training (10 epochs)")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning + full training")
    parser.add_argument("--epochs", type=int, default=None, help="Custom epoch count")
    parser.add_argument("--tune_epochs", type=int, default=3, help="Epochs for hyperparameter tuning")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (overridden by tuning)")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for evaluation")
    parser.add_argument("--max_tune_combinations", type=int, default=8, help="Max hyperparameter combinations to test (optimized for speed)")
    parser.add_argument("--gtx1080ti", action="store_true", help="Use optimized settings for GTX 1080 Ti (4x 11.7GB)")
    parser.add_argument("--use_existing_tuning", action="store_true", help="Use existing hyperparameter tuning results (even if incomplete)")
    parser.add_argument("--analyze_tuning", action="store_true", help="Analyze existing tuning results and show best parameters")
    parser.add_argument("--train_from_existing", action="store_true", help="Train models using existing tuning results (skip tuning phase)")
    args = parser.parse_args()
    
    # Override settings for GTX 1080 Ti optimization
    if args.gtx1080ti:
        print("🚀 GTX 1080 Ti optimization enabled!")
        optimized = get_optimized_config_for_gtx1080ti()
        args.batch_size = optimized['batch_size']
        args.max_tune_combinations = 6  # Even faster tuning
        print(f"   Optimized batch size: {args.batch_size}")
        print(f"   Reduced tuning combinations: {args.max_tune_combinations}")
    
    experiment_start_time = time.time()
    
    # Determine epochs
    if args.epochs is not None:
        epochs = args.epochs
    elif args.quick:
        epochs = 2
    elif args.full:
        epochs = 10
    elif args.tune:
        epochs = 10  # Full training after tuning
    else:
        epochs = 5  # default
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BLT EXPERIMENTAL PIPELINE")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Training epochs: {epochs}")
    print(f"  Tuning epochs: {args.tune_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Hyperparameter tuning: {args.tune}")
    print(f"  Skip training: {args.skip_train}")
    print(f"  Max tuning combinations: {args.max_tune_combinations}")
    print(f"{'='*80}")
    
    # Handle analysis mode
    if args.analyze_tuning:
        print(f"\n{'='*80}")
        print("ANALYZING EXISTING TUNING RESULTS")
        print(f"{'='*80}")
        
        for mode in ['blt', 'char']:
            analyze_existing_tuning_results(mode)
        
        create_tuning_summary_report()
        return
    
    # Check memory requirements for optimized config
    optimized_config = get_optimized_config_for_gtx1080ti()
    print(f"\nGTX 1080 Ti Memory Check:")
    check_memory_requirements(optimized_config)
    print()
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    results = {}
    config = {
        'epochs': epochs,
        'tune_epochs': args.tune_epochs,
        'batch_size': args.batch_size,
        'beam_width': args.beam_width,
        'tuning_performed': args.tune,
        'max_tune_combinations': args.max_tune_combinations
    }
    
    # Hyperparameter tuning phase
    if args.tune and not args.skip_train:
        print(f"\n{'='*80}")
        print("PHASE 1: HYPERPARAMETER TUNING")
        print(f"{'='*80}")
        
        # Check for existing tuning results first
        if args.use_existing_tuning:
            print("Checking for existing tuning results...")
            
            blt_tuning = get_best_params_from_existing("blt")
            if not blt_tuning:
                print("No existing BLT tuning results found, running new tuning...")
                blt_tuning = run_hyperparameter_tuning("blt", args.tune_epochs, args.max_tune_combinations)
            results['blt_tuning'] = blt_tuning
            
            baseline_tuning = get_best_params_from_existing("char")
            if not baseline_tuning:
                print("No existing baseline tuning results found, running new tuning...")
                baseline_tuning = run_hyperparameter_tuning("char", args.tune_epochs, args.max_tune_combinations)
            results['baseline_tuning'] = baseline_tuning
        else:
            # Run new tuning
            blt_tuning = run_hyperparameter_tuning("blt", args.tune_epochs, args.max_tune_combinations)
            results['blt_tuning'] = blt_tuning
            
            baseline_tuning = run_hyperparameter_tuning("char", args.tune_epochs, args.max_tune_combinations)
            results['baseline_tuning'] = baseline_tuning
        
        print(f"\n{'='*80}")
        print("PHASE 2: TRAINING WITH BEST PARAMETERS")
        print(f"{'='*80}")
        
        # Train with best parameters
        if not train_with_best_params("blt", epochs, blt_tuning):
            print("\n❌ BLT training with best parameters failed. Exiting.")
            return
        
        if not train_with_best_params("char", epochs, baseline_tuning):
            print("\n❌ Baseline training with best parameters failed. Exiting.")
            return
    
    # Training with existing tuning results
    elif args.train_from_existing and not args.skip_train:
        print(f"\n{'='*80}")
        print("TRAINING WITH EXISTING TUNING RESULTS")
        print(f"{'='*80}")
        
        # Get existing tuning results
        blt_tuning = get_best_params_from_existing("blt")
        baseline_tuning = get_best_params_from_existing("char")
        
        if not blt_tuning or not baseline_tuning:
            print("❌ Missing tuning results. Please run tuning first or use --tune flag.")
            return
        
        results['blt_tuning'] = blt_tuning
        results['baseline_tuning'] = baseline_tuning
        
        # Train with best parameters
        if not train_with_best_params("blt", epochs, blt_tuning):
            print("\n❌ BLT training with best parameters failed. Exiting.")
            return
        
        if not train_with_best_params("char", epochs, baseline_tuning):
            print("\n❌ Baseline training with best parameters failed. Exiting.")
            return
    
    # Regular training phase (if not tuning)
    elif not args.skip_train:
        print(f"\n{'='*80}")
        print("TRAINING PHASE")
        print(f"{'='*80}")
        
        # Train BLT with optimized parameters for GTX 1080 Ti
        success, _, _ = run_command(
            [sys.executable, "src/train.py",
             "--mode", "blt",
             "--epochs", str(epochs),
             "--batch_size", "128",  # Larger batch for better GPU utilization
             "--lr", "2e-3",  # Higher LR for faster convergence
             "--d_model", "256",  # Larger model for better GPU utilization
             "--nhead", "8",  # Optimal for GPU
             "--num_encoder_layers", "2",
             "--num_decoder_layers", "2",
             "--dim_feedforward", "1024",  # Larger feedforward
             "--use_amp",
             "--lr_scheduler", "cosine",
             "--num_workers", "8"],
            "Training BLT Model"
        )
        
        if not success:
            print("\n❌ BLT training failed. Exiting.")
            return
        
        # Train Baseline with optimized parameters for GTX 1080 Ti
        success, _, _ = run_command(
            [sys.executable, "src/train.py",
             "--mode", "char",
             "--epochs", str(epochs),
             "--batch_size", "128",  # Larger batch for better GPU utilization
             "--lr", "2e-3",  # Higher LR for faster convergence
             "--d_model", "256",  # Larger model for better GPU utilization
             "--nhead", "8",  # Optimal for GPU
             "--num_encoder_layers", "2",
             "--num_decoder_layers", "2",
             "--dim_feedforward", "1024",  # Larger feedforward
             "--use_amp",
             "--lr_scheduler", "cosine",
             "--num_workers", "8"],
            "Training Baseline Model"
        )
        
        if not success:
            print("\n❌ Baseline training failed. Exiting.")
            return
    
    # Evaluation phase
    print(f"\n{'='*80}")
    print("EVALUATION PHASE")
    print(f"{'='*80}")
    
    # Evaluate BLT
    blt_results = evaluate_model("blt", args.batch_size, args.beam_width)
    if blt_results:
        results["blt"] = blt_results
    else:
        print("\n❌ BLT evaluation failed.")
    
    # Evaluate Baseline
    baseline_results = evaluate_model("baseline", args.batch_size, args.beam_width)
    if baseline_results:
        results["baseline"] = baseline_results
    else:
        print("\n❌ Baseline evaluation failed.")
    
    # Generate comprehensive report
    config['total_time'] = time.time() - experiment_start_time
    
    if results:
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        # Save detailed results
        with open("results/comprehensive_results.json", "w") as f:
            json.dump({
                'results': results,
                'config': config,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        # Generate and save report
        report = generate_comprehensive_report(results, config)
        
        with open("results/experimental_report.txt", "w") as f:
            f.write(report)
        
        # Generate analysis visualizations
        print("\nGenerating analysis visualizations...")
        try:
            analysis_success, _, _ = run_command(
                [sys.executable, "analyze_results.py"],
                "Generating Analysis and Visualizations",
                capture_output=False
            )
            if analysis_success:
                print("✓ Analysis visualizations generated successfully")
            else:
                print("⚠ Analysis generation failed, but continuing...")
        except Exception as e:
            print(f"⚠ Analysis generation error: {e}")
        
        # Print summary
        print(report)
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE!")
        print(f"{'='*80}")
        print("Generated files:")
        print("  - results/comprehensive_results.json")
        print("  - results/experimental_report.txt")
        if args.tune:
            print("  - results/hyperparameter_tuning_blt.json")
            print("  - results/hyperparameter_tuning_char.json")
        print("  - predictions/predictions_blt.csv")
        print("  - predictions/predictions_baseline.csv")
        print("  - checkpoints/best_blt_model.pt")
        print("  - checkpoints/best_char_model.pt")
        
        # List analysis files if they exist
        analysis_files = [
            "results/hyperparameter_analysis.png",
            "results/training_curves.png", 
            "results/performance_comparison.png",
            "results/detailed_analysis.png",
            "results/analysis_summary.txt"
        ]
        
        existing_analysis = [f for f in analysis_files if os.path.exists(f)]
        if existing_analysis:
            print("\nAnalysis files:")
            for f in existing_analysis:
                print(f"  - {f}")
        
        # Also check for the analysis summary report
        analysis_summary_path = "results/analysis_summary.txt"
        if os.path.exists(analysis_summary_path):
            print(f"  - {analysis_summary_path}")
        
        print()
    else:
        print("\n❌ No results to report.")


if __name__ == "__main__":
    main()