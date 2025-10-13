#!/usr/bin/env python3
"""
Complete experimental pipeline for BLT vs Baseline comparison.

This script runs the full pipeline:
1. Train both models
2. Evaluate on test set
3. Generate comparison report

Usage:
    python run_experiments.py --quick  # Quick test with small epochs
    python run_experiments.py --full   # Full training
"""

import argparse
import os
import subprocess
import json
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with return code {result.returncode}")
        return False
    
    print(f"\n✓ Completed in {elapsed:.2f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run complete BLT experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 epochs)")
    parser.add_argument("--full", action="store_true", help="Full training (10 epochs)")
    parser.add_argument("--epochs", type=int, default=None, help="Custom epoch count")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for evaluation")
    args = parser.parse_args()
    
    # Determine epochs
    if args.epochs is not None:
        epochs = args.epochs
    elif args.quick:
        epochs = 2
    elif args.full:
        epochs = 10
    else:
        epochs = 5  # default
    
    print(f"\n{'='*60}")
    print(f"BLT Experimental Pipeline")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Skip training: {args.skip_train}")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    results = {}
    
    # Training phase
    if not args.skip_train:
        # Train BLT
        success = run_command(
            ["python", "src/train.py",
             "--mode", "blt",
             "--epochs", str(epochs),
             "--batch_size", str(args.batch_size),
             "--lr", "1e-3",
             "--d_model", "128",
             "--nhead", "4",
             "--num_encoder_layers", "2",
             "--num_decoder_layers", "2",
             "--dim_feedforward", "256"],
            "Training BLT Model"
        )
        
        if not success:
            print("\n❌ BLT training failed. Exiting.")
            return
        
        # Train Baseline
        success = run_command(
            ["python", "src/train.py",
             "--mode", "char",
             "--epochs", str(epochs),
             "--batch_size", str(args.batch_size),
             "--lr", "1e-3",
             "--d_model", "128",
             "--nhead", "4",
             "--num_encoder_layers", "2",
             "--num_decoder_layers", "2",
             "--dim_feedforward", "256"],
            "Training Baseline Model"
        )
        
        if not success:
            print("\n❌ Baseline training failed. Exiting.")
            return
    
    # Evaluation phase
    # Evaluate BLT
    success = run_command(
        ["python", "src/eval.py",
         "--mode", "blt",
         "--checkpoint", "checkpoints/best_blt_model.pt",
         "--test_csv", "data/test.csv",
         "--batch_size", str(args.batch_size),
         "--beam_width", str(args.beam_width),
         "--output_csv", "predictions/predictions_blt.csv"],
        "Evaluating BLT Model"
    )
    
    if not success:
        print("\n❌ BLT evaluation failed.")
    else:
        # Load BLT metrics
        metrics_path = "predictions/predictions_blt_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                results["blt"] = json.load(f)
    
    # Evaluate Baseline
    success = run_command(
        ["python", "src/eval.py",
         "--mode", "char",
         "--checkpoint", "checkpoints/best_char_model.pt",
         "--test_csv", "data/test.csv",
         "--batch_size", str(args.batch_size),
         "--beam_width", str(args.beam_width),
         "--output_csv", "predictions/predictions_baseline.csv"],
        "Evaluating Baseline Model"
    )
    
    if not success:
        print("\n❌ Baseline evaluation failed.")
    else:
        # Load Baseline metrics
        metrics_path = "predictions/predictions_baseline_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                results["baseline"] = json.load(f)
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("Experimental Results Summary")
    print(f"{'='*60}\n")
    
    if "blt" in results and "baseline" in results:
        print("BLT Model:")
        print(f"  Exact Match Accuracy: {results['blt']['exact_match_accuracy']*100:.2f}%")
        print(f"  Character Accuracy: {results['blt']['character_accuracy']*100:.2f}%")
        print(f"  Total Samples: {results['blt']['total_samples']}")
        
        print("\nBaseline Model:")
        print(f"  Exact Match Accuracy: {results['baseline']['exact_match_accuracy']*100:.2f}%")
        print(f"  Character Accuracy: {results['baseline']['character_accuracy']*100:.2f}%")
        print(f"  Total Samples: {results['baseline']['total_samples']}")
        
        print("\nComparison:")
        em_diff = (results['blt']['exact_match_accuracy'] - results['baseline']['exact_match_accuracy']) * 100
        char_diff = (results['blt']['character_accuracy'] - results['baseline']['character_accuracy']) * 100
        
        print(f"  Exact Match Δ: {em_diff:+.2f}%")
        print(f"  Character Accuracy Δ: {char_diff:+.2f}%")
        
        # Calculate additional comparisons
        edit_dist_delta = results['blt'].get('avg_edit_distance', 0) - results['baseline'].get('avg_edit_distance', 0)
        
        # Save comparison
        comparison = {
            "blt": results["blt"],
            "baseline": results["baseline"],
            "comparison": {
                "exact_match_delta": em_diff,
                "character_accuracy_delta": char_diff,
                "edit_distance_delta": edit_dist_delta,
                "blt_better_count": results['blt'].get('total_correct', 0) - results['baseline'].get('total_correct', 0)
            },
            "config": {
                "epochs": epochs,
                "batch_size": args.batch_size,
                "beam_width": args.beam_width
            }
        }
        
        with open("predictions/comparison_results.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✓ Comparison saved to predictions/comparison_results.json")
        
        # Print additional insights
        if edit_dist_delta < 0:
            print(f"  → BLT has {abs(edit_dist_delta):.2f} lower average edit distance")
        elif edit_dist_delta > 0:
            print(f"  → Baseline has {edit_dist_delta:.2f} lower average edit distance")
    
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}\n")
    print("Generated files:")
    print("  - checkpoints/best_blt_model.pt")
    print("  - checkpoints/best_char_model.pt")
    print("  - predictions/predictions_blt.csv")
    print("  - predictions/predictions_baseline.csv")
    print("  - predictions/comparison_results.json")
    print()


if __name__ == "__main__":
    main()
