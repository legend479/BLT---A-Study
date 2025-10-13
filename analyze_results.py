#!/usr/bin/env python3
"""
Analyze and visualize experimental results.

Usage:
    python analyze_results.py
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_json(path):
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_training_history():
    """Plot training loss curves."""
    blt_history = load_json("checkpoints/history_blt.json")
    char_history = load_json("checkpoints/history_char.json")
    
    if not blt_history or not char_history:
        print("⚠ Training history files not found. Run training first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    ax = axes[0, 0]
    epochs_blt = range(1, len(blt_history["train_loss"]) + 1)
    epochs_char = range(1, len(char_history["train_loss"]) + 1)
    
    ax.plot(epochs_blt, blt_history["train_loss"], 'o-', label='BLT', linewidth=2, markersize=6)
    ax.plot(epochs_char, char_history["train_loss"], 's-', label='Baseline', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Validation loss (if available) or epoch times
    ax = axes[0, 1]
    if "val_loss" in blt_history and blt_history["val_loss"]:
        ax.plot(epochs_blt, blt_history["val_loss"], 'o-', label='BLT', linewidth=2, markersize=6)
        ax.plot(epochs_char, char_history["val_loss"], 's-', label='Baseline', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.plot(epochs_blt, blt_history["epoch_times"], 'o-', label='BLT', linewidth=2, markersize=6)
        ax.plot(epochs_char, char_history["epoch_times"], 's-', label='Baseline', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Loss improvement rate
    ax = axes[1, 0]
    if len(blt_history["train_loss"]) > 1:
        blt_improvements = [blt_history["train_loss"][i-1] - blt_history["train_loss"][i] 
                           for i in range(1, len(blt_history["train_loss"]))]
        char_improvements = [char_history["train_loss"][i-1] - char_history["train_loss"][i] 
                            for i in range(1, len(char_history["train_loss"]))]
        
        ax.plot(range(2, len(blt_history["train_loss"]) + 1), blt_improvements, 'o-', 
                label='BLT', linewidth=2, markersize=6)
        ax.plot(range(2, len(char_history["train_loss"]) + 1), char_improvements, 's-', 
                label='Baseline', linewidth=2, markersize=6)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Improvement', fontsize=12)
        ax.set_title('Loss Improvement per Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Cumulative training time
    ax = axes[1, 1]
    blt_cumulative = [sum(blt_history["epoch_times"][:i+1]) for i in range(len(blt_history["epoch_times"]))]
    char_cumulative = [sum(char_history["epoch_times"][:i+1]) for i in range(len(char_history["epoch_times"]))]
    
    ax.plot(epochs_blt, blt_cumulative, 'o-', label='BLT', linewidth=2, markersize=6)
    ax.plot(epochs_char, char_cumulative, 's-', label='Baseline', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax.set_title('Cumulative Training Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions/training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Training curves saved to predictions/training_curves.png")
    plt.close()


def analyze_predictions():
    """Analyze prediction results."""
    blt_df = pd.read_csv("predictions/predictions_blt.csv") if os.path.exists("predictions/predictions_blt.csv") else None
    char_df = pd.read_csv("predictions/predictions_baseline.csv") if os.path.exists("predictions/predictions_baseline.csv") else None
    
    if blt_df is None or char_df is None:
        print("⚠ Prediction files not found. Run evaluation first.")
        return
    
    print("\n" + "="*60)
    print("Prediction Analysis")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total samples: {len(blt_df)}")
    print(f"  BLT correct: {blt_df['correct'].sum()} ({blt_df['correct'].mean()*100:.2f}%)")
    print(f"  Baseline correct: {char_df['correct'].sum()} ({char_df['correct'].mean()*100:.2f}%)")
    
    # Length analysis
    blt_df['input_len'] = blt_df['input'].str.len()
    char_df['input_len'] = char_df['input'].str.len()
    
    print("\nAccuracy by Input Length:")
    for length_bin in [(0, 20), (20, 50), (50, 100), (100, 1000)]:
        min_len, max_len = length_bin
        blt_mask = (blt_df['input_len'] >= min_len) & (blt_df['input_len'] < max_len)
        char_mask = (char_df['input_len'] >= min_len) & (char_df['input_len'] < max_len)
        
        if blt_mask.sum() > 0:
            blt_acc = blt_df[blt_mask]['correct'].mean() * 100
            char_acc = char_df[char_mask]['correct'].mean() * 100
            count = blt_mask.sum()
            print(f"  {min_len:3d}-{max_len:3d} chars (n={count:4d}): BLT={blt_acc:5.2f}% | Baseline={char_acc:5.2f}%")
    
    # Error analysis
    print("\nError Analysis:")
    blt_errors = blt_df[~blt_df['correct']]
    char_errors = char_df[~char_df['correct']]
    
    print(f"  BLT errors: {len(blt_errors)}")
    print(f"  Baseline errors: {len(char_errors)}")
    print(f"  Both wrong: {len(blt_errors.merge(char_errors, on='input'))}")
    print(f"  Only BLT wrong: {len(blt_errors[~blt_errors['input'].isin(char_errors['input'])])}")
    print(f"  Only Baseline wrong: {len(char_errors[~char_errors['input'].isin(blt_errors['input'])])}")
    
    # Sample errors
    print("\nSample Errors (BLT):")
    for i, row in blt_errors.head(3).iterrows():
        print(f"  Input: {row['input'][:50]}...")
        print(f"  Target: {row['target'][:50]}...")
        print(f"  Prediction: {row['prediction'][:50]}...")
        print()
    
    # Visualize accuracy distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy by length
    ax = axes[0]
    length_bins = [0, 20, 40, 60, 80, 100, 150, 200, 1000]
    blt_df['length_bin'] = pd.cut(blt_df['input_len'], bins=length_bins)
    char_df['length_bin'] = pd.cut(char_df['input_len'], bins=length_bins)
    
    blt_acc_by_len = blt_df.groupby('length_bin')['correct'].mean() * 100
    char_acc_by_len = char_df.groupby('length_bin')['correct'].mean() * 100
    
    x = range(len(blt_acc_by_len))
    width = 0.35
    ax.bar([i - width/2 for i in x], blt_acc_by_len.values, width, label='BLT', alpha=0.8)
    ax.bar([i + width/2 for i in x], char_acc_by_len.values, width, label='Baseline', alpha=0.8)
    ax.set_xlabel('Input Length', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Input Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in blt_acc_by_len.index], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error distribution
    ax = axes[1]
    categories = ['Both Correct', 'Only BLT Correct', 'Only Baseline Correct', 'Both Wrong']
    both_correct = ((blt_df['correct']) & (char_df['correct'])).sum()
    only_blt = ((blt_df['correct']) & (~char_df['correct'])).sum()
    only_char = ((~blt_df['correct']) & (char_df['correct'])).sum()
    both_wrong = ((~blt_df['correct']) & (~char_df['correct'])).sum()
    
    counts = [both_correct, only_blt, only_char, both_wrong]
    colors = ['green', 'blue', 'orange', 'red']
    
    ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Prediction Agreement', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts):
        ax.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('predictions/accuracy_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Accuracy analysis saved to predictions/accuracy_analysis.png")
    plt.close()


def plot_detailed_metrics():
    """Plot detailed metrics comparison."""
    blt_metrics = load_json("predictions/predictions_blt_metrics.json")
    char_metrics = load_json("predictions/predictions_baseline_metrics.json")
    
    if not blt_metrics or not char_metrics:
        print("⚠ Metrics files not found.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Metric comparison bar chart
    ax = axes[0, 0]
    metrics_names = ['Exact Match', 'Character Acc']
    blt_values = [blt_metrics['exact_match_accuracy'] * 100, blt_metrics['character_accuracy'] * 100]
    char_values = [char_metrics['exact_match_accuracy'] * 100, char_metrics['character_accuracy'] * 100]
    
    x = range(len(metrics_names))
    width = 0.35
    ax.bar([i - width/2 for i in x], blt_values, width, label='BLT', alpha=0.8, color='steelblue')
    ax.bar([i + width/2 for i in x], char_values, width, label='Baseline', alpha=0.8, color='coral')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for i, (bv, cv) in enumerate(zip(blt_values, char_values)):
        ax.text(i - width/2, bv + 1, f'{bv:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(i + width/2, cv + 1, f'{cv:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Edit distance comparison
    ax = axes[0, 1]
    if 'avg_edit_distance' in blt_metrics and 'avg_edit_distance' in char_metrics:
        metrics_names = ['Avg Edit Dist', 'Max Edit Dist', 'Avg Length Diff']
        blt_values = [
            blt_metrics.get('avg_edit_distance', 0),
            blt_metrics.get('max_edit_distance', 0),
            blt_metrics.get('avg_length_diff', 0)
        ]
        char_values = [
            char_metrics.get('avg_edit_distance', 0),
            char_metrics.get('max_edit_distance', 0),
            char_metrics.get('avg_length_diff', 0)
        ]
        
        x = range(len(metrics_names))
        ax.bar([i - width/2 for i in x], blt_values, width, label='BLT', alpha=0.8, color='steelblue')
        ax.bar([i + width/2 for i in x], char_values, width, label='Baseline', alpha=0.8, color='coral')
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Error Distance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Accuracy by length comparison
    ax = axes[1, 0]
    if 'accuracy_by_length' in blt_metrics and blt_metrics['accuracy_by_length']:
        length_bins = sorted(blt_metrics['accuracy_by_length'].keys())
        blt_accs = [blt_metrics['accuracy_by_length'][lb]['accuracy'] * 100 for lb in length_bins]
        char_accs = [char_metrics['accuracy_by_length'].get(lb, {}).get('accuracy', 0) * 100 for lb in length_bins]
        
        x = range(len(length_bins))
        ax.plot(x, blt_accs, 'o-', label='BLT', linewidth=2, markersize=8, color='steelblue')
        ax.plot(x, char_accs, 's-', label='Baseline', linewidth=2, markersize=8, color='coral')
        ax.set_xlabel('Input Length Range', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy by Input Length', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(length_bins, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    
    # Sample counts by length
    ax = axes[1, 1]
    if 'accuracy_by_length' in blt_metrics and blt_metrics['accuracy_by_length']:
        length_bins = sorted(blt_metrics['accuracy_by_length'].keys())
        counts = [blt_metrics['accuracy_by_length'][lb]['count'] for lb in length_bins]
        
        ax.bar(range(len(length_bins)), counts, alpha=0.7, color='mediumpurple')
        ax.set_xlabel('Input Length Range', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_title('Test Set Distribution by Length', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(length_bins)))
        ax.set_xticklabels(length_bins, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('predictions/detailed_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Detailed metrics saved to predictions/detailed_metrics.png")
    plt.close()


def generate_summary_report():
    """Generate a comprehensive summary report."""
    comparison = load_json("predictions/comparison_results.json")
    blt_metrics = load_json("predictions/predictions_blt_metrics.json")
    char_metrics = load_json("predictions/predictions_baseline_metrics.json")
    blt_history = load_json("checkpoints/history_blt.json")
    char_history = load_json("checkpoints/history_char.json")
    
    if not comparison:
        print("⚠ Comparison results not found. Run evaluation first.")
        return
    
    report = []
    report.append("="*80)
    report.append("BLT vs Baseline: Comprehensive Evaluation Report")
    report.append("="*80)
    report.append("")
    
    # Configuration
    report.append("CONFIGURATION")
    report.append("-" * 80)
    config = comparison.get("config", {})
    report.append(f"  Training Epochs:     {config.get('epochs', 'N/A')}")
    report.append(f"  Batch Size:          {config.get('batch_size', 'N/A')}")
    report.append(f"  Beam Width:          {config.get('beam_width', 'N/A')}")
    report.append("")
    
    # Training Statistics
    if blt_history and char_history:
        report.append("TRAINING STATISTICS")
        report.append("-" * 80)
        report.append(f"  BLT Model:")
        report.append(f"    Final Training Loss:     {blt_history['train_loss'][-1]:.4f}")
        report.append(f"    Total Training Time:     {sum(blt_history['epoch_times']):.1f}s")
        report.append(f"    Avg Time per Epoch:      {sum(blt_history['epoch_times'])/len(blt_history['epoch_times']):.1f}s")
        report.append("")
        report.append(f"  Baseline Model:")
        report.append(f"    Final Training Loss:     {char_history['train_loss'][-1]:.4f}")
        report.append(f"    Total Training Time:     {sum(char_history['epoch_times']):.1f}s")
        report.append(f"    Avg Time per Epoch:      {sum(char_history['epoch_times'])/len(char_history['epoch_times']):.1f}s")
        report.append("")
    
    # Evaluation Results
    report.append("EVALUATION RESULTS")
    report.append("-" * 80)
    blt = comparison.get("blt", {})
    baseline = comparison.get("baseline", {})
    
    report.append(f"  BLT Model:")
    report.append(f"    Total Samples:           {blt.get('total_samples', 0)}")
    report.append(f"    Correct Predictions:     {blt.get('total_correct', 0)}")
    report.append(f"    Incorrect Predictions:   {blt.get('total_incorrect', 0)}")
    report.append(f"    Exact Match Accuracy:    {blt.get('exact_match_accuracy', 0)*100:.2f}%")
    report.append(f"    Character Accuracy:      {blt.get('character_accuracy', 0)*100:.2f}%")
    if 'avg_edit_distance' in blt:
        report.append(f"    Avg Edit Distance:       {blt.get('avg_edit_distance', 0):.2f}")
        report.append(f"    Max Edit Distance:       {blt.get('max_edit_distance', 0)}")
        report.append(f"    Avg Length Difference:   {blt.get('avg_length_diff', 0):.2f}")
    report.append("")
    
    report.append(f"  Baseline Model:")
    report.append(f"    Total Samples:           {baseline.get('total_samples', 0)}")
    report.append(f"    Correct Predictions:     {baseline.get('total_correct', 0)}")
    report.append(f"    Incorrect Predictions:   {baseline.get('total_incorrect', 0)}")
    report.append(f"    Exact Match Accuracy:    {baseline.get('exact_match_accuracy', 0)*100:.2f}%")
    report.append(f"    Character Accuracy:      {baseline.get('character_accuracy', 0)*100:.2f}%")
    if 'avg_edit_distance' in baseline:
        report.append(f"    Avg Edit Distance:       {baseline.get('avg_edit_distance', 0):.2f}")
        report.append(f"    Max Edit Distance:       {baseline.get('max_edit_distance', 0)}")
        report.append(f"    Avg Length Difference:   {baseline.get('avg_length_diff', 0):.2f}")
    report.append("")
    
    # Comparison
    comp = comparison.get("comparison", {})
    report.append("COMPARATIVE ANALYSIS")
    report.append("-" * 80)
    report.append(f"  Exact Match Accuracy Δ:  {comp.get('exact_match_delta', 0):+.2f}%")
    report.append(f"  Character Accuracy Δ:    {comp.get('character_accuracy_delta', 0):+.2f}%")
    
    if blt_metrics and char_metrics and 'avg_edit_distance' in blt_metrics:
        edit_dist_delta = blt_metrics['avg_edit_distance'] - char_metrics['avg_edit_distance']
        report.append(f"  Avg Edit Distance Δ:     {edit_dist_delta:+.2f}")
    report.append("")
    
    # Accuracy by Length
    if blt_metrics and 'accuracy_by_length' in blt_metrics and blt_metrics['accuracy_by_length']:
        report.append("ACCURACY BY INPUT LENGTH")
        report.append("-" * 80)
        report.append(f"  {'Length Range':<15} {'BLT Acc':<12} {'Baseline Acc':<15} {'Samples':<10}")
        report.append(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*10}")
        
        for length_bin in sorted(blt_metrics['accuracy_by_length'].keys()):
            blt_acc = blt_metrics['accuracy_by_length'][length_bin]['accuracy'] * 100
            char_acc = char_metrics['accuracy_by_length'].get(length_bin, {}).get('accuracy', 0) * 100
            count = blt_metrics['accuracy_by_length'][length_bin]['count']
            report.append(f"  {length_bin:<15} {blt_acc:>6.2f}%      {char_acc:>6.2f}%         {count:>6}")
        report.append("")
    
    # Conclusion
    em_delta = comp.get('exact_match_delta', 0)
    char_delta = comp.get('character_accuracy_delta', 0)
    
    report.append("CONCLUSION")
    report.append("-" * 80)
    
    if em_delta > 1.0:
        conclusion = "BLT shows significant improvement over baseline."
        recommendation = "BLT is recommended for this task."
    elif em_delta > 0:
        conclusion = "BLT shows slight improvement over baseline."
        recommendation = "BLT offers marginal benefits; choice depends on other factors."
    elif em_delta > -1.0:
        conclusion = "BLT and baseline perform similarly."
        recommendation = "Either model is suitable; consider efficiency and complexity."
    else:
        conclusion = "Baseline outperforms BLT."
        recommendation = "Baseline is recommended for this task."
    
    report.append(f"  Performance: {conclusion}")
    report.append(f"  Recommendation: {recommendation}")
    report.append("")
    
    # Key Findings
    report.append("KEY FINDINGS")
    report.append("-" * 80)
    findings = []
    
    if abs(em_delta) < 0.5:
        findings.append("  • Both models achieve comparable accuracy on string reversal")
    
    if blt_history and char_history:
        time_ratio = sum(blt_history['epoch_times']) / sum(char_history['epoch_times'])
        if time_ratio < 0.9:
            findings.append(f"  • BLT trains {(1-time_ratio)*100:.1f}% faster than baseline")
        elif time_ratio > 1.1:
            findings.append(f"  • Baseline trains {(time_ratio-1)*100:.1f}% faster than BLT")
    
    if blt_metrics and 'accuracy_by_length' in blt_metrics:
        # Check if BLT performs better on longer sequences
        length_bins = sorted(blt_metrics['accuracy_by_length'].keys())
        if len(length_bins) >= 2:
            long_bin = length_bins[-1]
            blt_long = blt_metrics['accuracy_by_length'][long_bin]['accuracy']
            char_long = char_metrics['accuracy_by_length'].get(long_bin, {}).get('accuracy', 0)
            if blt_long > char_long + 0.01:
                findings.append(f"  • BLT performs better on longer sequences ({long_bin} chars)")
            elif char_long > blt_long + 0.01:
                findings.append(f"  • Baseline performs better on longer sequences ({long_bin} chars)")
    
    if char_delta > 0:
        findings.append(f"  • BLT achieves {char_delta:.2f}% higher character-level accuracy")
    elif char_delta < 0:
        findings.append(f"  • Baseline achieves {abs(char_delta):.2f}% higher character-level accuracy")
    
    for finding in findings:
        report.append(finding)
    
    report.append("")
    report.append("="*80)
    report.append("Report generated by analyze_results.py")
    report.append("="*80)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    with open("predictions/summary_report.txt", "w") as f:
        f.write(report_text)
    print("\n✓ Comprehensive summary report saved to predictions/summary_report.txt")


def main():
    print("\n" + "="*60)
    print("BLT Results Analysis")
    print("="*60)
    
    # Check if results exist
    if not os.path.exists("predictions"):
        print("\n⚠ No predictions directory found. Run experiments first.")
        return
    
    # Plot training curves
    print("\n1. Analyzing training history...")
    plot_training_history()
    
    # Analyze predictions
    print("\n2. Analyzing predictions...")
    analyze_predictions()
    
    # Plot detailed metrics
    print("\n3. Plotting detailed metrics...")
    plot_detailed_metrics()
    
    # Generate summary
    print("\n4. Generating summary report...")
    generate_summary_report()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - predictions/training_curves.png")
    print("  - predictions/accuracy_analysis.png")
    print("  - predictions/detailed_metrics.png")
    print("  - predictions/summary_report.txt")
    print()


if __name__ == "__main__":
    main()
