# src/eval.py
"""
Evaluation script: generates predictions and computes metrics.

Usage:
    python src/eval.py --mode blt --checkpoint checkpoints/best_blt_model.pt --test_csv data/test.csv
    python src/eval.py --mode char --checkpoint checkpoints/best_char_model.pt --test_csv data/test.csv
"""

import argparse
import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from tokenizer import Tokenizer
from baseline_model import BaselineModel
from blt_model import BLTModel
from dataset import preprocess_csv_to_pt, BLTProcessedDataset, CharProcessedDataset, collate_fn
from torch.utils.data import DataLoader


def compute_metrics(predictions, targets, inputs):
    """Compute comprehensive accuracy metrics."""
    exact_match = sum(p == t for p, t in zip(predictions, targets))
    total = len(predictions)
    
    char_correct = 0
    char_total = 0
    edit_distances = []
    length_diffs = []
    
    for pred, tgt in zip(predictions, targets):
        # Character accuracy
        for p_char, t_char in zip(pred, tgt):
            if p_char == t_char:
                char_correct += 1
            char_total += 1
        char_total += abs(len(pred) - len(tgt))
        
        # Edit distance (Levenshtein)
        edit_dist = compute_edit_distance(pred, tgt)
        edit_distances.append(edit_dist)
        
        # Length difference
        length_diffs.append(abs(len(pred) - len(tgt)))
    
    # Compute metrics by input length bins
    length_bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 1000)]
    accuracy_by_length = {}
    
    for min_len, max_len in length_bins:
        bin_name = f"{min_len}-{max_len}"
        bin_correct = 0
        bin_total = 0
        
        for inp, pred, tgt in zip(inputs, predictions, targets):
            if min_len <= len(inp) < max_len:
                if pred == tgt:
                    bin_correct += 1
                bin_total += 1
        
        if bin_total > 0:
            accuracy_by_length[bin_name] = {
                "accuracy": bin_correct / bin_total,
                "count": bin_total
            }
    
    return {
        "exact_match_accuracy": exact_match / total if total > 0 else 0.0,
        "character_accuracy": char_correct / char_total if char_total > 0 else 0.0,
        "total_samples": total,
        "total_correct": exact_match,
        "total_incorrect": total - exact_match,
        "avg_edit_distance": sum(edit_distances) / len(edit_distances) if edit_distances else 0.0,
        "avg_length_diff": sum(length_diffs) / len(length_diffs) if length_diffs else 0.0,
        "max_edit_distance": max(edit_distances) if edit_distances else 0,
        "accuracy_by_length": accuracy_by_length
    }


def compute_edit_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and generate predictions")
    parser.add_argument("--mode", type=str, required=True, choices=["blt", "char", "baseline"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    # Map baseline mode to char for internal processing
    internal_mode = "char" if args.mode == "baseline" else args.mode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load checkpoint with error handling
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_args = checkpoint.get("args", {})
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Load tokenizer with error handling
    tokenizer_path = args.tokenizer or f"data/processed/tokenizer_{internal_mode}.json"
    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = Tokenizer.load(tokenizer_path)
        vocab_size = tokenizer.vocab_size
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Preprocess test data
    print(f"\nPreprocessing test data...")
    test_pt = f"data/processed/test_{internal_mode}.pt"
    test_pt, _ = preprocess_csv_to_pt(
        csv_path=args.test_csv,
        out_pt=test_pt,
        mode=internal_mode,
        tokenizer=tokenizer,
        W=10,
        entropy_threshold=2.0,
        max_patch_len=15,
        buckets=4096,
        seed=1337,
        ngrams=(1, 2, 3),
        force_rebuild=False,
        manifest_out=f"data/processed/test_{internal_mode}_manifest.json",
        tokenizer_out=tokenizer_path,
        strict_ascii=False
    )
    
    # Create dataset and dataloader
    if internal_mode == "blt":
        test_ds = BLTProcessedDataset(test_pt)
    else:
        test_ds = CharProcessedDataset(test_pt)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, mode=internal_mode, pad_id=tokenizer.pad_id)
    )
    
    # Create model
    print(f"Creating {args.mode.upper()} model...")
    if internal_mode == "char":
        model = BaselineModel(
            vocab_size=vocab_size,
            d_model=model_args.get("d_model", 128),
            nhead=model_args.get("nhead", 4),
            num_encoder_layers=model_args.get("num_encoder_layers", 2),
            num_decoder_layers=model_args.get("num_decoder_layers", 2),
            dim_feedforward=model_args.get("dim_feedforward", 256),
            dropout=0.0
        )
    else:
        model = BLTModel(
            vocab_size=vocab_size,
            d_model=model_args.get("d_model", 128),
            nhead=model_args.get("nhead", 4),
            num_encoder_layers=model_args.get("num_encoder_layers", 2),
            num_decoder_layers=model_args.get("num_decoder_layers", 2),
            dim_feedforward=model_args.get("dim_feedforward", 256),
            dropout=0.0,
            max_patches=512,
            buckets=4096,
            ngrams=(1, 2, 3)
        )
    
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
    except (KeyError, RuntimeError) as e:
        print(f"Error loading model state: {e}")
        return
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Beam width: {args.beam_width}\n")
    
    # Generate predictions
    print("Generating predictions...")
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    # Clear cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            raw_inputs = batch["raw_inputs"]
            
            if internal_mode == "char":
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                input_mask = batch["input_mask"].to(device, non_blocking=True)
                
                if args.beam_width > 1:
                    decoded_ids_batch = model.beam_search_decode(
                        input_ids, input_mask,
                        sos_id=tokenizer.sos_id,
                        eos_id=tokenizer.eos_id,
                        max_len=args.max_len,
                        beam_width=args.beam_width,
                        length_penalty=args.length_penalty
                    )
                else:
                    decoded_ids_batch = model.greedy_decode(
                        input_ids, input_mask,
                        sos_id=tokenizer.sos_id,
                        eos_id=tokenizer.eos_id,
                        max_len=args.max_len
                    )
            else:
                patches = batch["patches"]
                patch_mask = batch["patch_mask"].to(device, non_blocking=True)
                
                if args.beam_width > 1:
                    decoded_ids_batch = model.beam_search_decode(
                        patches, patch_mask,
                        sos_id=tokenizer.sos_id,
                        eos_id=tokenizer.eos_id,
                        max_len=args.max_len,
                        beam_width=args.beam_width,
                        length_penalty=args.length_penalty
                    )
                else:
                    decoded_ids_batch = model.greedy_decode(
                        patches, patch_mask,
                        sos_id=tokenizer.sos_id,
                        eos_id=tokenizer.eos_id,
                        max_len=args.max_len
                    )
            
            # Decode predictions
            for decoded_ids in decoded_ids_batch:
                pred_text = tokenizer.decode(decoded_ids, strip_specials=True)
                all_predictions.append(pred_text)
            
            # Get targets
            target_ids = batch["target_ids"]
            for i in range(len(target_ids)):
                tgt_ids = target_ids[i].tolist()
                tgt_text = tokenizer.decode(tgt_ids, strip_specials=True)
                all_targets.append(tgt_text)
            
            all_inputs.extend(raw_inputs)
            
            # Clear variables to free memory
            del batch, target_ids, decoded_ids_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_predictions, all_targets, all_inputs)
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['total_correct']} | Incorrect: {metrics['total_incorrect']}")
    print(f"Exact match accuracy: {metrics['exact_match_accuracy']*100:.2f}%")
    print(f"Character accuracy: {metrics['character_accuracy']*100:.2f}%")
    print(f"Avg edit distance: {metrics['avg_edit_distance']:.2f}")
    print(f"Avg length difference: {metrics['avg_length_diff']:.2f}")
    
    if metrics['accuracy_by_length']:
        print(f"\nAccuracy by Input Length:")
        for bin_name, bin_metrics in metrics['accuracy_by_length'].items():
            print(f"  {bin_name:>10} chars: {bin_metrics['accuracy']*100:5.2f}% (n={bin_metrics['count']})")
    
    print(f"{'='*60}\n")
    
    # Save predictions
    if args.output_csv is None:
        args.output_csv = f"predictions/predictions_{args.mode}.csv"
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    results_df = pd.DataFrame({
        "input": all_inputs,
        "target": all_targets,
        "prediction": all_predictions,
        "correct": [p == t for p, t in zip(all_predictions, all_targets)]
    })
    
    results_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    
    # Save metrics
    metrics_path = args.output_csv.replace(".csv", "_metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
