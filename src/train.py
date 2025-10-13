# src/train.py
"""
Training script for both BLT and Baseline models.

Usage:
    python src/train.py --mode blt --epochs 10 --batch_size 32
    python src/train.py --mode char --epochs 10 --batch_size 32
"""

import argparse
import os
import json
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer import Tokenizer
from baseline_model import BaselineModel
from blt_model import BLTModel
from dataset import preprocess_csv_to_pt, make_dataloaders


def train_epoch(model, loader: DataLoader, optimizer, criterion, device, mode: str, pad_id: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)
        
        if mode == "char":
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            logits = model(input_ids, input_mask, target_ids, target_mask)
        else:  # blt
            patches = batch["patches"]
            patch_mask = batch["patch_mask"].to(device)
            logits = model(patches, patch_mask, target_ids, target_mask)
        
        # Compute loss (ignore padding)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), target_ids.reshape(B * T))
        
        # Mask out padding positions
        mask_flat = target_mask.reshape(B * T)
        loss = (loss * mask_flat.float()).sum() / mask_flat.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * mask_flat.sum().item()
        total_tokens += mask_flat.sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return {"loss": total_loss / total_tokens if total_tokens > 0 else 0.0}


@torch.no_grad()
def eval_epoch(model, loader: DataLoader, criterion, device, mode: str, pad_id: int) -> Dict[str, float]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    for batch in pbar:
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)
        
        if mode == "char":
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            logits = model(input_ids, input_mask, target_ids, target_mask)
        else:  # blt
            patches = batch["patches"]
            patch_mask = batch["patch_mask"].to(device)
            logits = model(patches, patch_mask, target_ids, target_mask)
        
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), target_ids.reshape(B * T))
        mask_flat = target_mask.reshape(B * T)
        loss = (loss * mask_flat.float()).sum() / mask_flat.sum()
        
        total_loss += loss.item() * mask_flat.sum().item()
        total_tokens += mask_flat.sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return {"loss": total_loss / total_tokens if total_tokens > 0 else 0.0}


def main():
    parser = argparse.ArgumentParser(description="Train BLT or Baseline model")
    parser.add_argument("--mode", type=str, default="blt", choices=["blt", "char"], help="Model mode")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Training CSV path")
    parser.add_argument("--val_csv", type=str, default=None, help="Validation CSV path (optional)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--force_preprocess", action="store_true", help="Force reprocessing data")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Preprocess data
    print(f"\n{'='*60}")
    print(f"Preprocessing data in {args.mode} mode...")
    print(f"{'='*60}")
    
    train_pt = f"data/processed/train_{args.mode}.pt"
    tokenizer_path = f"data/processed/tokenizer_{args.mode}.json"
    
    train_pt, train_manifest = preprocess_csv_to_pt(
        csv_path=args.train_csv,
        out_pt=train_pt,
        mode=args.mode,
        tokenizer=None,
        W=10,
        entropy_threshold=2.0,
        max_patch_len=15,
        buckets=4096,
        seed=1337,
        ngrams=(1, 2, 3),
        force_rebuild=args.force_preprocess,
        manifest_out=f"data/processed/train_{args.mode}_manifest.json",
        tokenizer_out=tokenizer_path,
        strict_ascii=False
    )
    
    val_pt = None
    if args.val_csv:
        val_pt = f"data/processed/val_{args.mode}.pt"
        val_pt, _ = preprocess_csv_to_pt(
            csv_path=args.val_csv,
            out_pt=val_pt,
            mode=args.mode,
            tokenizer=Tokenizer.load(tokenizer_path),
            W=10,
            entropy_threshold=2.0,
            max_patch_len=15,
            buckets=4096,
            seed=1337,
            ngrams=(1, 2, 3),
            force_rebuild=args.force_preprocess,
            manifest_out=f"data/processed/val_{args.mode}_manifest.json",
            tokenizer_out=tokenizer_path,
            strict_ascii=False
        )
    
    # Load tokenizer
    tokenizer = Tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"PAD ID: {pad_id}")
    
    # Create dataloaders
    train_loader, val_loader = make_dataloaders(
        train_pt=train_pt,
        val_pt=val_pt,
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pad_id=pad_id,
        shuffle=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n{'='*60}")
    print(f"Creating {args.mode.upper()} model...")
    print(f"{'='*60}")
    
    if args.mode == "char":
        model = BaselineModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_len=1024
        )
    else:  # blt
        model = BLTModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_patches=512,
            buckets=4096,
            ngrams=(1, 2, 3)
        )
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=pad_id)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, args.mode, pad_id)
        history["train_loss"].append(train_metrics["loss"])
        
        # Validate
        if val_loader:
            val_metrics = eval_epoch(model, val_loader, criterion, device, args.mode, pad_id)
            history["val_loss"].append(val_metrics["loss"])
            
            print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_path = os.path.join(args.save_dir, f"best_{args.mode}_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "args": vars(args)
                }, checkpoint_path)
                print(f"✓ Saved best model to {checkpoint_path}")
        else:
            print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        
        epoch_time = time.time() - epoch_start
        history["epoch_times"].append(epoch_time)
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"latest_{args.mode}_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "args": vars(args)
        }, checkpoint_path)
    
    # Save training history
    history_path = os.path.join(args.save_dir, f"history_{args.mode}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"History saved to {history_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
