#!/usr/bin/env python3
"""
Training script for BLT and Baseline models with multi-GPU support and comprehensive logging.

Usage:
    python src/train.py --mode blt --epochs 10 --batch_size 32
    python src/train.py --mode char --epochs 10 --batch_size 32
    
Multi-GPU usage:
    python -m torch.distributed.launch --nproc_per_node=2 src/train.py --mode blt --epochs 10
"""

import argparse
import os
import sys
import json
import time
import math
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

# Import our modules
from tokenizer import Tokenizer
from baseline_model import BaselineModel
from blt_model import BLTModel
from dataset import preprocess_csv_to_pt, BLTProcessedDataset, CharProcessedDataset, collate_fn, make_dataloaders

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(save_dir: str, mode: str, rank: int = 0) -> logging.Logger:
    """Setup comprehensive logging."""
    if rank == 0:  # Only log from main process
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f"training_{mode}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        return logger
    else:
        # Silent logger for non-main processes
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
        return logger


def setup_directories(save_dir: str) -> None:
    """Create necessary directories."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def get_device(local_rank: int = 0) -> torch.device:
    """Get the best available device with multi-GPU support."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        if local_rank == 0:  # Only print from main process
            print(f"Using CUDA device: {torch.cuda.get_device_name(local_rank)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
            if torch.cuda.device_count() > 1:
                print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        if local_rank == 0:
            print("Using CPU")
    return device


def create_tokenizer(mode: str) -> Tokenizer:
    """Create or load tokenizer."""
    tokenizer_path = f"data/processed/tokenizer_{mode}.json"
    
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.load(tokenizer_path)
    else:
        print(f"Creating new tokenizer")
        tokenizer = Tokenizer()
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
    
    return tokenizer


def preprocess_data(mode: str, tokenizer: Tokenizer, force_rebuild: bool = False, logger: Optional[logging.Logger] = None) -> Tuple[str, Optional[str]]:
    """Preprocess training and validation data."""
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training data not found: {train_csv}")
    
    # Process training data
    train_pt = f"data/processed/train_{mode}.pt"
    if logger:
        logger.info(f"Preprocessing training data for {mode} mode...")
    else:
        print(f"Preprocessing training data for {mode} mode...")
        
    train_pt, _ = preprocess_csv_to_pt(
        csv_path=train_csv,
        out_pt=train_pt,
        mode=mode,
        tokenizer=tokenizer,
        W=10,
        entropy_threshold=2.0,
        max_patch_len=15,
        buckets=4096,
        seed=1337,
        ngrams=(1, 2, 3),
        force_rebuild=force_rebuild,
        manifest_out=f"data/processed/train_{mode}_manifest.json",
        tokenizer_out=f"data/processed/tokenizer_{mode}.json",
        strict_ascii=False
    )
    
    # Process validation data if available
    val_pt = None
    if os.path.exists(test_csv):
        val_pt = f"data/processed/val_{mode}.pt"
        if logger:
            logger.info(f"Preprocessing validation data for {mode} mode...")
        else:
            print(f"Preprocessing validation data for {mode} mode...")
            
        val_pt, _ = preprocess_csv_to_pt(
            csv_path=test_csv,
            out_pt=val_pt,
            mode=mode,
            tokenizer=tokenizer,
            W=10,
            entropy_threshold=2.0,
            max_patch_len=15,
            buckets=4096,
            seed=1337,
            ngrams=(1, 2, 3),
            force_rebuild=force_rebuild,
            manifest_out=f"data/processed/val_{mode}_manifest.json",
            tokenizer_out=f"data/processed/tokenizer_{mode}.json",
            strict_ascii=False
        )
    
    return train_pt, val_pt


def create_model(mode: str, vocab_size: int, args: argparse.Namespace) -> nn.Module:
    """Create model based on mode and arguments."""
    if mode == "char":
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
    elif mode == "blt":
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
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return model


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    """Create optimizer."""
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace, steps_per_epoch: int) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    if args.lr_scheduler == "none":
        return None
    elif args.lr_scheduler == "cosine":
        total_steps = args.epochs * steps_per_epoch
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif args.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)
    elif args.lr_scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")


def compute_loss(model: nn.Module, batch: Dict[str, Any], mode: str, pad_id: int, device: torch.device) -> torch.Tensor:
    """Compute loss for a batch."""
    target_ids = batch["target_ids"].to(device, non_blocking=True)
    target_mask = batch["target_mask"].to(device, non_blocking=True)
    
    # Prepare inputs and targets for teacher forcing
    input_ids = target_ids[:, :-1]  # Remove last token
    target_ids = target_ids[:, 1:]  # Remove first token (SOS)
    input_mask = target_mask[:, :-1]
    target_mask = target_mask[:, 1:]
    
    if mode == "char":
        src_ids = batch["input_ids"].to(device, non_blocking=True)
        src_mask = batch["input_mask"].to(device, non_blocking=True)
        logits = model(src_ids, src_mask, input_ids, input_mask)
    elif mode == "blt":
        patches = batch["patches"]
        patch_mask = batch["patch_mask"].to(device, non_blocking=True)
        logits = model(patches, patch_mask, input_ids, input_mask)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute cross-entropy loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='mean')
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
    
    return loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    mode: str,
    pad_id: int,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    rank: int = 0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """Train for one epoch with multi-GPU support."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Only show progress bar on main process
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        try:
            if args.use_amp and scaler is not None:
                with autocast():
                    loss = compute_loss(model, batch, mode, pad_id, device)
                scaler.scale(loss).backward()
                
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_loss(model, batch, mode, pad_id, device)
                loss.backward()
                
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
            
            if scheduler is not None and args.lr_scheduler != "plateau":
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar only on main process
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            # Log detailed metrics every 100 batches
            if logger and batch_idx % 100 == 0 and rank == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if logger:
                    logger.warning(f"OOM error at batch {batch_idx}, skipping...")
                else:
                    print(f"OOM error at batch {batch_idx}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    # Synchronize losses across processes for distributed training
    if torch.distributed.is_initialized():
        loss_tensor = torch.tensor(total_loss, device=device)
        batch_tensor = torch.tensor(num_batches, device=device)
        torch.distributed.all_reduce(loss_tensor)
        torch.distributed.all_reduce(batch_tensor)
        total_loss = loss_tensor.item() / torch.distributed.get_world_size()
        num_batches = batch_tensor.item() / torch.distributed.get_world_size()
    
    return {
        "train_loss": total_loss / num_batches if num_batches > 0 else float('inf'),
        "lr": optimizer.param_groups[0]["lr"]
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    mode: str,
    pad_id: int,
    device: torch.device,
    args: argparse.Namespace,
    rank: int = 0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """Validate for one epoch with multi-GPU support."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
        
        for batch in pbar:
            try:
                if args.use_amp:
                    with autocast():
                        loss = compute_loss(model, batch, mode, pad_id, device)
                else:
                    loss = compute_loss(model, batch, mode, pad_id, device)
                
                total_loss += loss.item()
                num_batches += 1
                
                if rank == 0 and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if logger:
                        logger.warning(f"OOM error in validation, skipping batch...")
                    else:
                        print(f"OOM error in validation, skipping batch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    # Synchronize validation losses across processes
    if torch.distributed.is_initialized():
        loss_tensor = torch.tensor(total_loss, device=device)
        batch_tensor = torch.tensor(num_batches, device=device)
        torch.distributed.all_reduce(loss_tensor)
        torch.distributed.all_reduce(batch_tensor)
        total_loss = loss_tensor.item() / torch.distributed.get_world_size()
        num_batches = batch_tensor.item() / torch.distributed.get_world_size()
    
    return {
        "val_loss": total_loss / num_batches if num_batches > 0 else float('inf')
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    save_path: str,
    is_best: bool = False
) -> None:
    """Save model checkpoint with DDP support."""
    # Extract the actual model from DDP wrapper if needed
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "args": vars(args)
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace(".pt", "_best.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
    device: torch.device
) -> Tuple[int, float]:
    """Load model checkpoint with DDP support."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP wrapped models
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float('inf'))
    
    return epoch, loss


def main():
    parser = argparse.ArgumentParser(description="Train BLT or Baseline model with multi-GPU support")
    
    # Model and training args
    parser.add_argument("--mode", type=str, required=True, choices=["blt", "char"], help="Model mode")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Model architecture args
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training options
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild preprocessed data")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank
    
    # Setup device and logging
    device = get_device(local_rank)
    setup_directories(args.save_dir)
    logger = setup_logging(args.save_dir, args.mode, rank)
    
    if rank == 0:
        logger.info(f"Training {args.mode.upper()} model")
        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create tokenizer (only on main process to avoid conflicts)
    if rank == 0:
        tokenizer = create_tokenizer(args.mode)
        logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Preprocess data
        train_pt, val_pt = preprocess_data(args.mode, tokenizer, args.force_rebuild, logger)
    
    # Synchronize all processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Load tokenizer on all processes
    if rank != 0:
        tokenizer_path = f"data/processed/tokenizer_{args.mode}.json"
        tokenizer = Tokenizer.load(tokenizer_path)
        train_pt = f"data/processed/train_{args.mode}.pt"
        val_pt = f"data/processed/val_{args.mode}.pt" if os.path.exists("data/test.csv") else None
    
    # Create datasets
    if args.mode == "blt":
        train_ds = BLTProcessedDataset(train_pt)
        val_ds = BLTProcessedDataset(val_pt) if val_pt else None
    else:
        train_ds = CharProcessedDataset(train_pt)
        val_ds = CharProcessedDataset(val_pt) if val_pt else None
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if val_ds and world_size > 1 else None
    
    # Create data loaders with distributed support
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, mode=args.mode, pad_id=tokenizer.pad_id),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else 2
    )
    
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_fn(b, mode=args.mode, pad_id=tokenizer.pad_id),
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else 2
        )
    
    if rank == 0:
        logger.info(f"Training batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = create_model(args.mode, tokenizer.vocab_size, args)
    model = model.to(device)
    
    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    if rank == 0:
        # Count parameters (use .module for DDP wrapped models)
        param_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in param_model.parameters())
        trainable_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp and device.type == "cuda" else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "epoch_times": []
    }
    
    # Early stopping
    patience_counter = 0
    
    if rank == 0:
        logger.info(f"Starting training from epoch {start_epoch}...")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            # Set epoch for distributed sampler
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                args.mode, tokenizer.pad_id, device, epoch, args, rank, logger
            )
            
            # Validate
            val_metrics = {}
            if val_loader:
                val_metrics = validate_epoch(
                    model, val_loader, args.mode, tokenizer.pad_id, device, args, rank, logger
                )
                
                # Update scheduler if using plateau
                if scheduler and args.lr_scheduler == "plateau":
                    scheduler.step(val_metrics["val_loss"])
            
            epoch_time = time.time() - epoch_start_time
            
            # Only save history and checkpoints on main process
            if rank == 0:
                # Update history
                history["train_loss"].append(train_metrics["train_loss"])
                history["val_loss"].append(val_metrics.get("val_loss", 0.0))
                history["lr"].append(train_metrics["lr"])
                history["epoch_times"].append(epoch_time)
                
                # Log epoch summary
                logger.info(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
                logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
                if val_metrics:
                    logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
                logger.info(f"Learning Rate: {train_metrics['lr']:.2e}")
                
                # Save checkpoint
                current_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
                is_best = current_loss < best_val_loss
                
                if is_best:
                    best_val_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save regular checkpoint
                checkpoint_path = os.path.join(args.save_dir, f"{args.mode}_epoch_{epoch+1}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch, current_loss, args, checkpoint_path, is_best)
                
                # Save best model with standard name expected by evaluation
                if is_best:
                    best_model_path = os.path.join(args.save_dir, f"best_{args.mode}_model.pt")
                    save_checkpoint(model, optimizer, scheduler, epoch, current_loss, args, best_model_path, False)
                    logger.info(f"New best model saved: {best_model_path}")
                
                # Save history (required by run_experiments.py)
                history_path = os.path.join(args.save_dir, f"history_{args.mode}.json")
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)
                
                # Early stopping
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break
            
            # Synchronize all processes before next epoch
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
    
    except KeyboardInterrupt:
        if rank == 0:
            logger.info("Training interrupted by user")
    except Exception as e:
        if rank == 0:
            logger.error(f"Training failed with error: {e}")
        raise
    
    if rank == 0:
        logger.info(f"Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Final model save
        final_model_path = os.path.join(args.save_dir, f"final_{args.mode}_model.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, current_loss, args, final_model_path, False)
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training history saved to: {history_path}")
    
    # Cleanup distributed training
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()